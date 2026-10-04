"""Conditional U-Net adapted from the user-provided Untitled0.ipynb.

Architecture lineage: base channels 44, (1,2,2,4), one residual block per level.
Uses the same (z,t,r,y) API as TraceDiT so training and Hub sampling share a path.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def gn_groups(ch, max_groups=8):
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class JVPGroupNorm(nn.GroupNorm):
    def forward(self, x):
        # PyTorch 2.2 forward AD can fail on channels-last GroupNorm strides.
        # Normalize a contiguous view; convolutions still use channels-last.
        return super().forward(x.contiguous())


class ScalarEmbed(nn.Module):
    """Sinusoidal scalar embedding with cached frequencies."""
    def __init__(self, out_dim, fourier_dim=64):
        super().__init__()
        self.fourier_dim = fourier_dim
        half = fourier_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32) / max(1, half)
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        args = x.float()[:, None] * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.fourier_dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = JVPGroupNorm(gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, 2 * out_ch)
        self.norm2 = JVPGroupNorm(gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(F.silu(c)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class AttentionBlock(nn.Module):
    """Static math attention path: forward-mode-AD safe and compile-friendly."""
    def __init__(self, ch, max_heads=4):
        super().__init__()
        heads = min(max_heads, ch)
        while ch % heads:
            heads -= 1
        self.heads = heads
        self.norm = JVPGroupNorm(gn_groups(ch), ch)
        self.qkv = nn.Conv1d(ch, 3 * ch, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        L = h * w
        z = self.norm(x).reshape(b, c, L)
        q, k, v = self.qkv(z).chunk(3, dim=1)
        d = c // self.heads
        q = q.reshape(b, self.heads, d, L).transpose(-1, -2)
        k = k.reshape(b, self.heads, d, L).transpose(-1, -2)
        v = v.reshape(b, self.heads, d, L).transpose(-1, -2)
        out = torch.softmax((q @ k.transpose(-1, -2)) * (d ** -0.5), dim=-1) @ v
        out = self.proj(out.transpose(-1, -2).reshape(b, c, L)).reshape(b, c, h, w)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ConditionalMultiTraceUNet(nn.Module):
    """One shared conditional field u_theta(z,r,t,y)."""
    def __init__(self, input_size=32, in_channels=3, num_classes=10,
                 base_channels=84, channel_mult=(1, 2, 2, 4), num_res_blocks=1,
                 cond_dim=356, attn_resolutions=(8,), fourier_dim=64):
        super().__init__()
        cfg = dict(image_size=input_size, num_classes=num_classes,
                   base_channels=base_channels, channel_mult=channel_mult,
                   num_res_blocks=num_res_blocks, cond_dim=cond_dim,
                   attn_resolutions=attn_resolutions, fourier_dim=fourier_dim)
        base = cfg["base_channels"]
        mults = cfg["channel_mult"]
        nrb = cfg["num_res_blocks"]
        cd = cfg["cond_dim"]

        self.attn_res = set(cfg["attn_resolutions"])
        self.t_emb = ScalarEmbed(cd, cfg["fourier_dim"])
        self.r_emb = ScalarEmbed(cd, cfg["fourier_dim"])
        self.y_emb = nn.Embedding(cfg["num_classes"], cd)
        nn.init.normal_(self.y_emb.weight, std=0.02)
        self.cond_out = nn.Sequential(nn.SiLU(), nn.Linear(cd, cd))

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down_levels = nn.ModuleList()
        ch = base
        res = cfg["image_size"]
        skip_channels = [ch]

        for li, m in enumerate(mults):
            out_ch = base * m
            blocks = nn.ModuleList()
            atts = nn.ModuleList()
            for _ in range(nrb):
                blocks.append(ResBlock(ch, out_ch, cd))
                ch = out_ch
                atts.append(AttentionBlock(ch) if res in self.attn_res else nn.Identity())
                skip_channels.append(ch)
            down = Downsample(ch) if li != len(mults) - 1 else nn.Identity()
            self.down_levels.append(nn.ModuleDict({"blocks": blocks, "atts": atts, "down": down}))
            if li != len(mults) - 1:
                skip_channels.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, cd)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, cd)

        self.up_levels = nn.ModuleList()
        stack = list(skip_channels)
        res = cfg["image_size"] // (2 ** (len(mults) - 1))
        for rev_idx, m in enumerate(reversed(mults)):
            li = len(mults) - 1 - rev_idx
            out_ch = base * m
            blocks = nn.ModuleList()
            atts = nn.ModuleList()
            for _ in range(nrb + 1):
                skip_ch = stack.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, cd))
                ch = out_ch
                atts.append(AttentionBlock(ch) if res in self.attn_res else nn.Identity())
            up = Upsample(ch) if li != 0 else nn.Identity()
            self.up_levels.append(nn.ModuleDict({"blocks": blocks, "atts": atts, "up": up}))
            if li != 0:
                res *= 2

        if stack:
            raise RuntimeError(f"Unused build-time skips: {stack}")

        self.out_norm = JVPGroupNorm(gn_groups(ch), ch)
        self.u_head = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.u_head.weight)
        nn.init.zeros_(self.u_head.bias)

    def forward(self, x, t, r, y=None, use_flash_attention=False):
        # Reference notebook uses (x,r,t,y); repository API uses (x,t,r,y).
        if y is None:
            raise ValueError("This U-Net requires class labels")
        c = self.cond_out(self.t_emb(t) + self.r_emb(r) + self.y_emb(y))
        hs = []
        x = self.in_conv(x)
        hs.append(x)

        for level in self.down_levels:
            for block, attn in zip(level["blocks"], level["atts"]):
                x = attn(block(x, c))
                hs.append(x)
            if not isinstance(level["down"], nn.Identity):
                x = level["down"](x)
                hs.append(x)

        x = self.mid2(self.mid_attn(self.mid1(x, c)), c)

        for level in self.up_levels:
            for block, attn in zip(level["blocks"], level["atts"]):
                x = attn(block(torch.cat([x, hs.pop()], dim=1), c))
            x = level["up"](x)

        if hs:
            raise RuntimeError(f"Unconsumed runtime skips: {len(hs)}")
        return self.u_head(F.silu(self.out_norm(x)))
