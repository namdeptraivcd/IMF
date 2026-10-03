# Adapted from haidog-yaqub/MeanFlow, MIT; see third_party/MeanFlow.LICENSE.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class RMSNorm(nn.Module):
    """RMSNorm compatible with PyTorch 2.2 and forward-mode differentiation."""
    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        if self.weight is not None:
            value = value * self.weight
        return value.to(dtype)


class PatchEmbed(nn.Module):
    """Non-overlapping convolutional patch projection, equivalent to source timm usage."""
    def __init__(self, input_size, patch_size, in_channels, dim):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x):
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size} input, got {x.shape[-2:]}")
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Sequential):
    def __init__(self, in_features, hidden_features, act_layer, drop=0):
        super().__init__(nn.Linear(in_features, hidden_features), act_layer(),
                         nn.Linear(hidden_features, in_features))


def expand_ada_params(param, x):
    """Expand global (B, D) adaLN params to (B, T, D); pass through token-level params."""
    if param.ndim == 2:
        return param.unsqueeze(1)
    return param


def modulate(x, scale, shift):
    """
    Apply adaLN modulation.
    x: (B, T, D)
    scale, shift: (B, D) for global adaLN or (B, T, D) for token-level adaLN
    """
    scale = expand_ada_params(scale, x)
    shift = expand_ada_params(shift, x)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256, scale=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq
        self.scale = scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * self.scale
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def _native_attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ v

    def _flash_attention(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def forward(self, x, use_flash_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if (
            use_flash_attention
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            x = self._flash_attention(q, k, v)
        else:
            x = self._native_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, use_flash_attention=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + expand_ada_params(gate_msa, x) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa),
            use_flash_attention=use_flash_attention,
        )
        x = x + expand_ada_params(gate_mlp, x) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale, shift)
        x = self.linear(x)
        return x


class TraceDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        if input_size <= 0 or patch_size <= 0 or input_size % patch_size:
            raise ValueError("input_size must be positive and divisible by patch_size")
        if dim <= 0 or dim % 4 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by 4 and num_heads")
        if depth <= 0 or mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive")
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        self.use_cond = num_classes is not None
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None

        num_patches = self.x_embedder.num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, dim))

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP (t, r):
        for embedder in (self.t_embedder, self.r_embedder):
            embedder.initialize_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        for final_layer in (self.final_layer,):
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None, use_flash_attention=False):
        """Return u(z,r,t); API preserves source order (x,t,r).

        x: (B,C,H,W); t,r: (B,); y: class indices when conditioned.
        The diagonal velocity uses this same head with r=t.
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.r_embedder(r)
        if self.use_cond:
            if y is None:
                raise ValueError("Class-conditioned model requires labels y")
            c = c + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c, use_flash_attention=use_flash_attention)
        return self.unpatchify(self.final_layer(x, c))


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
