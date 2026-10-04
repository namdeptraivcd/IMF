"""Equal-grid Multi-Trace objective and 1..K sampling from Untitled0.ipynb."""
import torch
from trace_imf import TraceIMF


class MultiTraceIMF(TraceIMF):
    def __init__(self, max_nfe=5, jvp_precision="bf16", channels_last=True, **kwargs):
        super().__init__(**kwargs)
        if max_nfe < 1 or jvp_precision not in ("fp32", "bf16"):
            raise ValueError("Invalid max_nfe or JVP precision")
        self.max_nfe = max_nfe
        self.jvp_precision = jvp_precision
        self.channels_last = channels_last

    def draw_interval(self, device, generator=None, nfe=None):
        # One K and j for the entire effective batch, including all microbatches.
        k = int(torch.randint(1, self.max_nfe + 1, (), device=device, generator=generator)) if nfe is None else nfe
        if not 1 <= k <= self.max_nfe:
            raise ValueError("NFE outside the trained range")
        j = int(torch.randint(k, (), device=device, generator=generator))
        return k, j

    def loss(self, model, x, labels=None, *, interval=None, nfe=None, generator=None,
             t=None, r=None, noise=None, derivative_model=None):
        if t is None:
            k, j = interval or self.draw_interval(x.device, generator, nfe)
            r = torch.full((len(x),), j / k, device=x.device)
            t = r + torch.rand(len(x), device=x.device, generator=generator) / k
        elif r is None:
            raise ValueError("Explicit Multi-Trace t requires explicit r")
        return super().loss(model, x, labels, derivative_model=derivative_model,
                            t=t, r=r, noise=noise, generator=generator)

    @torch.no_grad()
    def sample(self, model, n_samples=None, labels=None, *, device=None, noise=None, nfe=1):
        if not 1 <= nfe <= self.max_nfe:
            raise ValueError("NFE outside the trained range")
        device = device or next(model.parameters()).device
        if labels is None:
            raise ValueError("Class-conditioned sampling requires labels")
        labels = labels.to(device)
        z = (torch.randn(len(labels), self.channels, self.image_size, self.image_size, device=device)
             if noise is None else noise.to(device).clone())
        if len(z) != len(labels):
            raise ValueError("Noise and labels must have the same batch size")
        if self.channels_last:
            z = z.contiguous(memory_format=torch.channels_last)
        was_training = model.training
        model.eval()
        try:
            for index in range(nfe, 0, -1):
                t = torch.full((len(z),), index / nfe, device=device)
                r = torch.full_like(t, (index - 1) / nfe)
                z = z - model(z, t, r, y=labels, use_flash_attention=False) / nfe
            return (z.clamp(-1, 1) + 1) * 0.5
        finally:
            model.train(was_training)
