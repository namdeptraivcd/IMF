"""Two-trace quadratic objective from idea.pdf, sections 10 and 14."""
import torch
import torch.nn.functional as F


class TraceIMF:
    def __init__(self, channels=3, image_size=32, num_classes=10, lambda_diag=1.0):
        if lambda_diag <= 0:
            raise ValueError("lambda_diag must be positive to train both traces")
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.lambda_diag = lambda_diag

    def loss(self, model, x, labels=None, *, derivative_model=None, t=None, noise=None,
             r=None, generator=None):
        """x is raw image data in [0,1]. Optional t/noise permit exact checks.

        Use the unwrapped model for JVP under Accelerate/DDP. The single joint
        grad-enabled call through model trains both edge and diagonal safely.
        """
        b = x.shape[0]
        x = x.float() * 2 - 1
        t = torch.rand(b, device=x.device, generator=generator) if t is None else t.to(x.device).float()
        noise = torch.randn(x.shape, device=x.device, generator=generator) if noise is None else noise.to(x.device).float()
        if t.shape != (b,) or noise.shape != x.shape:
            raise ValueError("Expected t with shape (B,) and noise with the image shape")
        if self.num_classes is not None and labels is None:
            raise ValueError("Class-conditioned training requires labels")
        if self.num_classes is None:
            labels = None
        t_image = t[:, None, None, None]
        z = (1 - t_image) * x + t_image * noise
        target = noise - x
        r = torch.zeros_like(t) if r is None else r.to(x.device).float()
        if r.shape != t.shape or (r < 0).any() or (t > 1).any() or (r > t).any():
            raise ValueError("Require 0 <= r <= t <= 1, with r and t shaped (B,)")
        if getattr(self, "channels_last", False):
            z = z.contiguous(memory_format=torch.channels_last)

        # Both branches use the SAME u head, evaluated at r=0 and r=t.
        pair_labels = None if labels is None else torch.cat((labels, labels))
        predictions = model(torch.cat((z, z)), torch.cat((t, t)),
                            torch.cat((r, t)), y=pair_labels,
                            use_flash_attention=True)
        edge, diagonal = predictions.chunk(2)
        jvp_model = model if derivative_model is None else derivative_model

        # FP32 JVP: native attention supports forward AD; SDPA is used only in
        # the grad-enabled forward. No parameter gradient through the JVP.
        use_bf16 = getattr(self, "jvp_precision", "fp32") == "bf16" and x.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=use_bf16,
                                           dtype=torch.bfloat16):
            def edge_fn(z_value, t_value, r_value):
                # Accelerate may autocast the instance's forward even after
                # unwrap_model. The class forward bypasses that AMP wrapper
                # without mutating the grad-enabled training forward.
                return type(jvp_model).forward(jvp_model, z_value, t_value, r_value,
                                               y=labels, use_flash_attention=False)

            _, derivative = torch.func.jvp(
                edge_fn, (z, t, r),
                (diagonal.detach().float(), torch.ones_like(t), torch.zeros_like(r)),
            )

        compound = edge.float() + (t - r)[:, None, None, None] * derivative.detach().float()
        edge_loss = F.mse_loss(compound, target)
        diag_loss = F.mse_loss(diagonal.float(), target)
        loss = edge_loss + self.lambda_diag * diag_loss
        return loss, {"edge_loss": edge_loss.detach(), "diag_loss": diag_loss.detach(),
                      "jvp_rms": derivative.detach().square().mean().sqrt()}

    @torch.no_grad()
    def sample(self, model, n_samples=None, labels=None, *, device=None, noise=None):
        """Exactly one evaluation at (t,r)=(1,0), then map back to [0,1]."""
        was_training = model.training
        model.eval()
        try:
            if device is None:
                device = next(model.parameters()).device
            if labels is not None:
                labels = labels.to(device)
                n_samples = len(labels)
            if self.num_classes is not None and labels is None:
                raise ValueError("Class-conditioned sampling requires labels")
            if self.num_classes is None:
                labels = None
            if noise is None:
                if n_samples is None or n_samples <= 0:
                    raise ValueError("n_samples must be positive")
                noise = torch.randn(n_samples, self.channels, self.image_size,
                                    self.image_size, device=device)
            else:
                noise = noise.to(device)
            t = torch.ones(noise.shape[0], device=device)
            u = model(noise, t, torch.zeros_like(t), y=labels,
                      use_flash_attention=True)
            return ((noise - u).clamp(-1, 1) + 1) * 0.5
        finally:
            model.train(was_training)
