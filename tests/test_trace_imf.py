import unittest

import torch
from torch import nn

from configs.cifar10_22m import config
from models.dit import TraceDiT
from trace_imf import TraceIMF
from train import build_model


class AffineField(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.4))
        self.b = nn.Parameter(torch.tensor(0.7))
        self.c = nn.Parameter(torch.tensor(-0.2))
        self.calls = []

    def forward(self, z, t, r, y=None, use_flash_attention=False):
        self.calls.append((t.detach().clone(), r.detach().clone()))
        return self.a * z + self.b * t[:, None, None, None] + self.c * r[:, None, None, None]


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_quadratic_objective_jvp_and_semigradient(self):
        model = AffineField()
        objective = TraceIMF(channels=1, image_size=2, num_classes=None, lambda_diag=0.3)
        x = torch.tensor([[[[0.1, 0.3], [0.5, 0.8]]]])
        noise = torch.tensor([[[[0.7, -0.2], [0.4, -0.3]]]])
        t = torch.tensor([0.6])
        loss, metrics = objective.loss(model, x, t=t, noise=noise)
        ti = t[:, None, None, None]
        z = (1 - ti) * (2 * x - 1) + ti * noise
        target = noise - (2 * x - 1)
        # Closed-form derivative: dz/dt = u(z,t,t), dr/dt=0.
        diagonal = model.a * z + (model.b + model.c) * ti
        derivative = model.a * diagonal + model.b
        edge = model.a * z + model.b * ti
        compound = edge + ti * derivative.detach()
        expected = (compound - target).square().mean() + 0.3 * (diagonal - target).square().mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(metrics["jvp_rms"], derivative.square().mean().sqrt())
        loss.backward()
        edge_residual = (compound - target).detach()
        diag_residual = (diagonal - target).detach()
        torch.testing.assert_close(model.a.grad, 2 * (edge_residual * z).mean() + 0.6 * (diag_residual * z).mean())
        torch.testing.assert_close(model.b.grad, 2 * (edge_residual * ti).mean() + 0.6 * (diag_residual * ti).mean())
        torch.testing.assert_close(model.c.grad, 0.6 * (diag_residual * ti).mean())
        self.assertFalse(metrics["jvp_rms"].requires_grad)

    def test_sampling_one_evaluation_and_restore_mode(self):
        model = AffineField()
        objective = TraceIMF(channels=1, image_size=2, num_classes=None)
        noise = torch.full((2, 1, 2, 2), 0.1)
        samples = objective.sample(model, noise=noise, device="cpu")
        self.assertEqual(len(model.calls), 1)
        t, r = model.calls[0]
        torch.testing.assert_close(t, torch.ones(2))
        torch.testing.assert_close(r, torch.zeros(2))
        expected = ((noise - (model.a * noise + model.b)).clamp(-1, 1) + 1) / 2
        torch.testing.assert_close(samples, expected)
        self.assertTrue(model.training)

    def test_parameter_budget(self):
        model, count = build_model(config)
        self.assertEqual(count, 22_082_956)
        self.assertLess(abs(count - 22_000_000), 220_000)
        self.assertNotIn("pos_embed", dict(model.named_parameters()))
        self.assertEqual(sum(p.numel() for p in model.buffers()), 98_304)

    def test_tiny_backbone_jvp_finite_difference_and_backward(self):
        torch.manual_seed(7)
        model = TraceDiT(input_size=4, patch_size=2, in_channels=1,
                         dim=32, depth=2, num_heads=4, num_classes=2)
        # Nonzero gates/output exercise derivatives beyond zero initialization.
        with torch.no_grad():
            for block in model.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(model.final_layer.linear.weight, std=0.02)
        z = torch.randn(2, 1, 4, 4)
        t, r = torch.tensor([0.3, 0.7]), torch.zeros(2)
        labels = torch.tensor([0, 1])
        tangent = torch.randn_like(z)
        fn = lambda z, t, r: model(z, t, r, y=labels)
        _, derivative = torch.func.jvp(fn, (z, t, r), (tangent, torch.ones_like(t), torch.zeros_like(r)))
        # Time embedding is scaled by 1000: use a smaller perturbation for accuracy.
        eps = 1e-5
        finite_difference = (fn(z + eps * tangent, t + eps, r) - fn(z - eps * tangent, t - eps, r)) / (2 * eps)
        torch.testing.assert_close(derivative, finite_difference, atol=0.12, rtol=0.02)
        objective = TraceIMF(channels=1, image_size=4, num_classes=2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = objective.loss(model, torch.rand(2, 1, 4, 4), labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_jvp_bypasses_instance_autocast_wrapper(self):
        model = TraceDiT(input_size=4, patch_size=2, in_channels=1,
                         dim=32, depth=1, num_heads=4, num_classes=None)
        seen_dtypes = []
        model.final_layer.linear.register_forward_hook(
            lambda module, inputs, output: seen_dtypes.append(output.dtype))
        original_forward = model.forward

        def amp_forward(*args, **kwargs):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return original_forward(*args, **kwargs).float()

        model.forward = amp_forward
        objective = TraceIMF(channels=1, image_size=4, num_classes=None)
        loss, _ = objective.loss(model, torch.rand(2, 1, 4, 4), derivative_model=model)
        loss.backward()
        self.assertEqual(seen_dtypes, [torch.bfloat16, torch.float32])
        self.assertIs(model.forward, amp_forward)


if __name__ == "__main__":
    unittest.main()
