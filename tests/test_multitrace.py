import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.multitrace_cifar10_15m import config, scale_training_budget
from configs.multitrace_cifar10_22m import config as config_22m
from data import ResidentCIFARBatcher
from ema import ModelEMA
from hub import export_model
from models import build_backbone
from monitoring import evaluate
from multi_trace_imf import MultiTraceIMF
from train import run_training


class LinearField(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(-0.1))
        self.calls = []

    def forward(self, z, t, r, y=None, use_flash_attention=False):
        self.calls.append((t.detach().clone(), r.detach().clone()))
        return self.a * z + self.b * t[:, None, None, None] + self.c * r[:, None, None, None]


class MultiTraceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_reference_and_scaled_parameter_counts(self):
        with torch.device("meta"):
            scaled = build_backbone(config)
            scaled_22m = build_backbone(config_22m)
            original = copy.deepcopy(config)
            original["model"].update(base_channels=44, cond_dim=160)
            reference = build_backbone(original)
        self.assertEqual(sum(p.numel() for p in reference.parameters()), 5_946_579)
        self.assertEqual(sum(p.numel() for p in scaled.parameters()), 14_992_182)
        self.assertEqual(sum(p.numel() for p in scaled_22m.parameters()), 22_002_655)
        self.assertEqual(config["batch_size"], 512)
        self.assertEqual(config["gradient_accumulation_steps"], 1)

    def test_image_budget_and_ema_horizon_scale_with_effective_batch(self):
        base = scale_training_budget(128, 4)
        half = scale_training_budget(64, 4)
        self.assertEqual(base["n_steps"], 100_000)
        self.assertEqual(half["n_steps"], 200_000)
        self.assertEqual(half["warmup_steps"], 10_000)
        self.assertAlmostEqual(half["ema_decay"] ** 2, base["ema_decay"])
        self.assertEqual(scale_training_budget(64, 8), base | {"batch_size": 64, "gradient_accumulation_steps": 8})

    def test_nonzero_interval_jvp_and_semigradient(self):
        model = LinearField()
        objective = MultiTraceIMF(channels=1, image_size=2, num_classes=2, jvp_precision="fp32")
        x, noise = torch.rand(2, 1, 2, 2), torch.randn(2, 1, 2, 2)
        t, r, labels = torch.tensor([0.6, 0.8]), torch.tensor([0.25, 0.5]), torch.tensor([0, 1])
        actual, _ = objective.loss(model, x, labels, t=t, r=r, noise=noise)
        actual.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        ti, ri = t[:, None, None, None], r[:, None, None, None]
        real = x * 2 - 1
        z = (1 - ti) * real + ti * noise
        diagonal = model.a * z + (model.b + model.c) * ti
        edge = model.a * z + model.b * ti + model.c * ri
        derivative = model.a.detach() * diagonal.detach() + model.b.detach()
        expected = ((edge + (ti - ri) * derivative - (noise - real)) ** 2).mean()
        expected += ((diagonal - (noise - real)) ** 2).mean()
        torch.testing.assert_close(actual, expected)
        expected.backward()
        for p, grad in zip(model.parameters(), gradients):
            torch.testing.assert_close(p.grad, grad)

    def test_sampling_uses_exact_grid_nfe_and_keeps_input_noise(self):
        model = LinearField()
        objective = MultiTraceIMF(channels=1, image_size=2, num_classes=2)
        noise = torch.randn(2, 1, 2, 2)
        original = noise.clone()
        labels = torch.tensor([0, 1])
        for k in range(1, 6):
            model.calls.clear()
            images = objective.sample(model, labels=labels, noise=noise, nfe=k)
            self.assertEqual(len(model.calls), k)
            for i, (t, r) in enumerate(model.calls):
                torch.testing.assert_close(t, torch.full_like(t, (k - i) / k))
                torch.testing.assert_close(r, torch.full_like(r, (k - i - 1) / k))
            self.assertTrue(model.training)
            self.assertTrue((images >= 0).all() and (images <= 1).all())
            torch.testing.assert_close(noise, original)

    def test_channels_last_unet_jvp_matches_finite_difference(self):
        cfg = copy.deepcopy(config)
        cfg["model"].update(input_size=8, in_channels=1, num_classes=2, base_channels=4,
                            channel_mult=(1, 2), cond_dim=16, attn_resolutions=(4,), fourier_dim=8)
        model = build_backbone(cfg).to(memory_format=torch.channels_last)
        # Avoid the zero head making the directional derivative trivially zero.
        with torch.no_grad():
            model.u_head.weight.normal_(0, 0.01)
        x = torch.randn(2, 1, 8, 8).contiguous(memory_format=torch.channels_last)
        t, r, labels = torch.full((2,), 0.6), torch.full((2,), 0.2), torch.tensor([0, 1])
        v = torch.randn_like(x)
        def field(z, time):
            return model(z, time, r, y=labels)
        with torch.no_grad():
            _, derivative = torch.func.jvp(field, (x, t), (v, torch.ones_like(t)))
            epsilon = 1e-3
            difference = (field(x + epsilon*v, t + epsilon) - field(x - epsilon*v, t - epsilon)) / (2*epsilon)
        torch.testing.assert_close(derivative, difference, rtol=0.02, atol=2e-4)

    def test_fixed_validation_covers_each_k_without_training_rng(self):
        model = LinearField()
        objective = MultiTraceIMF(channels=1, image_size=2, num_classes=2, jvp_precision="fp32")
        loader = DataLoader(TensorDataset(torch.rand(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])),
            batch_size=2, generator=torch.Generator().manual_seed(9))
        before = torch.get_rng_state().clone()
        first = evaluate(model, objective, loader, torch.device("cpu"), 99, 2)
        second = evaluate(model, objective, loader, torch.device("cpu"), 99, 2)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertAlmostEqual(first["loss"], sum(first[f"loss_nfe_{k}"] for k in range(1, 6)) / 5)

    def test_resident_batcher_restores_the_next_batch_and_augmentation(self):
        dataset = SimpleNamespace(data=np.arange(8*2*2*3, dtype=np.uint8).reshape(8, 2, 2, 3), targets=list(range(8)))
        batcher = ResidentCIFARBatcher(dataset, 2, "cpu")
        next(batcher)
        state, rng = batcher.state_dict(), torch.get_rng_state()
        first = next(batcher)
        restored = ResidentCIFARBatcher(dataset, 2, "cpu", state=state)
        torch.set_rng_state(rng)
        second = next(restored)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_accumulation_matches_full_batch_gradient_and_ema_once(self):
        model = LinearField()
        reference = copy.deepcopy(model)
        objective = MultiTraceIMF(channels=1, image_size=2, num_classes=2, jvp_precision="fp32")
        x, noise = torch.rand(4, 1, 2, 2), torch.randn(4, 1, 2, 2)
        t, r, labels = torch.full((4,), 0.7), torch.full((4,), 0.5), torch.tensor([0, 1, 0, 1])
        loss, _ = objective.loss(reference, x, labels, t=t, r=r, noise=noise)
        loss.backward()
        for i in (0, 2):
            sl = slice(i, i+2)
            loss, _ = objective.loss(model, x[sl], labels[sl], t=t[sl], r=r[sl], noise=noise[sl])
            (loss / 2).backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        model.calls.clear()  # discard the test spy's temporary forward-AD wrappers
        ema = ModelEMA(model, 0.9)
        before = copy.deepcopy(model.state_dict())
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        ema.update(model)
        self.assertEqual(ema.updates, 1)
        for name, value in ema.model.state_dict().items():
            torch.testing.assert_close(value, before[name] * 0.9 + model.state_dict()[name] * 0.1)

    def test_unet_accumulation_resume_ema_and_export_inference(self):
        cfg = copy.deepcopy(config)
        cfg.update(n_steps=4, warmup_steps=1, batch_size=2, gradient_accumulation_steps=2,
                   num_workers=0, image_size=8, log_step=1, sample_step=4, checkpoint_step=1,
                   keep_last_checkpoints=3, ema_decay=0.9, mixed_precision="no", preload_gpu=False,
                   monitoring=dict(enabled=True, tensorboard=False, validation_every=4,
                                   validation_batches=1, plot_every=4, save_best=True, validate_raw=True))
        cfg["model"].update(input_size=8, in_channels=1, num_classes=2, base_channels=4,
                            channel_mult=(1, 2), cond_dim=16, attn_resolutions=(4,), fourier_dim=8)
        cfg["multi_trace_imf"]["jvp_precision"] = "fp32"
        model = build_backbone(cfg)
        count = sum(p.numel() for p in model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            args = SimpleNamespace(fake_data=True, output_dir=folder, run_suffix="proof",
                                   run_dir=str(run_dir), resume=None, stop_after=2)
            run_training(model, count, cfg, args)
            cp = run_dir / "ckpts/step_0000002.pt"
            initial = torch.load(cp, weights_only=False)
            self.assertEqual(initial["ema_updates"], 2)
            args.resume, args.stop_after = str(cp), None
            run_training(build_backbone(cfg), count, cfg, args)
            final_path = run_dir / "ckpts/step_0000004.pt"
            final = torch.load(final_path, weights_only=False)
            self.assertEqual(final["ema_updates"], 4)
            self.assertEqual(final["scheduler"]["last_epoch"], 4)
            middle = torch.load(run_dir / "ckpts/step_0000003.pt", weights_only=False)
            for name in middle["ema"]:
                torch.testing.assert_close(middle["ema"][name], initial["ema"][name] * 0.9 + middle["model"][name] * 0.1)
            self.assertTrue(all(state["step"].item() == 4 for state in final["optimizer"]["state"].values()))
            rows = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
            self.assertEqual([r["images_seen"] for r in rows if "step" in r], [4, 8, 12, 16])
            self.assertTrue((run_dir / "diagnostics/multitrace.png").is_file())
            validation = [json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines()]
            self.assertIn("raw_loss_nfe_5", validation[-1])
            for k in range(2, 6):
                self.assertTrue((run_dir / "images" / f"step_0000004_nfe{k}.png").is_file())
            changed = copy.deepcopy(cfg)
            changed["gradient_accumulation_steps"] = 1
            args.resume = str(final_path)
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                run_training(build_backbone(changed), count, changed, args)
            # Mark only this fixture as real to exercise the export contract.
            final["synthetic_data"] = False
            torch.save(final, final_path)
            exported = export_model(final_path, Path(folder) / "export")
            from safetensors.torch import load_file
            weights = load_file(str(exported / "model.safetensors"))
            for name, value in final["ema"].items():
                torch.testing.assert_close(weights[name], value)
            self.assertEqual(json.loads((exported / "config.json").read_text())["weights_source"], "ema")
            import subprocess
            result = subprocess.run([sys.executable, str(exported / "sample.py"), "--model-dir",
                str(exported), "--nfe", "5", "--output", str(Path(folder) / "sample.png")],
                cwd=folder, check=True, capture_output=True, text=True)
            self.assertIn("5-NFE", result.stdout)
            self.assertTrue((Path(folder) / "sample.png").is_file())


if __name__ == "__main__":
    unittest.main()
