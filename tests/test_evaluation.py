"""Exercise final FID orchestration with fake features; never report these as quality scores."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from configs.multitrace_cifar10_22m import config
from evaluation import evaluate_checkpoint
from models import build_backbone


class FakeFeatures(nn.Module):
    pass


class SampleSpy:
    num_classes, channels, image_size, max_nfe = 2, 1, 8, 5
    def __init__(self):
        self.calls = []
    def sample(self, model, *, labels, device, noise, nfe):
        self.calls.append((nfe, labels.cpu().clone(), noise.cpu().clone()))
        return (noise.clamp(-1, 1) + 1) / 2


class EvaluationTests(unittest.TestCase):
    def test_fid_uses_ema_fixed_noise_cached_real_stats_and_retries_partial_results(self):
        cfg = copy.deepcopy(config)
        cfg.update(n_steps=4, num_workers=0)
        cfg["model"].update(input_size=8, in_channels=1, num_classes=2, base_channels=4,
                            channel_mult=(1, 2), cond_dim=16, attn_resolutions=(4,), fourier_dim=8)
        cfg["evaluation"].update(num_generated=4, batch_size=2, nfes=(1, 2))
        model = build_backbone(cfg)
        raw, ema = model.state_dict(), copy.deepcopy(model.state_dict())
        ema["u_head.bias"].fill_(0.123)
        dataset = SimpleNamespace(data=np.zeros((4, 8, 8, 1), dtype=np.uint8))
        class TinyDataset:
            data = dataset.data
            def __len__(self): return 4
            def __getitem__(self, index): return torch.zeros(1, 8, 8), index % 2
        spy = SampleSpy()
        measured_weights = []
        def statistics(extractor, batches, count, device, description):
            images = torch.cat(list(batches))
            self.assertEqual(len(images), count)
            return np.zeros(2), np.eye(2)
        def sample_objective(configuration):
            return spy
        original_sample = spy.sample
        def checked_sample(model, **kwargs):
            measured_weights.append(float(model.u_head.bias[0]))
            return original_sample(model, **kwargs)
        spy.sample = checked_sample
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ckpts").mkdir()
            path = root / "ckpts/step_0000004.pt"
            torch.save(dict(config=cfg, step=4, model=raw, ema=ema, synthetic_data=False), path)
            with patch("pytorch_fid.inception.InceptionV3", return_value=FakeFeatures()) as features, \
                 patch("pytorch_fid.fid_score.calculate_frechet_distance", side_effect=[0.5, RuntimeError("interrupted")]) as fid, \
                 patch("data.build_dataset", return_value=TinyDataset()), \
                 patch("objectives.build_objective", side_effect=sample_objective), \
                 patch("evaluation.feature_statistics", side_effect=statistics) as stats:
                features.BLOCK_INDEX_BY_DIM = {2048: 3}
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    evaluate_checkpoint(path, root / "cache")
                first = json.loads((root / "fid.json").read_text())
                self.assertEqual(first["fid"], {"1": 0.5})
                self.assertEqual(first["protocol"]["weights_source"], "ema")
                self.assertEqual(first["protocol"]["real_count"], 4)
                self.assertFalse(first["protocol"]["reference_notebook_comparable"])
                self.assertEqual(stats.call_count, 3)  # real + K1 + K2
                self.assertTrue(features.call_args.kwargs["use_fid_inception"])
                fid.side_effect = [0.75]
                stats.reset_mock()
                spy.calls.clear()
                evaluate_checkpoint(path, root / "cache")
                self.assertEqual(stats.call_count, 1)  # only unfinished K2
                self.assertEqual({call[0] for call in spy.calls}, {2})
                self.assertEqual(json.loads((root / "fid.json").read_text())["fid"], {"1": 0.5, "2": 0.75})
                self.assertTrue(all(abs(value - 0.123) < 1e-6 for value in measured_weights))
                before = len(spy.calls)
                evaluate_checkpoint(path, root / "cache")
                self.assertEqual(len(spy.calls), before)
            # Independent full evaluation verifies common random numbers across K.
            (root / "fid.json").unlink()
            spy.calls.clear()
            with patch("pytorch_fid.inception.InceptionV3", return_value=FakeFeatures()) as features, \
                 patch("pytorch_fid.fid_score.calculate_frechet_distance", return_value=0.0), \
                 patch("data.build_dataset", return_value=TinyDataset()), \
                 patch("objectives.build_objective", return_value=spy), \
                 patch("evaluation.feature_statistics", side_effect=statistics):
                features.BLOCK_INDEX_BY_DIM = {2048: 3}
                evaluate_checkpoint(path, root / "cache")
            for first, second in zip(spy.calls[:2], spy.calls[2:]):
                torch.testing.assert_close(first[1], second[1])
                torch.testing.assert_close(first[2], second[2])

    def test_fid_rejects_incomplete_or_synthetic_checkpoints_before_downloading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            for synthetic, step in ((True, 4), (False, 3)):
                torch.save(dict(config=dict(n_steps=4), step=step, synthetic_data=synthetic), path)
                with patch("pytorch_fid.inception.InceptionV3") as features:
                    with self.assertRaisesRegex(ValueError, "completed real-data"):
                        evaluate_checkpoint(path, Path(folder) / "cache")
                    features.assert_not_called()


if __name__ == "__main__":
    unittest.main()
