import copy
import json
import os
from pathlib import Path
import tempfile
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from configs.cifar10_22m import config
from hub import export_model, prepare_repository, upload_model, upload_training_checkpoint
from models.dit import TraceDiT
from train import run_training


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = copy.deepcopy(config)
        self.cfg.update(n_steps=4, batch_size=2, num_workers=0, log_step=1,
                        sample_step=4, checkpoint_step=1, keep_last_checkpoints=2,
                        mixed_precision="no", monitoring=dict(enabled=True, tensorboard=True,
                            save_best=True, validation_every=2, validation_batches=1, plot_every=2))
        self.cfg["model"].update(input_size=4, patch_size=2, in_channels=1,
                                 dim=32, depth=1, num_heads=4, num_classes=2)
        self.cfg["image_size"] = 4
        self.model = TraceDiT(**self.cfg["model"])
        self.count = sum(p.numel() for p in self.model.parameters())

    def test_nonfinite_gradients_write_failure_and_no_checkpoint(self):
        class InvalidField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, z, t, r, y=None, use_flash_attention=False):
                return z * self.weight * float("nan")
        with tempfile.TemporaryDirectory() as folder:
            cfg = copy.deepcopy(self.cfg)
            cfg["monitoring"]["enabled"] = False
            run_dir = Path(folder) / "failure-run"
            args = SimpleNamespace(fake_data=True, output_dir=folder, run_suffix="failure",
                                   run_dir=str(run_dir), resume=None, stop_after=None)
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                run_training(InvalidField(), 1, cfg, args)
            failure = json.loads((run_dir / "failure.json").read_text())
            self.assertEqual(failure["step"], 1)
            self.assertEqual(failure["nonfinite_gradient_parameters"], ["weight"])
            self.assertFalse(list((run_dir / "ckpts").glob("*.pt")))

    def test_trainer_resume_optimizer_scheduler_and_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            args = SimpleNamespace(fake_data=True, output_dir=folder, run_suffix="proof",
                                   run_dir=str(run_dir), resume=None, stop_after=2)
            run_training(self.model, self.count, self.cfg, args)
            checkpoint_path = run_dir / "ckpts/step_0000002.pt"
            first = torch.load(checkpoint_path, weights_only=False)
            self.assertEqual(first["scheduler"]["last_epoch"], 2)
            self.assertEqual(len(first["rng_states"]), 1)
            args.resume, args.stop_after = str(checkpoint_path), None
            run_training(TraceDiT(**self.cfg["model"]), self.count, self.cfg, args)
            final_path = run_dir / "ckpts/step_0000004.pt"
            final = torch.load(final_path, weights_only=False)
            self.assertEqual(final["step"], 4)
            self.assertEqual(final["scheduler"]["last_epoch"], 4)
            self.assertTrue(all(state["step"].item() == 4 for state in final["optimizer"]["state"].values()))
            self.assertEqual([p.name for p in sorted((run_dir / "ckpts").glob("*.pt"))],
                             ["step_0000003.pt", "step_0000004.pt"])
            self.assertFalse(list((run_dir / "ckpts").glob("*.tmp")))
            records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in records if "step" in r], [1, 2, 3, 4])
            for row in (record for record in records if "step" in record):
                self.assertIn("gradient_groups", row)
                self.assertGreater(row["steps_per_second"], 0)
                self.assertGreaterEqual(row["eta_seconds"], 0)
                self.assertLessEqual(row["grad_norm_postclip"], row["grad_norm"] + 1e-6)
            validation = [json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in validation], [0, 2, 4])
            self.assertTrue((run_dir / "diagnostics/dashboard.png").is_file())
            self.assertTrue((run_dir / "diagnostics/gradient_heatmap.png").is_file())
            self.assertTrue((run_dir / "diagnostics/runtime.png").is_file())
            self.assertTrue((run_dir / "best_validation.safetensors").is_file())
            self.assertEqual(json.loads((run_dir / "progress.json").read_text())["phase"], "complete")
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            segments = list((run_dir / "tensorboard").iterdir())
            self.assertEqual(len(segments), 2)
            accumulator = EventAccumulator(str(segments[-1])).Reload()
            self.assertIn("train/grad_norm", accumulator.Tags()["scalars"])
            with self.assertRaisesRegex(ValueError, "real-data"):
                export_model(final_path, Path(folder) / "export")
            incompatible_cfg = copy.deepcopy(self.cfg)
            incompatible_cfg["n_steps"] += 1
            args.resume = str(final_path)
            with self.assertRaisesRegex(ValueError, "n_steps"):
                run_training(TraceDiT(**self.cfg["model"]), self.count, incompatible_cfg, args)

    def test_trainer_uploads_pilot_and_final_resume_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "hub-run"
            cfg = copy.deepcopy(self.cfg)
            cfg.update(n_steps=2, sample_step=2, checkpoint_step=1,
                       monitoring={"enabled": False})
            args = SimpleNamespace(fake_data=True, output_dir=folder, run_suffix="hub",
                run_dir=str(run_dir), resume=None, stop_after=1,
                hub_checkpoint_repo="test-user/test-model", hub_checkpoint_every=2)
            receipt = {"repo_id": "test-user/test-model", "run_name": "hub-run",
                       "checkpoint": "fixture", "step": 0, "revision": "abc",
                       "path_in_repo": "fixture", "url": "https://huggingface.co/checkpoint"}
            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), \
                 patch("huggingface_hub.HfApi"), \
                 patch("hub.upload_training_checkpoint", return_value=receipt.copy()) as upload:
                run_training(self.model, self.count, cfg, args)
                args.resume = str(run_dir / "ckpts/step_0000001.pt")
                args.stop_after = None
                run_training(TraceDiT(**cfg["model"]), self.count, cfg, args)
            self.assertEqual([call.args[2].name for call in upload.call_args_list],
                             ["step_0000001.pt", "step_0000002.pt"])
            events = [json.loads(line) for line in
                      (run_dir / "hub_checkpoints.jsonl").read_text().splitlines()]
            self.assertEqual([event["status"] for event in events], ["uploaded", "uploaded"])

    def test_hub_failure_is_logged_without_losing_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "failed-upload"
            cfg = copy.deepcopy(self.cfg)
            cfg.update(n_steps=1, sample_step=1, checkpoint_step=1,
                       monitoring={"enabled": False})
            args = SimpleNamespace(fake_data=True, output_dir=folder, run_suffix="hub-failure",
                run_dir=str(run_dir), resume=None, stop_after=None,
                hub_checkpoint_repo="test-user/test-model", hub_checkpoint_every=1)
            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), \
                 patch("huggingface_hub.HfApi"), patch("train.time.sleep"), \
                 patch("hub.upload_training_checkpoint", side_effect=OSError("network down")) as upload:
                run_training(self.model, self.count, cfg, args)
            self.assertTrue((run_dir / "ckpts/step_0000001.pt").is_file())
            self.assertEqual(upload.call_count, 3)
            event = json.loads((run_dir / "hub_checkpoints.jsonl").read_text())
            self.assertEqual(event["status"], "failed")
            self.assertEqual(event["attempts"], 3)

    def test_export_safetensors_complete_run_only_and_sample_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ckpts").mkdir()
            checkpoint = {"config": self.cfg, "step": 4,
                          "model": self.model.state_dict(), "trainable_parameters": self.count,
                          "synthetic_data": False}
            path = root / "ckpts/step_0000004.pt"
            torch.save(checkpoint, path)
            exported = export_model(path, root / "export")
            from safetensors.torch import load_file
            reloaded = TraceDiT(**self.cfg["model"])
            reloaded.load_state_dict(load_file(str(exported / "model.safetensors")), strict=True)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[key])
            self.assertTrue((exported / "sample.py").is_file())
            self.assertTrue((exported / "third_party/MeanFlow.LICENSE").is_file())
            self.assertFalse((exported / "optimizer.pt").exists())
            with patch.object(sys, "argv", [str(exported / "sample.py"), "--model-dir",
                                            str(exported), "--output", str(root / "sample.png")]):
                runpy.run_path(str(exported / "sample.py"), run_name="__main__")
            from PIL import Image
            with Image.open(root / "sample.png") as image:
                self.assertEqual(image.size, (8, 16))
            checkpoint["step"] = 3
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "all configured"):
                export_model(path, root / "incomplete")

    def test_upload_filters_files_and_verifies_returned_commit(self):
        api = MagicMock()
        api.upload_folder.return_value.oid = "abc123"
        api.list_repo_files.return_value = ["model.safetensors", "config.json", "sample.py",
                                            "models/dit.py", "trace_imf.py"]
        url = upload_model(api, "test-user/test-model", "/tmp/export")
        self.assertEqual(url, "https://huggingface.co/test-user/test-model/tree/abc123")
        arguments = api.upload_folder.call_args.kwargs
        self.assertIn("model.safetensors", arguments["allow_patterns"])
        self.assertNotIn("*.pt", arguments["allow_patterns"])
        self.assertNotIn(".env", arguments["allow_patterns"])
        self.assertEqual(api.list_repo_files.call_args.kwargs["revision"], "abc123")
        api.list_repo_files.return_value = ["config.json"]
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            upload_model(api, "test-user/test-model", "/tmp/export")

    def test_training_checkpoint_upload_uses_versioned_path_and_verifies_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "step_0001000.pt"
            checkpoint.write_bytes(b"full resume state")
            api = MagicMock()
            api.upload_file.return_value.oid = "checkpoint-commit"
            remote_path = "training-checkpoints/run-v1/step_0001000.pt"
            api.list_repo_files.return_value = [remote_path]
            receipt = upload_training_checkpoint(
                api, "test-user/test-model", checkpoint, "run-v1")
            self.assertEqual(receipt["step"], 1000)
            self.assertEqual(receipt["path_in_repo"], remote_path)
            self.assertIn("checkpoint-commit", receipt["url"])
            self.assertEqual(api.upload_file.call_args.kwargs["path_in_repo"], remote_path)
            self.assertEqual(api.list_repo_files.call_args.kwargs["revision"], "checkpoint-commit")
            api.list_repo_files.return_value = []
            with self.assertRaisesRegex(RuntimeError, "missing uploaded checkpoint"):
                upload_training_checkpoint(api, "test-user/test-model", checkpoint, "run-v1")

    def test_repo_visibility_is_preserved(self):
        with patch("huggingface_hub.HfApi") as factory:
            api = factory.return_value
            api.model_info.return_value.private = False
            with self.assertRaisesRegex(ValueError, "visibility"):
                prepare_repository("test-user/test-model", "test-token", private=True)
            api.create_repo.assert_not_called()
            self.assertIs(prepare_repository("test-user/test-model", "test-token", private=False), api)
            self.assertIs(prepare_repository("test-user/test-model", "test-token", private=None), api)

    def test_auto_visibility_creates_new_repositories_private(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        with patch("huggingface_hub.HfApi") as factory:
            api = factory.return_value
            api.model_info.side_effect = RepositoryNotFoundError(
                "missing", response=MagicMock(status_code=404))
            self.assertIs(prepare_repository("test-user/new-model", "test-token", private=None), api)
            self.assertTrue(api.create_repo.call_args.kwargs["private"])


if __name__ == "__main__":
    unittest.main()
