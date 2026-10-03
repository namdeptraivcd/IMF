import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.dit import TraceDiT
from monitoring import (convergence_summary, evaluate, finish_gradient_snapshot,
                        gradient_snapshot, rollback_logs)
from trace_imf import TraceIMF


class MonitoringTests(unittest.TestCase):
    def test_scaled_gradients_and_measured_update_ratio(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        model.weight.grad = torch.tensor([[6.0, 8.0]]) * 128
        snapshot = gradient_snapshot(model, scale=128)
        model.weight.grad.div_(128)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        values = finish_gradient_snapshot(model, snapshot)["weight"]
        self.assertAlmostEqual(values["grad_norm"], 10.0)
        self.assertAlmostEqual(values["weight_norm"], 5.0)
        self.assertAlmostEqual(values["update_to_weight"], 0.2, places=6)

    def test_zero_initialized_weight_ratios_are_undefined(self):
        model = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        snapshot = gradient_snapshot(model)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        values = finish_gradient_snapshot(model, snapshot)["weight"]
        self.assertIsNone(values["grad_to_weight"])
        self.assertIsNone(values["update_to_weight"])

    def test_fixed_validation_and_training_rng_are_preserved(self):
        torch.set_num_threads(2)
        model = TraceDiT(input_size=4, in_channels=1, dim=32, depth=1, num_heads=4, num_classes=2)
        objective = TraceIMF(channels=1, image_size=4, num_classes=2)
        dataset = TensorDataset(torch.rand(4, 1, 4, 4), torch.tensor([0, 1, 0, 1]))
        loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(123))
        before = torch.get_rng_state().clone()
        first = evaluate(model, objective, loader, torch.device("cpu"), seed=999, max_batches=2)
        second = evaluate(model, objective, loader, torch.device("cpu"), seed=999, max_batches=2)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_plateau_is_only_a_phase_labeled_candidate(self):
        rows = [{"step": i*100, "loss": 1.0} for i in range(5)]
        self.assertEqual(convergence_summary([], rows[:2], 2000)["status"], "insufficient_validation_history")
        self.assertEqual(convergence_summary([], rows, 2000)["status"], "plateau_candidate_early")
        self.assertEqual(convergence_summary([], rows, 500)["status"], "plateau_candidate_late")
        rows[-1]["loss"] = 0.8
        self.assertEqual(convergence_summary([], rows, 2000)["status"], "improving")

    def test_resume_removes_future_metrics_and_stale_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "validation.jsonl").write_text(' {"step": 2, "loss": 1}\n{"step": 4, "loss": 0.5}\n')
            (root / "best_validation.json").write_text('{"step": 4}')
            (root / "best_validation.safetensors").write_bytes(b"fixture")
            rollback_logs(root, 2)
            self.assertEqual(json.loads((root / "validation.jsonl").read_text())["step"], 2)
            self.assertFalse((root / "best_validation.safetensors").exists())

    def test_notebook_clone_and_resume_pin_the_original_git_commit(self):
        import nbformat
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/trace_imf_modal.ipynb", as_version=4)
        nbformat.validate(notebook)
        code = next(cell.source for cell in notebook.cells if cell.id == "clone")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repository = root / "git-source"
            repository.mkdir()
            required = ["train.py", "monitoring.py", "hub.py", "sample.py", "models/dit.py", "models/__init__.py",
                        "trace_imf.py", "data.py", "configs/__init__.py", "configs/cifar10_22m.py", "requirements-notebook.txt"]
            for name in required:
                path = repository / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("# original fixture\n")
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL, text=True).strip()
            git("init", "--initial-branch=main")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
            original_commit = git("rev-parse", "HEAD")
            class MockMountedVolume:
                def is_dir(self):
                    return True
                def resolve(self):
                    return Path("/mnt/test-volume")
                def __truediv__(self, child):
                    return root / child
            state = {"Path": Path, "VOLUME_ROOT": MockMountedVolume(), "RUN_NAME": root.name,
                     "REPO_URL": str(repository), "REPO_REF": "main", "AUTO_RESUME": True}
            with patch.object(sys, "path", sys.path.copy()), patch.dict("os.environ", {"GITHUB_TOKEN": ""}):
                try:
                    exec(compile(code, "clone-cell", "exec"), state)
                    self.assertEqual(state["GIT_COMMIT"], original_commit)
                    state["RUN_DIR"].mkdir(parents=True)
                    (state["RUN_DIR"] / "source_manifest.json").write_text(json.dumps(state["SOURCE_MANIFEST"]))
                    (repository / "monitoring.py").write_text("# later fixture\n")
                    git("add", ".")
                    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "later")
                    self.assertNotEqual(git("rev-parse", "HEAD"), original_commit)
                    exec(compile(code, "clone-cell", "exec"), state)
                    self.assertEqual(state["GIT_COMMIT"], original_commit)
                    self.assertEqual((state["SOURCE_DIR"] / "monitoring.py").read_text(), "# original fixture\n")
                finally:
                    if "SOURCE_DIR" in state:
                        shutil.rmtree(state["SOURCE_DIR"])


if __name__ == "__main__":
    unittest.main()
