import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/multitrace_imf_modal.ipynb"


class MultiTraceNotebookTests(unittest.TestCase):
    def source(self, name):
        notebook = json.loads(NOTEBOOK.read_text())
        return next(c["source"] for c in notebook["cells"] if c["id"] == name)

    def test_notebook_schema_and_code_syntax(self):
        import nbformat
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertFalse(cell.outputs)

    def test_pilot_does_not_evaluate_export_or_upload_partial_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            state = dict(Path=Path, json=json, RUN_DIR=run, TRAIN_STEPS=100_000,
                cfg={}, RESUME_CHECKPOINT=None, AUTO_RESUME=True, STOP_AFTER_UPDATES=1000,
                CONFIG_PATH=root / "config.py", SOURCE_DIR=root, child_env={},
                REFRESH_SECONDS=10, VOLUME_ROOT=root, RUN_NAME="pilot", REPO_ID="test/model",
                hub_api=object(), GIT_COMMIT="fixture")
            import sys
            import subprocess
            state.update(sys=sys, subprocess=subprocess)
            with patch("monitoring.run_with_dashboard") as train, patch("hub.export_model") as export, \
                 patch("hub.upload_model") as upload, patch("subprocess.run") as evaluation:
                exec(compile(self.source("train-and-upload"), "pipeline", "exec"), state)
                command = train.call_args.args[0]
                self.assertEqual(command[command.index("--stop-after") + 1], "1000")
                evaluation.assert_not_called()
                export.assert_not_called()
                upload.assert_not_called()
                self.assertIsNone(state["MODEL_URL"])

    def test_completed_checkpoint_evaluates_then_exports_uploads_without_retraining(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            (run / "ckpts").mkdir(parents=True)
            final = run / "ckpts/step_0100000.pt"
            final.write_bytes(b"fixture")
            import sys
            import subprocess
            state = dict(Path=Path, json=json, sys=sys, subprocess=subprocess,
                RUN_DIR=run, TRAIN_STEPS=100_000, cfg={}, RESUME_CHECKPOINT=None,
                AUTO_RESUME=True, STOP_AFTER_UPDATES=None, CONFIG_PATH=root / "config.py",
                SOURCE_DIR=root, child_env={}, REFRESH_SECONDS=10, VOLUME_ROOT=root,
                RUN_NAME="complete", REPO_ID="test/model", hub_api=object(), GIT_COMMIT="fixture")
            events = []
            with patch("monitoring.run_with_dashboard") as train, \
                 patch("subprocess.run", side_effect=lambda *a, **kw: events.append("evaluate")), \
                 patch("hub.export_model", side_effect=lambda *a, **kw: events.append("export")), \
                 patch("hub.upload_model", side_effect=lambda *a, **kw: events.append("upload") or "https://huggingface.co/test/model"):
                exec(compile(self.source("train-and-upload"), "pipeline", "exec"), state)
                train.assert_not_called()
                self.assertEqual(events, ["evaluate", "export", "upload"])
                self.assertTrue((run / "hub_upload.json").is_file())

    def test_modal_volume_symlink_is_accepted_in_the_new_notebook(self):
        code = self.source("clone")
        guard = code[:code.index("if not re.fullmatch")]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            mount_root, backing = root / "mnt", root / "__modal/volumes/vo-test"
            mount_root.mkdir()
            backing.mkdir(parents=True)
            mounted = mount_root / "imf-training"
            mounted.symlink_to(backing, target_is_directory=True)
            guard = guard.replace('Path("/mnt")', f"Path({str(mount_root)!r})")
            exec(compile(guard, "guard", "exec"), dict(Path=Path, VOLUME_ROOT=mounted))


if __name__ == "__main__":
    unittest.main()
