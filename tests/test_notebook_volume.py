"""Regression proof for Modal's /mnt symlinks to /__modal/volumes storage."""
import json
from pathlib import Path
import tempfile
import unittest


class NotebookVolumeTests(unittest.TestCase):
    def guard(self, volume_root, mount_root):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               "notebooks/trace_imf_modal.ipynb").read_text())
        code = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "clone")
        prefix = code[:code.index("if not re.fullmatch")]
        prefix = prefix.replace('Path("/mnt")', f"Path({str(mount_root)!r})")
        exec(compile(prefix, "notebook-volume-guard", "exec"),
             {"Path": Path, "VOLUME_ROOT": volume_root})

    def test_modal_symlink_mount_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            mount_root = root / "mnt"
            mount_root.mkdir()
            backing = root / "__modal" / "volumes" / "vo-test"
            backing.mkdir(parents=True)
            mounted = mount_root / "imf-training"
            mounted.symlink_to(backing, target_is_directory=True)
            self.assertTrue(mounted.is_dir())
            self.assertFalse(mounted.resolve().is_relative_to(mount_root))
            self.guard(mounted, mount_root)

    def test_normal_mount_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            mount_root = Path(folder).resolve() / "mnt"
            mounted = mount_root / "imf-training"
            mounted.mkdir(parents=True)
            self.guard(mounted, mount_root)

    def test_missing_mount_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            mount_root = Path(folder).resolve() / "mnt"
            with self.assertRaisesRegex(RuntimeError, "Attach Volume"):
                self.guard(mount_root / "missing", mount_root)

    def test_existing_path_outside_mount_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            mount_root = root / "mnt"
            mount_root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Attach Volume"):
                self.guard(root, mount_root)


if __name__ == "__main__":
    unittest.main()
