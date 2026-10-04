"""Generate the native Modal notebook: clone IMF, monitor, resume, upload."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, name):
    result = {"cell_type": kind, "id": name, "metadata": {},
              "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        ast.parse(result["source"])
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [cell("markdown", """
        # Trace-iMF 22M · Modal Notebook · diagnostics / resume / Hugging Face

        Chạy trực tiếp trên https://modal.com/notebooks. Notebook **clone repo IMF**,
        train CIFAR-10 với 22.082.956 tham số, theo dõi hội tụ trực tiếp và tự upload
        model cùng diagnostics lên Hugging Face sau khi train đủ bước.

        **Chuẩn bị trước khi Run all:**

        1. Compute profile: **1 GPU L4 hoặc A100**, 4 CPU, RAM 16 GiB.
        2. Files: attach Volume `imf-training` tại `/mnt/imf-training`.
        3. Attach Modal Secret có `HF_TOKEN` quyền write. Nếu GitHub repo private,
           attach thêm Secret `GITHUB_TOKEN` quyền đọc repo.
        4. Chỉnh cell settings, rồi Run all. Không đặt token literal trong notebook.

        Mặc định 200.000 bước có tính phí GPU. Đặt `TRAIN_STEPS=20` và các interval
        bằng 5 để thử toàn bộ luồng trước lượt train dài. Chọn RUN_NAME mới khi đổi
        config. Resume tự pin Git commit đã dùng cho run, để tránh đổi code giữa chừng.
        Notebook không cần `modal deploy` hoặc Modal API token.

        Nguồn: [Modal setup](https://modal.com/docs/guide/notebooks),
        [Hub upload](https://huggingface.co/docs/huggingface_hub/guides/upload).
        """, "intro"), cell("code", """
        from pathlib import Path

        REPO_URL = "https://github.com/namdeptraivcd/IMF.git"
        REPO_REF = "codex/modal-training"  # có thể dùng full Git commit SHA
        VOLUME_ROOT = Path("/mnt/imf-training")
        RUN_NAME = "cifar10_22m_v1"
        TRAIN_STEPS = 200_000
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        LOG_EVERY = 50
        PLOT_EVERY = 200
        VALIDATION_EVERY = 1_000
        VALIDATION_BATCHES = 4  # cố định 256 test images nếu batch=64
        SAMPLE_EVERY = 1_000
        CHECKPOINT_EVERY = 1_000
        KEEP_LAST_CHECKPOINTS = 3
        REFRESH_SECONDS = 10
        HF_REPO_ID = None  # hoặc "username/trace-imf-cifar10-22m"
        HF_PRIVATE = True
        AUTO_RESUME = True
        RESUME_CHECKPOINT = None  # None: checkpoint mới nhất; hoặc đường dẫn .pt
        """, "settings"), cell("markdown", """
        ## Clone code và pin commit

        Clone từ GitHub, checkout REPO_REF. Khi resume, dùng Git commit từ
        source_manifest.json của run trước. Kernel restart cần chạy lại notebook
        từ đầu với cùng RUN_NAME/config. Không chạy hai kernel vào cùng run.
        """, "clone-note"), cell("code", """
        import hashlib
        import json
        import os
        import re
        import shutil
        import subprocess
        import sys
        import tempfile

        # Modal exposes /mnt mounts as symlinks into /__modal/volumes.
        # Validate the configured mount path without resolving that symlink.
        if not VOLUME_ROOT.is_dir() or not Path(os.path.abspath(VOLUME_ROOT)).is_relative_to(Path("/mnt")):
            raise RuntimeError(f"Attach Volume tại {VOLUME_ROOT} trước khi chạy")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME):
            raise ValueError("RUN_NAME chỉ dùng chữ, số, underscore và gạch ngang")
        RUN_DIR = VOLUME_ROOT / "runs" / RUN_NAME
        manifest_path = RUN_DIR / "source_manifest.json"
        saved_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        chosen_ref = saved_manifest["git_commit"] if saved_manifest and AUTO_RESUME else REPO_REF
        SOURCE_DIR = Path("/tmp") / f"imf-repo-{RUN_NAME}"
        git_env = os.environ.copy()
        git_env["GIT_TERMINAL_PROMPT"] = "0"
        if git_env.get("GITHUB_TOKEN"):
            askpass = Path("/tmp/imf-git-askpass.py")
            askpass.write_text("#!/usr/bin/env python3\\nimport os,sys\\n"
                "print('x-access-token' if 'Username' in sys.argv[1] else os.environ['GITHUB_TOKEN'])\\n")
            askpass.chmod(0o700)
            git_env["GIT_ASKPASS"] = str(askpass)
        if not (SOURCE_DIR / ".git").exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            REPO_URL, str(SOURCE_DIR)], env=git_env, check=True)
        remote = subprocess.check_output(["git", "-C", str(SOURCE_DIR), "remote", "get-url", "origin"], text=True).strip()
        if remote != REPO_URL:
            raise ValueError("SOURCE_DIR chứa repo khác; restart kernel hoặc đổi RUN_NAME")
        subprocess.run(["git", "-C", str(SOURCE_DIR), "fetch", "origin", chosen_ref], env=git_env, check=True)
        subprocess.run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"], env=git_env, check=True)
        GIT_COMMIT = subprocess.check_output(["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
        required = ["train.py", "monitoring.py", "hub.py", "sample.py", "models/dit.py", "models/__init__.py",
                    "trace_imf.py", "data.py", "configs/__init__.py",
                    "configs/cifar10_22m.py", "requirements-notebook.txt"]
        missing = [name for name in required if not (SOURCE_DIR / name).is_file()]
        if missing:
            raise RuntimeError(f"Git ref thiếu code mới: {missing}. Push branch/ref triển khai rồi chạy lại.")
        SOURCE_MANIFEST = {"repo_url": REPO_URL, "git_commit": GIT_COMMIT,
            "files": {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() for name in required}}
        if saved_manifest and SOURCE_MANIFEST != saved_manifest:
            raise ValueError("Source manifest thay đổi; chọn RUN_NAME mới")
        sys.path.insert(0, str(SOURCE_DIR))
        print("Cloned IMF commit:", GIT_COMMIT)
        """, "clone"), cell("code", """
        if shutil.which("uv"):
            install_command = ["uv", "pip", "install", "--python", sys.executable,
                               "-r", str(SOURCE_DIR / "requirements-notebook.txt")]
        else:
            install_command = [sys.executable, "-m", "pip", "install",
                               "-r", str(SOURCE_DIR / "requirements-notebook.txt")]
        subprocess.run(install_command, check=True)
        """, "dependencies"), cell("code", """
        import torch
        import torchvision
        from huggingface_hub import HfApi
        from hub import prepare_repository

        if not torch.cuda.is_available():
            raise RuntimeError("Chọn GPU trong Compute profile của Modal rồi restart kernel")
        if min(TRAIN_STEPS, BATCH_SIZE, LOG_EVERY, PLOT_EVERY, VALIDATION_EVERY,
               VALIDATION_BATCHES, SAMPLE_EVERY, CHECKPOINT_EVERY, KEEP_LAST_CHECKPOINTS,
               REFRESH_SECONDS) <= 0 or NUM_WORKERS < 0:
            raise ValueError("Số bước/batch/interval phải dương; workers không âm")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Attach Secret có HF_TOKEN quyền write rồi restart kernel")
        account = HfApi(token=token).whoami()["name"]
        REPO_ID = HF_REPO_ID or f"{account}/trace-imf-cifar10-22m"
        hub_api = prepare_repository(REPO_ID, token, private=HF_PRIVATE)
        del token
        print("GPU:", torch.cuda.get_device_name(0))
        print("torch / torchvision:", torch.__version__, torchvision.__version__)
        print("Hub destination:", f"https://huggingface.co/{REPO_ID}", "private" if HF_PRIVATE else "public")
        """, "preflight"), cell("code", """
        import copy
        import pprint
        from configs.cifar10_22m import config as base_config

        cfg = copy.deepcopy(base_config)
        cfg.update(n_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                   data_root=str(VOLUME_ROOT / "data/cifar10"),
                   log_step=LOG_EVERY, sample_step=SAMPLE_EVERY,
                   checkpoint_step=CHECKPOINT_EVERY, keep_last_checkpoints=KEEP_LAST_CHECKPOINTS,
                   mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
                   source_manifest=SOURCE_MANIFEST,
                   monitoring=dict(enabled=True, tensorboard=True, save_best=True,
                       plot_every=PLOT_EVERY, validation_every=VALIDATION_EVERY,
                       validation_batches=VALIDATION_BATCHES, sample_n_per_class=4))
        config_dir = VOLUME_ROOT / "configs"
        config_dir.mkdir(exist_ok=True)
        CONFIG_PATH = config_dir / f"{RUN_NAME}.py"
        CONFIG_PATH.write_text("config = " + pprint.pformat(cfg, sort_dicts=False) + "\\n")
        child_env = os.environ.copy()
        child_env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
                         MPLCONFIGDIR="/tmp/imf-matplotlib")
        subprocess.run([sys.executable, "train.py", "--config", str(CONFIG_PATH), "--check-model"],
                       cwd=SOURCE_DIR, env=child_env, check=True)
        print("Train steps / batch / precision:", TRAIN_STEPS, BATCH_SIZE, cfg["mixed_precision"])
        print("Run on Volume:", RUN_DIR)
        """, "runtime-config"), cell("code", """
        # Đúng model 22M trên GPU, fake data để kiểm tra memory/JVP/backward.
        # Tắt diagnostics trong smoke để không lẫn ảnh/log thử vào run thật.
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["monitoring"] = {"enabled": False}
        with tempfile.TemporaryDirectory(prefix="trace-imf-gpu-smoke-") as smoke_dir:
            smoke_config = Path(smoke_dir) / "config.py"
            smoke_config.write_text("config = " + pprint.pformat(smoke_cfg) + "\\n")
            subprocess.run([sys.executable, "train.py", "--config", str(smoke_config),
                            "--fake-data", "--steps", "3", "--batch-size", "2",
                            "--num-workers", "0", "--output-dir", smoke_dir],
                           cwd=SOURCE_DIR, env=child_env, check=True)
        print("GPU smoke passed")
        """, "gpu-smoke"), cell("markdown", """
        ## Đọc diagnostics trong lúc train

        - **Progress bar**: global step, step/s, ETA giờ; `tqdm` trong trainer cũng
          ghi tốc độ và thời gian còn lại. ETA dùng tốc độ thực tế, bao gồm overhead
          đánh giá/vẽ/lưu, và có thể dao động mạnh ở đầu run.
        - **Dashboard live**: train edge/diagonal/total loss, held-out loss,
          global gradient trước/sau clip, clipping fraction, LR, JVP RMS,
          actual update/weight từng nhóm và spread của fixed-noise samples.
        - **Gradient heatmap**: RMS trước clip của embedding, từng block và head;
          log có missing/zero gradient tensors. Các gate zero-init có thể làm
          gradient trunk bằng 0 ở bước đầu; đó không tự động là lỗi.
        - **Validation**: CIFAR-10 test split, cố định ảnh/t/noise qua các lần eval.
          Test split được dùng như validation để tune; cần evaluation độc lập khi
          báo cáo metric cuối. Loss MSE có noise floor, không cần tiến về 0.
        - **Samples cố định**: 4 ảnh/class, cùng noise và nhãn ở mỗi lần sample.
          Spread nhỏ là gợi ý để kiểm tra ảnh; không tự chứng minh mode collapse.
        - **Convergence report**: ít nhất 5 eval points, gợi ý plateau nếu cải thiện
          <1% trong cửa sổ đó; ghi early/middle/late theo phần budget đã chạy.
          Đây là heuristic để đọc cùng gradient/LR/ảnh; không tự early-stop và
          không chứng minh hội tụ. Warmup dài, LR nhỏ hoặc regression noise floor
          đều có thể làm đường loss phẳng.
        - **Files bền vững**: JSONL, CSV, PNG, TensorBoard events và best validation
          weights trên Volume. Khi NaN/Inf, trainer ghi failure.json kèm tên
          parameter có gradient lỗi, dừng run và không upload.

        Nếu train loss giảm nhưng validation xấu đi, kiểm tra generalization;
        nếu cả hai phẳng, xem LR, warmup, grad/update ratio và clipping. Nếu val
        vẫn cải thiện ở cuối budget, có thể cần experiment dài hơn với RUN_NAME mới.
        Best validation weights được lưu riêng với step thực tế, bên cạnh final model.
        """, "diagnostic-guide"), cell("markdown", """
        ## Train với dashboard → tự upload khi hoàn tất

        Resume khôi phục model, optimizer, scheduler, RNG và global step. DataLoader
        bắt đầu shuffle mới; không đảm bảo replay bit-for-bit. Logs sau checkpoint
        được loại trước khi nối lịch sử. TensorBoard dùng segment mới cho mỗi resume.

        Chạy lại cell pipeline nếu upload lỗi: final checkpoint có sẵn sẽ bỏ qua
        train. Nếu restart kernel, chạy từ đầu với cùng RUN_NAME/config. Source commit
        tự pin từ run trước. Đổi config cần RUN_NAME mới. Nếu run directory tồn tại
        nhưng chưa có checkpoint, dùng tên mới. Interrupt cell sẽ dừng subprocess;
        resume từ checkpoint gần nhất, các bước chưa checkpoint có thể mất.
        """, "pipeline-note"), cell("code", """
        from hub import export_model, upload_model
        from monitoring import run_with_dashboard

        FINAL_CHECKPOINT = RUN_DIR / "ckpts" / f"step_{TRAIN_STEPS:07d}.pt"
        saved_config = RUN_DIR / "config.json"
        if saved_config.exists() and json.loads(saved_config.read_text()) != cfg:
            raise ValueError("Run có config khác; giữ cấu hình cũ để resume hoặc chọn RUN_NAME mới")
        checkpoints = sorted((RUN_DIR / "ckpts").glob("step_*.pt"))
        selected_resume = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else (checkpoints[-1] if checkpoints else None)
        if not FINAL_CHECKPOINT.exists():
            command = [sys.executable, "train.py", "--config", str(CONFIG_PATH), "--run-dir", str(RUN_DIR)]
            if selected_resume:
                if not AUTO_RESUME:
                    raise RuntimeError("Có checkpoint; bật AUTO_RESUME hoặc đổi RUN_NAME")
                if not selected_resume.resolve().is_relative_to((RUN_DIR / "ckpts").resolve()):
                    raise ValueError("Resume checkpoint phải thuộc RUN_DIR/ckpts")
                command += ["--resume", str(selected_resume)]
                print("Resume:", selected_resume.name)
            elif RUN_DIR.exists():
                raise RuntimeError("Run tồn tại nhưng chưa có checkpoint; chọn RUN_NAME mới")
            run_with_dashboard(command, SOURCE_DIR, child_env, RUN_DIR, TRAIN_STEPS,
                               refresh_seconds=REFRESH_SECONDS)
        else:
            print("Final checkpoint đã có; bỏ qua train và thử upload")
        EXPORT_DIR = VOLUME_ROOT / "hub_exports" / RUN_NAME
        export_model(FINAL_CHECKPOINT, EXPORT_DIR, source_dir=SOURCE_DIR)
        MODEL_URL = upload_model(hub_api, REPO_ID, EXPORT_DIR)
        (RUN_DIR / "hub_upload.json").write_text(json.dumps(
            {"repo_id": REPO_ID, "url": MODEL_URL, "training_steps": TRAIN_STEPS,
             "git_commit": GIT_COMMIT}, indent=2))
        print("Hub commit verified:", MODEL_URL)
        """, "train-and-upload"), cell("code", """
        from IPython.display import display, Image, Markdown

        display(Markdown(f"Model đã upload: [{REPO_ID}]({MODEL_URL})"))
        for name in ("dashboard.png", "gradient_heatmap.png", "runtime.png"):
            path = RUN_DIR / "diagnostics" / name
            if path.exists():
                display(Image(filename=str(path)))
        report = RUN_DIR / "diagnostics/convergence.json"
        if report.exists():
            print(json.dumps(json.loads(report.read_text()), indent=2))
        preview = EXPORT_DIR / "sample_grid.png"
        if preview.exists():
            display(Image(filename=str(preview)))
        print("Resume checkpoint:", FINAL_CHECKPOINT)
        print("TensorBoard logs:", RUN_DIR / "tensorboard")
        print("Best weights + metrics:", RUN_DIR / "best_validation.safetensors", RUN_DIR / "best_validation.json")
        print("Stop kernel khi kết thúc để ngừng sử dụng GPU.")
        """, "results"), cell("markdown", """
        ## Dùng model / TensorBoard từ máy khác

        Model là custom PyTorch, có code sampling và model card. Repo mới mặc định
        private; token có quyền truy cập cần được đặt trong environment khi download.

        ```python
        from huggingface_hub import snapshot_download
        folder = snapshot_download("USERNAME/trace-imf-cifar10-22m")
        ```

        ```bash
        python -m pip install -r /path/to/download/requirements.txt
        python /path/to/download/sample.py --model-dir /path/to/download --output samples.png
        # Nếu có best validation weights:
        python /path/to/download/sample.py --model-dir /path/to/download --weights best_validation.safetensors
        python -m pip install tensorboard
        tensorboard --logdir /path/to/download/tensorboard
        ```

        Optimizer checkpoints ở lại Volume; Hub nhận final safetensors, best weights
        khi có, config, sampling source, model card và diagnostics/TensorBoard logs.
        Chưa có FID hoặc kết luận chất lượng dựa riêng vào loss/gradient.
        """, "download-note")]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "notebooks/trace_imf_modal.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Created {destination}: {len(cells)} cells, Git clone workflow")


if __name__ == "__main__":
    build()
