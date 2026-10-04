"""Derive the 22M Multi-Trace notebook from the shared Modal clone/resume workflow."""
import json
from build_modal_notebook import ROOT, cell, make_notebook


def build():
    notebook = make_notebook()
    replacements = {}
    replacements["intro"] = cell("markdown", """
        # Multi-Trace U-Net 22M · Modal · resume / diagnostics / Hugging Face

        Tham chiếu `Untitled0.ipynb`: cùng U-Net, shared u head, K=1..5,
        interval loss và EMA; tăng backbone **5.946.579 → 22.002.655 parameters**.
        Base channels 44 → 84, condition dimension 160 → 356.

        Đây là mở rộng Multi-Trace 1–5 NFE; objective one-step nguyên bản trong
        proposal chỉ train hai trace r=0/r=t. Config và lý do scale ở SCALING.md.

        **Trước khi Run all:** chọn 1 GPU (H200 đang dùng được), CPU 2 cores,
        RAM 8 GiB; attach Volume `imf-training` tại `/mnt/imf-training`;
        attach Secret `HF_TOKEN` quyền write, thêm `GITHUB_TOKEN` nếu repo private.
        Không đặt token trực tiếp trong notebook.

        Effective batch 512 = micro-batch 128 × accumulation 4. Giữ 100.000
        optimizer updates, warmup 5.000, LR 1e-4 → 1e-5, clipping 1.0, EMA .9999.
        Đổi effective batch thì số updates/warmup/EMA tự tính theo image budget
        51.2M/2.56M ảnh. LR và clipping không nhân theo số parameters.

        Mặc định **chạy thử đến update 1000, lưu checkpoint rồi tạm dừng** để đo
        throughput/ETA trước khi dùng tiếp credit. Đặt `STOP_AFTER_UPDATES=None`
        và chạy lại pipeline để resume đến hết schedule, đánh giá FID 1–5 NFE
        rồi tự upload EMA model và diagnostics lên Hugging Face.
        Giới hạn thử không đổi LR schedule. 100k updates + FID có thể vượt $30;
        notebook không tự suy ra chi phí từ tên GPU.
        Dùng RUN_NAME mới; không resume checkpoint DiT từ notebook cũ.
        """, "intro")
    replacements["settings"] = cell("code", """
        from pathlib import Path

        REPO_URL = "https://github.com/namdeptraivcd/IMF.git"
        REPO_REF = "codex/modal-training"
        VOLUME_ROOT = Path("/mnt/imf-training")
        RUN_NAME = "multitrace_unet22m_hfcheckpoints_v1"
        BATCH_SIZE = 128  # physical image microbatch; joint edge/diag forward uses 256 inputs
        ACCUMULATION_STEPS = 4
        NUM_WORKERS = 2
        STOP_AFTER_UPDATES = 1_000  # None: train full image budget and auto upload
        LOG_EVERY = 50
        PLOT_EVERY = 500
        VALIDATION_EVERY = 1_000
        VALIDATION_BATCHES = 5  # fixed images and all intervals for each K=1..5
        SAMPLE_EVERY = 1_000
        CHECKPOINT_EVERY = 1_000
        KEEP_LAST_CHECKPOINTS = 3
        HF_CHECKPOINT_EVERY = 10_000  # pilot/final are also uploaded even off-cycle
        REFRESH_SECONDS = 10
        FID_NUM_GENERATED = 10_000
        FID_BATCH_SIZE = 128
        HF_REPO_ID = None  # or "username/multitrace-imf-cifar10-unet22m"
        HF_PRIVATE = None  # None: giữ visibility repo hiện có; repo mới mặc định private
        AUTO_RESUME = True
        RESUME_CHECKPOINT = None
        """, "settings")
    # Shared clone logic includes the Modal symlink fix and pins the saved commit.
    clone = next(c["source"] for c in notebook["cells"] if c["id"] == "clone")
    clone = clone.replace("missing = [name", 'required += ["models/unet.py", "multi_trace_imf.py", "objectives.py", "ema.py",\n'
        '             "evaluation.py", "configs/multitrace_cifar10_22m.py"]\nmissing = [name')
    clone += '\nfor module_name in ("hub", "monitoring", "models", "objectives", "trace_imf", "multi_trace_imf"):\n'
    clone += '    loaded = sys.modules.get(module_name)\n    if loaded and getattr(loaded, "__file__", None):\n'
    clone += '        if not Path(loaded.__file__).resolve().is_relative_to(SOURCE_DIR.resolve()):\n'
    clone += '            raise RuntimeError("Restart kernel trước khi đổi notebook/repo source để xóa module cache")\n'
    replacements["clone"] = cell("code", clone, "clone")
    preflight = next(c["source"] for c in notebook["cells"] if c["id"] == "preflight")
    preflight = preflight.replace("min(TRAIN_STEPS, BATCH_SIZE", "min(ACCUMULATION_STEPS, FID_NUM_GENERATED, FID_BATCH_SIZE, HF_CHECKPOINT_EVERY, BATCH_SIZE")
    preflight = preflight.replace("trace-imf-cifar10-22m", "multitrace-imf-cifar10-unet22m")
    preflight = preflight.replace(
        'print("Hub destination:", f"https://huggingface.co/{REPO_ID}", "private" if HF_PRIVATE else "public")',
        'repo_private = hub_api.model_info(REPO_ID).private\n'
        'print("Hub destination:", f"https://huggingface.co/{REPO_ID}", "private" if repo_private else "public")')
    preflight += '\nif STOP_AFTER_UPDATES is not None and STOP_AFTER_UPDATES <= 0:\n    raise ValueError("STOP_AFTER_UPDATES phải dương hoặc None")\n'
    preflight += '\nif HF_CHECKPOINT_EVERY % CHECKPOINT_EVERY:\n    raise ValueError("HF_CHECKPOINT_EVERY phải là bội số của CHECKPOINT_EVERY")\n'
    preflight += '\nif FID_NUM_GENERATED < 10 or FID_NUM_GENERATED % 10:\n    raise ValueError("FID_NUM_GENERATED phải chia hết cho 10 để cân bằng classes")\n'
    replacements["preflight"] = cell("code", preflight, "preflight")
    replacements["runtime-config"] = cell("code", """
        import copy
        import pprint
        from configs.multitrace_cifar10_22m import config as base_config, scale_training_budget

        cfg = copy.deepcopy(base_config)
        cfg.update(scale_training_budget(BATCH_SIZE, ACCUMULATION_STEPS))
        TRAIN_STEPS = cfg["n_steps"]
        cfg.update(num_workers=NUM_WORKERS, data_root=str(VOLUME_ROOT / "data/cifar10"),
                   log_step=LOG_EVERY, sample_step=SAMPLE_EVERY,
                   checkpoint_step=CHECKPOINT_EVERY, keep_last_checkpoints=KEEP_LAST_CHECKPOINTS,
                   mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
                   source_manifest=SOURCE_MANIFEST,
                   monitoring=dict(enabled=True, tensorboard=True, save_best=True, validate_raw=True,
                       plot_every=PLOT_EVERY, validation_every=VALIDATION_EVERY,
                       validation_batches=VALIDATION_BATCHES, sample_n_per_class=2),
                   evaluation=dict(num_generated=FID_NUM_GENERATED, batch_size=FID_BATCH_SIZE,
                                   nfes=[1, 2, 3, 4, 5], seed=2026))
        if not torch.cuda.is_bf16_supported():
            cfg["multi_trace_imf"]["jvp_precision"] = "fp32"
        # JSON normalization keeps resume comparisons stable for list/tuple settings.
        cfg = json.loads(json.dumps(cfg))
        config_dir = VOLUME_ROOT / "configs"
        config_dir.mkdir(exist_ok=True)
        CONFIG_PATH = config_dir / f"{RUN_NAME}.py"
        CONFIG_PATH.write_text("config = " + pprint.pformat(cfg, sort_dicts=False) + "\\n")
        child_env = os.environ.copy()
        child_env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
                         MPLCONFIGDIR="/tmp/imf-matplotlib", TORCH_HOME=str(VOLUME_ROOT / "cache/torch"))
        subprocess.run([sys.executable, "train.py", "--config", str(CONFIG_PATH), "--check-model"],
                       cwd=SOURCE_DIR, env=child_env, check=True)
        print("Microbatch / accumulation / effective batch:", BATCH_SIZE, ACCUMULATION_STEPS,
              BATCH_SIZE * ACCUMULATION_STEPS)
        print("Updates / warmup / EMA:", TRAIN_STEPS, cfg["warmup_steps"], cfg["ema_decay"])
        print("Run:", RUN_DIR)
        """, "runtime-config")
    smoke = next(c["source"] for c in notebook["cells"] if c["id"] == "gpu-smoke")
    smoke = smoke.replace('print("GPU smoke passed")',
        'print("GPU smoke passed at microbatch=2; training microbatch memory is measured in the pilot run")')
    replacements["gpu-smoke"] = cell("code", smoke, "gpu-smoke")
    replacements["pipeline-note"] = cell("markdown", """
        ## Train → resume → FID → upload

        Resume khôi phục raw model, optimizer, scheduler, EMA, RNG và global update.
        GPU-resident CIFAR loader khôi phục permutation/cursor; CPU/FakeData loader
        bắt đầu shuffle mới. Logs sau checkpoint bị loại. Progress/ETA tính theo
        optimizer update, gồm 4 microbatches; EMA và LR chỉ update một lần.

        Khi thử 1000 updates xong, xem runtime/ETA và clip_fraction. Đặt
        `STOP_AFTER_UPDATES=None`, chạy lại settings và pipeline để tiếp tục;
        không cần đổi config hoặc RUN_NAME. Nếu restart kernel, Run all.
        Thay đổi architecture/effective batch/image budget cần RUN_NAME mới.
        Không chạy hai kernel vào cùng run. Interrupt có thể mất các update sau
        checkpoint gần nhất. Nếu OOM, dùng RUN_NAME mới và batch64 × accumulation8.

        Checkpoint đầy đủ vẫn lưu vào Volume mỗi 1.000 updates. Bản resume gồm raw
        model, EMA, optimizer, scheduler, scaler, RNG và data cursor được upload vào
        `training-checkpoints/<RUN_NAME>/` trên cùng Hugging Face model repo mỗi
        10.000 updates; checkpoint pilot và final luôn được upload. Upload retry ba
        lần; nếu Hub/network lỗi thì training tiếp tục và ghi trạng thái vào
        `hub_checkpoints.jsonl`. Các file này lớn nên không nên đặt interval 1.000.

        FID dùng **pytorch-fid 0.3.0**, 50k CIFAR train images không augmentation,
        mặc định 10k generated/class-balanced cho mỗi NFE1..5, cùng noise/labels.
        Đây là FID10k, không phải FID50k và không so trực tiếp với FID trong notebook
        tham chiếu (ImageNet Inception weights khác). Eval có progress/ETA riêng,
        mất thêm GPU time; real stats được cache, kết quả từng NFE được lưu để retry.
        Nếu upload lỗi, chạy lại pipeline: bỏ qua train và phần FID đã xong.
        Inference bundle chỉ được upload khi đủ training updates và FID hoàn tất.
        """, "pipeline-note")
    pipeline = next(c["source"] for c in notebook["cells"] if c["id"] == "train-and-upload")
    pipeline = pipeline.replace('if selected_resume:\n',
        'if STOP_AFTER_UPDATES is not None:\n        command += ["--stop-after", str(STOP_AFTER_UPDATES)]\n    if selected_resume:\n')
    pipeline = pipeline.replace('if STOP_AFTER_UPDATES is not None:\n',
        'command += ["--hub-checkpoint-repo", REPO_ID, "--hub-checkpoint-every", str(HF_CHECKPOINT_EVERY)]\n    if STOP_AFTER_UPDATES is not None:\n', 1)
    pipeline = pipeline[:pipeline.index('EXPORT_DIR =')]
    pipeline += '''# Retry the newest checkpoint if an earlier in-training upload failed.
latest_checkpoints = sorted((RUN_DIR / "ckpts").glob("step_*.pt"))
if latest_checkpoints:
    latest_checkpoint = latest_checkpoints[-1]
    upload_log = RUN_DIR / "hub_checkpoints.jsonl"
    upload_events = ([json.loads(line) for line in upload_log.read_text().splitlines() if line.strip()]
                     if upload_log.exists() else [])
    already_uploaded = any(event.get("status") == "uploaded" and
                           event.get("checkpoint") == latest_checkpoint.name
                           for event in upload_events)
    if not already_uploaded:
        from hub import upload_training_checkpoint
        try:
            receipt = upload_training_checkpoint(hub_api, REPO_ID, latest_checkpoint, RUN_NAME)
        except Exception as error:
            print("WARNING: latest checkpoint remains only on Volume; rerun pipeline to retry:", error)
        else:
            receipt.update(status="uploaded", attempts=1)
            with upload_log.open("a") as stream:
                stream.write(json.dumps(receipt) + "\\n")
            print("Latest Hub checkpoint verified:", receipt["url"])

MODEL_URL = None
EXPORT_DIR = VOLUME_ROOT / "hub_exports" / RUN_NAME
if FINAL_CHECKPOINT.exists():
    subprocess.run([sys.executable, "evaluation.py", "--checkpoint", str(FINAL_CHECKPOINT),
                    "--cache-dir", str(VOLUME_ROOT / "cache/fid")],
                   cwd=SOURCE_DIR, env=child_env, check=True)
    export_model(FINAL_CHECKPOINT, EXPORT_DIR, source_dir=SOURCE_DIR)
    MODEL_URL = upload_model(hub_api, REPO_ID, EXPORT_DIR)
    (RUN_DIR / "hub_upload.json").write_text(json.dumps(
        {"repo_id": REPO_ID, "url": MODEL_URL, "training_steps": TRAIN_STEPS,
         "git_commit": GIT_COMMIT}, indent=2))
    print("Hub commit verified:", MODEL_URL)
else:
    print("Pilot paused with checkpoint; set STOP_AFTER_UPDATES=None and rerun settings + pipeline to resume")
'''
    replacements["train-and-upload"] = cell("code", pipeline, "train-and-upload")
    results = next(c["source"] for c in notebook["cells"] if c["id"] == "results")
    results = results.replace('display(Markdown(f"Model đã upload: [{REPO_ID}]({MODEL_URL})"))',
        'if MODEL_URL:\n    display(Markdown(f"EMA model đã upload: [{REPO_ID}]({MODEL_URL})"))')
    results = results.replace('"runtime.png")', '"runtime.png", "multitrace.png")')
    results = results.replace('print("Resume checkpoint:", FINAL_CHECKPOINT)',
        'print("Latest checkpoint:", sorted((RUN_DIR / "ckpts").glob("step_*.pt"))[-1])')
    results += '\nfid_path = RUN_DIR / "fid.json"\nif fid_path.exists():\n    print(fid_path.read_text())\n'
    replacements["results"] = cell("code", results, "results")
    replacements["download-note"] = cell("markdown", """
        ## Inference / TensorBoard

        Hugging Face nhận checkpoint resume định kỳ trong
        `training-checkpoints/<RUN_NAME>/`, rồi final EMA `model.safetensors`, config,
        inference source, best EMA validation weights, FID, samples và diagnostics.
        Download repo bằng snapshot_download
        rồi cài requirements.txt. Dùng `sample.py --model-dir ... --nfe 1` hoặc
        `--nfe 5`; `--weights best_validation.safetensors` chọn best snapshot.
        TensorBoard: `tensorboard --logdir /path/to/download/tensorboard`.
        `diagnostics/multitrace.png` so loss theo K; từng đường có số quan sát khác
        nhau khi train, validation dùng data/noise/interval cố định và EMA.
        """, "download-note")
    guide = next(c["source"] for c in notebook["cells"] if c["id"] == "diagnostic-guide")
    guide = guide.replace("256 test images", "fixed test images")
    guide += '\nMulti-Trace: validation có cả raw và EMA loss để phân biệt EMA lag với training plateau;\nsample/best weights dùng EMA. Mỗi K=1..5 có loss riêng và sample cùng noise.\nGradient heatmap tách từng encoder/decoder level. Train log có effective_batch_size,\nimages_seen, images_per_second, EMA updates và số update quan sát cho mỗi K.\n'
    replacements["diagnostic-guide"] = cell("markdown", guide, "diagnostic-guide")
    notebook["cells"] = [replacements.get(c["id"], c) for c in notebook["cells"]]
    destination = ROOT / "notebooks/multitrace_imf_modal.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Created {destination}")


if __name__ == "__main__":
    build()
