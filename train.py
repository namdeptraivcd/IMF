"""Trace-iMF trainer, adapted from backbone_IMF's Accelerate training flow."""
import argparse
import importlib.util
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models import build_backbone
from objectives import build_objective


def load_config(path):
    spec = importlib.util.spec_from_file_location("train_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def build_model(cfg):
    model = build_backbone(cfg)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = cfg["parameter_target"]
    if abs(count - target) > target * cfg["parameter_tolerance"]:
        raise ValueError(f"Model has {count:,} trainable parameters; target is {target:,}")
    return model, count


def smoke_test(model, cfg):
    # CPU proof with the actual configured model; no dataset download required.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    objective = build_objective(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    x = torch.rand(2, cfg["model"]["in_channels"], cfg["model"]["input_size"],
                   cfg["model"]["input_size"])
    labels = None if objective.num_classes is None else torch.tensor([0, 1])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective.loss(model, x, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke-test loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Missing or non-finite parameter gradient")
        optimizer.step()
        print(json.dumps({"smoke_step": step + 1, "loss": loss.item(),
                          **{k: v.item() for k, v in metrics.items()}}))
    samples = objective.sample(model, n_samples=2, labels=labels, device="cpu")
    if samples.shape != x.shape or not torch.isfinite(samples).all():
        raise RuntimeError("Invalid one-step samples")
    print(f"Smoke test passed: two optimizer steps; sample shape={tuple(samples.shape)}")


def run_training(model, count, cfg, args):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list, set_seed
    from torchvision.utils import make_grid, save_image
    from tqdm import tqdm
    from data import build_dataset, cycle, ResidentCIFARBatcher
    from ema import ModelEMA
    from monitoring import (TrainingMonitor, evaluate, finish_gradient_snapshot,
                            gradient_snapshot, rollback_logs)

    precision = cfg["mixed_precision"] if torch.cuda.is_available() else "no"
    accumulation = cfg.get("gradient_accumulation_steps", 1)
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if cfg.get("allow_tf32") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if cfg.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    accelerator = Accelerator(cpu=not torch.cuda.is_available(), mixed_precision=precision, kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=False)])
    hub_checkpoint_repo = getattr(args, "hub_checkpoint_repo", None)
    hub_checkpoint_every = getattr(args, "hub_checkpoint_every", None)
    hub_checkpoint_api = None
    if hub_checkpoint_repo:
        hub_checkpoint_every = hub_checkpoint_every or cfg["checkpoint_step"]
        if hub_checkpoint_every <= 0 or hub_checkpoint_every % cfg["checkpoint_step"]:
            raise ValueError("Hub checkpoint interval must be a positive multiple of checkpoint_step")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when Hub checkpoint backup is enabled")
        if accelerator.is_main_process:
            from huggingface_hub import HfApi
            hub_checkpoint_api = HfApi(token=token)
        del token
    resume_path = getattr(args, "resume", None)
    checkpoint = None
    start_step = 0
    if resume_path:
        # Resume only checkpoints produced by this trainer, including RNG state.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        original = checkpoint["config"]
        for key in ("architecture", "objective", "model", "trace_imf", "multi_trace_imf",
                    "n_steps", "lr", "warmup_steps", "min_lr_ratio", "weight_decay", "dataset",
                    "image_size", "seed", "batch_size", "gradient_accumulation_steps", "ema_decay",
                    "betas", "grad_clip", "preload_gpu", "mixed_precision", "channels_last", "allow_tf32"):
            # JSON normalizes tuples to lists; compare their serialized values.
            if json.dumps(cfg.get(key)) != json.dumps(original.get(key)):
                raise ValueError(f"Resume config mismatch: {key}")
        if cfg.get("source_manifest") != original.get("source_manifest"):
            raise ValueError("Resume source snapshot differs from the checkpoint")
        if checkpoint.get("synthetic_data", False) != args.fake_data:
            raise ValueError("Cannot resume a synthetic run as a real-data run or vice versa")
        if checkpoint.get("world_size", 1) != accelerator.num_processes:
            raise ValueError("Resume requires the same distributed world size")
        model.load_state_dict(checkpoint["model"], strict=True)
        start_step = checkpoint["step"]
    set_seed(cfg["seed"], device_specific=True)
    if args.fake_data:
        from torchvision.datasets import FakeData
        from torchvision.transforms import ToTensor
        dataset = FakeData(size=max(cfg["batch_size"] * accelerator.num_processes * 4, 64),
                           image_size=(cfg["model"]["in_channels"], cfg["image_size"],
                                       cfg["image_size"]),
                           num_classes=cfg["model"]["num_classes"] or 1, transform=ToTensor())
    else:
        with accelerator.main_process_first():
            dataset = build_dataset(cfg)
    if len(dataset) < cfg["batch_size"]:
        raise ValueError("Dataset is smaller than one batch with drop_last=True")
    resident = cfg.get("preload_gpu", False) and accelerator.device.type == "cuda" and not args.fake_data
    if resident and (cfg["dataset"] != "cifar10" or accelerator.num_processes != 1):
        raise ValueError("GPU-resident CIFAR profile requires one GPU/process")
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
        pin_memory=accelerator.device.type == "cuda")
    optimizer_options = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                             betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if cfg.get("fused_optimizer") and accelerator.device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)

    def lr_lambda(step):
        warmup = min(cfg["warmup_steps"], max(cfg["n_steps"] - 1, 0))
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = min(max((step - warmup) / max(cfg["n_steps"] - warmup, 1), 0), 1)
        return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    # Scheduler uses optimizer-step units, independent of distributed world size.
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg.get("ema_decay") else None
    if ema:
        ema.model.to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if checkpoint is not None and checkpoint.get("scaler") and accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(checkpoint["scaler"])
    objective = build_objective(cfg)
    name = [args.run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S_%f")]
    broadcast_object_list(name)
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
    elif resume_path:
        run_dir = Path(resume_path).resolve().parent.parent
    else:
        run_dir = Path(args.output_dir) / f"{cfg['name']}_{name[0]}"
    if accelerator.is_main_process:
        if checkpoint is None:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "images").mkdir()
            (run_dir / "ckpts").mkdir()
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            if cfg.get("source_manifest"):
                (run_dir / "source_manifest.json").write_text(json.dumps(cfg["source_manifest"], indent=2))
            (run_dir / "train.jsonl").write_text(json.dumps({"trainable_parameters": count,
                "world_size": accelerator.num_processes, "mixed_precision": precision,
                "synthetic_data": args.fake_data, "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda, "device": str(accelerator.device),
                "global_batch_size": cfg["batch_size"] * accumulation * accelerator.num_processes,
                "micro_batch_size": cfg["batch_size"], "gradient_accumulation_steps": accumulation,
                "ema_decay": cfg.get("ema_decay"),
                "gpu_name": torch.cuda.get_device_name() if accelerator.device.type == "cuda" else None}) + "\n")
        else:
            if not (run_dir / "config.json").exists():
                raise ValueError("Resume run directory is missing config.json")
            rollback_logs(run_dir, start_step)
            with (run_dir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps({"resume_from_step": start_step}) + "\n")
    accelerator.wait_for_everyone()
    loader = (ResidentCIFARBatcher(dataset, cfg["batch_size"], accelerator.device,
                                  checkpoint.get("batcher") if checkpoint else None)
              if resident else cycle(loader))
    bare_model = accelerator.unwrap_model(model)
    if ema:
        if checkpoint is not None:
            if "ema" not in checkpoint or checkpoint["ema"] is None:
                raise ValueError("Resume checkpoint is missing EMA state")
            ema.model.load_state_dict(checkpoint["ema"], strict=True)
            ema.updates = checkpoint["ema_updates"]
    evaluation_model = ema.model if ema else bare_model
    def run_validation():
        with accelerator.autocast():
            values = evaluate(evaluation_model, objective, val_loader, accelerator.device,
                              cfg["seed"] + 1001, monitor_cfg.get("validation_batches", 4))
            if ema and monitor_cfg.get("validate_raw", False):
                raw = evaluate(bare_model, objective, val_loader, accelerator.device,
                               cfg["seed"] + 1001, monitor_cfg.get("validation_batches", 4))
                values.update({f"raw_{name}": value for name, value in raw.items() if name != "examples"})
            return values
    if checkpoint is not None and "rng_states" in checkpoint:
        rng = checkpoint["rng_states"][accelerator.process_index]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if accelerator.device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"], accelerator.device)
    stop_step = min(cfg["n_steps"], getattr(args, "stop_after", None) or cfg["n_steps"])
    if start_step >= stop_step:
        accelerator.print(f"Checkpoint already reached step {start_step}: {run_dir}")
        return run_dir
    started_at = time.monotonic()

    def save_progress(step, phase="training"):
        if accelerator.is_main_process:
            elapsed = time.monotonic() - started_at
            speed = (step - start_step) / max(elapsed, 1e-9)
            state = {"step": step, "total_steps": cfg["n_steps"], "phase": phase,
                     "elapsed_seconds": elapsed, "steps_per_second": speed,
                     "images_seen": step * cfg["batch_size"] * accumulation * accelerator.num_processes,
                     "images_per_second": speed * cfg["batch_size"] * accumulation * accelerator.num_processes,
                     "eta_seconds": (cfg["n_steps"] - step) / speed if speed > 0 else None}
            temporary = run_dir / "progress.json.tmp"
            temporary.write_text(json.dumps(state))
            os.replace(temporary, run_dir / "progress.json")
            return state
        return {}


    save_progress(start_step, "preparing validation")
    monitor_cfg = cfg.get("monitoring", {})
    monitoring_enabled = monitor_cfg.get("enabled", False)
    monitor = TrainingMonitor(run_dir, cfg, start_step) if monitoring_enabled and accelerator.is_main_process else None
    val_loader = None
    if monitor:
        if args.fake_data:
            val_dataset = FakeData(size=max(cfg["batch_size"] * monitor_cfg.get("validation_batches", 4), 64),
                image_size=(cfg["model"]["in_channels"], cfg["image_size"], cfg["image_size"]),
                num_classes=cfg["model"]["num_classes"] or 1, transform=ToTensor(), random_offset=100_000)
        else:
            val_dataset = build_dataset(cfg, train=False)
        generator = torch.Generator().manual_seed(cfg["seed"] + 1000)
        indices = torch.randperm(len(val_dataset), generator=generator)[:
            cfg["batch_size"] * monitor_cfg.get("validation_batches", 4)]
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_dataset, indices.tolist()),
            batch_size=cfg["batch_size"], shuffle=False, num_workers=0, generator=generator)
        initial_validation = run_validation()
        monitor.validation({"step": start_step, "weights_source": "ema" if ema else "raw",
                            **initial_validation}, evaluation_model)
        monitor.render()
    accelerator.wait_for_everyone()
    model.train()
    totals = torch.zeros(7, device=accelerator.device)
    logged_steps = 0
    nfe_totals = {}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(start_step + 1, stop_step + 1), initial=start_step,
                    total=cfg["n_steps"], disable=not accelerator.is_main_process,
                    dynamic_ncols=True, desc=cfg["name"], unit="step")


    for step in progress:
        interval = objective.draw_interval(accelerator.device) if hasattr(objective, "draw_interval") else None
        batch_totals = torch.zeros(4, device=accelerator.device)
        for micro_index in range(accumulation):
            # Average gradients over the effective image batch. DDP communicates
            # once, after the last microbatch; optimizer/scheduler/EMA update once.
            context = accelerator.no_sync(model) if micro_index < accumulation - 1 else nullcontext()
            with context:
                x, labels = next(loader)
                extra = {"interval": interval} if interval is not None else {}
                micro_loss, micro_metrics = objective.loss(model, x, labels, derivative_model=bare_model, **extra)
                accelerator.backward(micro_loss / accumulation)
                batch_totals += torch.stack((micro_loss.detach(), micro_metrics["edge_loss"],
                                             micro_metrics["diag_loss"], micro_metrics["jvp_rms"])) / accumulation
        loss = batch_totals[0]
        metrics = dict(zip(("edge_loss", "diag_loss", "jvp_rms"), batch_totals[1:]))
        if interval is not None:
            values = nfe_totals.setdefault(interval[0], [0.0, 0])
            values[0] += float(loss)
            values[1] += 1
        log_now = step % cfg["log_step"] == 0 or step == stop_step or step == start_step + 1
        snapshot = None
        if monitor and log_now:
            scale = accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.0
            snapshot = gradient_snapshot(bare_model, scale=scale)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            if accelerator.is_main_process:
                bad = [name for name, parameter in bare_model.named_parameters()
                       if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                (run_dir / "failure.json").write_text(json.dumps({"step": step,
                    "loss": str(float(loss)), "grad_norm": str(float(grad_norm)),
                    "nonfinite_gradient_parameters": bad}, indent=2))
                save_progress(step, "nonfinite_loss_or_gradient")
            raise RuntimeError(f"Non-finite loss or gradient at step {step}")
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()
            if ema:
                ema.update(bare_model)
        gradient_groups = finish_gradient_snapshot(bare_model, snapshot) if snapshot else None
        del snapshot
        optimizer.zero_grad(set_to_none=True)
        clip_factor = (cfg["grad_clip"] / (grad_norm + 1e-6)).clamp(max=1.0)
        totals += torch.stack((loss.detach(), metrics["edge_loss"], metrics["diag_loss"],
                               metrics["jvp_rms"], grad_norm.detach(),
                               grad_norm.detach() * clip_factor, (grad_norm > cfg["grad_clip"]).float()))
        logged_steps += 1
        if log_now:
            means = accelerator.reduce(totals / logged_steps, reduction="mean").tolist()
            if accelerator.is_main_process:
                timing = save_progress(step)
                record = dict(step=step, lr=lr_used, window_steps=logged_steps,
                    elapsed_seconds=timing["elapsed_seconds"], steps_per_second=timing["steps_per_second"],
                    eta_seconds=timing["eta_seconds"],
                    images_seen=timing["images_seen"], images_per_second=timing["images_per_second"],
                    effective_batch_size=cfg["batch_size"] * accumulation * accelerator.num_processes,
                    **dict(zip(("loss", "edge_loss", "diag_loss", "jvp_rms", "grad_norm",
                                "grad_norm_postclip", "clip_fraction"), means)))
                if ema:
                    record.update(ema_updates=ema.updates, ema_decay=ema.decay)
                for nfe, (value, observations) in nfe_totals.items():
                    record[f"loss_nfe_{nfe}"] = value / observations
                    record[f"updates_nfe_{nfe}"] = observations
                if accelerator.device.type == "cuda":
                    record.update(gpu_allocated_gb=torch.cuda.memory_allocated() / 2**30,
                                  gpu_reserved_gb=torch.cuda.memory_reserved() / 2**30,
                                  gpu_peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
                    torch.cuda.reset_peak_memory_stats()
                if gradient_groups:
                    record["gradient_groups"] = gradient_groups
                    record["gradient_snapshot_step"] = step
                if monitor:
                    monitor.train(record)
                else:
                    with (run_dir / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                progress.set_postfix(loss=f"{means[0]:.4f}", grad=f"{means[4]:.3g}",
                                     lr=f"{lr_used:.2g}", refresh=False)
            totals.zero_()
            logged_steps = 0
            nfe_totals.clear()
        validation_now = monitoring_enabled and (step % monitor_cfg.get("validation_every", 1000) == 0 or step == stop_step)
        if validation_now:
            if monitor:
                save_progress(step, "validation")
                validation = run_validation()
                monitor.validation({"step": step, "weights_source": "ema" if ema else "raw", **validation}, evaluation_model)
            accelerator.wait_for_everyone()
        if step % cfg["sample_step"] == 0 or step == stop_step:
            if accelerator.is_main_process:
                n_per_class = monitor_cfg.get("sample_n_per_class", 1)
                labels = (None if objective.num_classes is None else
                          torch.arange(objective.num_classes, device=accelerator.device).repeat(n_per_class))
                n_samples = len(labels) if labels is not None else 10 * n_per_class
                sample_generator = torch.Generator(device=accelerator.device).manual_seed(cfg["seed"] + 2000)
                fixed_noise = torch.randn(n_samples, objective.channels, objective.image_size,
                    objective.image_size, device=accelerator.device, generator=sample_generator)
                for nfe in range(1, getattr(objective, "max_nfe", 1) + 1):
                    extra = {"nfe": nfe} if hasattr(objective, "max_nfe") else {}
                    with accelerator.autocast():
                        samples = objective.sample(evaluation_model, labels=labels,
                            device=accelerator.device, noise=fixed_noise, **extra)
                    suffix = "" if nfe == 1 else f"_nfe{nfe}"
                    save_image(make_grid(samples, nrow=cfg["sample_nrow"]),
                        run_dir / "images" / f"step_{step:07d}{suffix}.png")
                    if monitor:
                        monitor.samples(step, samples, nfe=nfe)
            accelerator.wait_for_everyone()
        if monitor and (step % monitor_cfg.get("plot_every", 200) == 0 or validation_now or step == stop_step):
            save_progress(step, "plotting")
            monitor.render()
        if monitoring_enabled:
            accelerator.wait_for_everyone()
        if step % cfg["checkpoint_step"] == 0 or step == stop_step:
            save_progress(step, "checkpointing")
            from accelerate.utils import gather_object
            rng_states = gather_object([{"torch": torch.get_rng_state(),
                "numpy": np.random.get_state(), "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state(accelerator.device)
                        if accelerator.device.type == "cuda" else None}])
            if accelerator.is_main_process:
                checkpoint_path = run_dir / "ckpts" / f"step_{step:07d}.pt"
                temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                accelerator.save({"model": bare_model.state_dict(), "step": step,
                    "ema": ema.model.state_dict() if ema else None,
                    "ema_updates": ema.updates if ema else 0,
                    "batcher": loader.state_dict() if resident else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "config": cfg, "trainable_parameters": count,
                    "synthetic_data": args.fake_data, "world_size": accelerator.num_processes,
                    "rng_states": rng_states,
                    "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None}, temporary_path)
                os.replace(temporary_path, checkpoint_path)
                keep = cfg.get("keep_last_checkpoints")
                if keep is not None and keep > 0:
                    for old_path in sorted((run_dir / "ckpts").glob("step_*.pt"))[:-keep]:
                        old_path.unlink()
                upload_due = hub_checkpoint_repo and (
                    step % hub_checkpoint_every == 0 or step == stop_step or step == cfg["n_steps"])
                if upload_due:
                    from hub import upload_training_checkpoint
                    save_progress(step, "uploading checkpoint to Hugging Face")
                    failure = None
                    for attempt in range(1, 4):
                        try:
                            receipt = upload_training_checkpoint(
                                hub_checkpoint_api, hub_checkpoint_repo, checkpoint_path, run_dir.name)
                        except Exception as error:  # preserve training through transient Hub/network failures
                            failure = error
                            if attempt < 3:
                                time.sleep(2 ** attempt)
                        else:
                            receipt.update(status="uploaded", attempts=attempt)
                            with (run_dir / "hub_checkpoints.jsonl").open("a") as stream:
                                stream.write(json.dumps(receipt) + "\n")
                            accelerator.print(f"Hub checkpoint verified: {receipt['url']}")
                            failure = None
                            break
                    if failure is not None:
                        event = {"status": "failed", "step": step, "checkpoint": checkpoint_path.name,
                                 "repo_id": hub_checkpoint_repo, "attempts": 3,
                                 "error_type": type(failure).__name__, "error": str(failure)}
                        with (run_dir / "hub_checkpoints.jsonl").open("a") as stream:
                            stream.write(json.dumps(event) + "\n")
                        accelerator.print(
                            f"WARNING: Hub checkpoint upload failed after 3 attempts; "
                            f"training continues and local checkpoint remains at {checkpoint_path}")
            accelerator.wait_for_everyone()
        if log_now or validation_now or step == stop_step:
            phase = "complete" if step == cfg["n_steps"] else "paused" if step == stop_step else "training"
            save_progress(step, phase)
    status = "Training complete" if stop_step == cfg["n_steps"] else f"Checkpoint saved at step {stop_step}"
    accelerator.print(f"{status}: {run_dir}")
    if monitor:
        monitor.close()
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_22m.py")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--run-suffix", "--run_suffix", dest="run_suffix")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--run-dir", help="Exact persistent directory for this run")
    parser.add_argument("--resume", help="Local trusted trainer checkpoint to restore")
    parser.add_argument("--stop-after", type=int, help="Stop at this global step, preserving the full LR schedule")
    parser.add_argument("--hub-checkpoint-repo", help="Hugging Face model repo for full resume checkpoints")
    parser.add_argument("--hub-checkpoint-every", type=int,
                        help="Upload every N optimizer updates; must be a checkpoint interval multiple")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for arg, key in ((args.steps, "n_steps"), (args.batch_size, "batch_size"),
                     (args.num_workers, "num_workers")):
        if arg is not None:
            cfg[key] = arg
    if cfg["n_steps"] <= 0 or cfg["batch_size"] <= 0 or cfg["num_workers"] < 0:
        parser.error("steps and batch size must be positive; workers must be nonnegative")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("stop-after must be positive")
    torch.manual_seed(cfg["seed"])
    model, count = build_model(cfg)
    print(f"Trainable parameters: {count:,} ({count / 1e6:.6f}M)", flush=True)
    if args.smoke_test:
        smoke_test(model, cfg)
    elif not args.check_model:
        run_training(model, count, cfg, args)


if __name__ == "__main__":
    main()
