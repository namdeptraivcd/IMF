"""Convergence diagnostics and notebook progress from durable training logs."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The writer may still be appending the last line.
    return result


def rollback_logs(run_dir, step):
    """Drop metrics after the restored checkpoint so curves keep a single history."""
    for name in ("train.jsonl", "validation.jsonl", "samples.jsonl"):
        path = Path(run_dir) / name
        if path.exists():
            records = [r for r in read_records(path) if r.get("step", 0) <= step]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
    best = Path(run_dir) / "best_validation.json"
    if best.exists() and json.loads(best.read_text())["step"] > step:
        best.unlink()
        (Path(run_dir) / "best_validation.safetensors").unlink(missing_ok=True)


def group_name(name):
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]


def gradient_snapshot(model, scale=1.0):
    """Measure unscaled pre-clip gradients; clone weights only on diagnostic steps."""
    groups, before = {}, {}
    for name, parameter in model.named_parameters():
        group = groups.setdefault(group_name(name), {"weight_sq": 0.0, "grad_sq": 0.0,
            "numel": 0, "missing_grad_tensors": 0, "zero_grad_tensors": 0})
        before[name] = parameter.detach().clone()
        group["numel"] += parameter.numel()
        group["weight_sq"] += parameter.detach().float().square().sum()
        if parameter.grad is None:
            group["missing_grad_tensors"] += 1
        else:
            grad = parameter.grad.detach().float() / scale
            group["grad_sq"] += grad.square().sum()
            group["zero_grad_tensors"] += int(torch.count_nonzero(grad) == 0)
    return groups, before


def finish_gradient_snapshot(model, snapshot):
    groups, before = snapshot
    update_sq = {name: 0.0 for name in groups}
    for name, parameter in model.named_parameters():
        update_sq[group_name(name)] += (parameter.detach().float() - before[name].float()).square().sum()
    result = {}
    for name, stats in groups.items():
        weight_norm = math.sqrt(float(stats["weight_sq"]))
        grad_norm = math.sqrt(float(stats["grad_sq"]))
        update_norm = math.sqrt(float(update_sq[name]))
        result[name] = {"weight_norm": weight_norm, "grad_norm": grad_norm,
            "grad_rms": grad_norm / math.sqrt(stats["numel"]),
            "grad_to_weight": grad_norm / weight_norm if weight_norm > 0 else None,
            "update_to_weight": update_norm / weight_norm if weight_norm > 0 else None,
            "missing_grad_tensors": stats["missing_grad_tensors"],
            "zero_grad_tensors": stats["zero_grad_tensors"]}
    return result


@torch.no_grad()
def evaluate(model, objective, loader, device, seed, max_batches):
    """Fixed held-out images, times and noise; do not consume the training RNG."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"loss": 0.0, "edge_loss": 0.0, "diag_loss": 0.0, "jvp_rms": 0.0}
    examples = 0
    try:
        for index, (images, labels) in enumerate(loader):
            if index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            times = torch.rand(images.shape[0], device=device, generator=generator)
            noise = torch.randn(images.shape, device=device, generator=generator)
            loss, metrics = objective.loss(model, images, labels, derivative_model=model,
                                           t=times, noise=noise)
            values = {"loss": loss, **metrics}
            for key in totals:
                totals[key] += float(values[key]) * images.shape[0]
            examples += images.shape[0]
    finally:
        model.train(was_training)
    if not examples:
        raise ValueError("Validation loader produced no examples")
    return {key: value / examples for key, value in totals.items()} | {"examples": examples}


def convergence_summary(train, validation, total_steps, window=5, plateau_fraction=0.01):
    """Describe evidence, without declaring convergence or stopping the optimizer."""
    summary = {"status": "insufficient_validation_history", "required_evaluations": window,
               "warning": "Regression loss has an irreducible noise floor; these are diagnostic hints, not FID."}
    if not validation:
        return summary
    latest = validation[-1]
    best = min(validation, key=lambda row: row["loss"])
    summary.update(step=latest["step"], best_validation_step=best["step"],
                   best_validation_loss=best["loss"], latest_validation_loss=latest["loss"])
    if len(validation) < window:
        return summary
    recent = validation[-window:]
    improvement = (recent[0]["loss"] - min(r["loss"] for r in recent[1:])) / max(abs(recent[0]["loss"]), 1e-12)
    phase = "early" if latest["step"] < total_steps * 0.25 else "late" if latest["step"] > total_steps * 0.75 else "middle"
    summary.update(relative_improvement=improvement, window_start_step=recent[0]["step"],
                   window_end_step=latest["step"], budget_phase=phase,
                   status=f"plateau_candidate_{phase}" if improvement < plateau_fraction else "improving")
    summary["validation_regressed_from_best"] = latest["loss"] > best["loss"] * (1 + plateau_fraction)
    if train:
        summary["recent_clip_fraction"] = float(np.mean([row.get("clip_fraction", 0) for row in train[-window:]]))
    return summary


class TrainingMonitor:
    def __init__(self, run_dir, cfg, start_step=0):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.writer = None
        (self.run_dir / "diagnostics").mkdir(exist_ok=True)
        if cfg.get("monitoring", {}).get("tensorboard", True):
            from torch.utils.tensorboard import SummaryWriter
            segment = f"segment_{start_step:07d}_{time.time_ns()}"
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard" / segment))

    def train(self, record):
        with (self.run_dir / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.scalars("train", record)

    def validation(self, record, model):
        previous = read_records(self.run_dir / "validation.jsonl")
        records = [row for row in previous if row["step"] != record["step"]] + [record]
        temporary = self.run_dir / "validation.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in sorted(records, key=lambda row: row["step"])))
        temporary.replace(self.run_dir / "validation.jsonl")
        self.scalars("validation", record)
        if self.cfg.get("monitoring", {}).get("save_best", True) and (
                not (self.run_dir / "best_validation.safetensors").exists() or
                not previous or record["loss"] < min(row["loss"] for row in previous)):
            from safetensors.torch import save_file
            path = self.run_dir / "best_validation.safetensors"
            temporary = path.with_suffix(".safetensors.tmp")
            save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(temporary))
            temporary.replace(path)
            (self.run_dir / "best_validation.json").write_text(json.dumps(record, indent=2))

    def samples(self, step, images):
        record = {"step": step, "pixel_mean": float(images.mean()),
                  "pixel_std": float(images.std()),
                  "between_samples_std": float(images.std(dim=0).mean())}
        with (self.run_dir / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.scalars("samples", record)
        if self.writer:
            self.writer.add_images("samples/fixed_noise", images, step)

    def scalars(self, phase, record):
        if not self.writer:
            return
        step = record["step"]
        for name, value in record.items():
            if name != "step" and isinstance(value, (int, float)) and math.isfinite(value):
                self.writer.add_scalar(f"{phase}/{name}", value, step)
        for group, values in record.get("gradient_groups", {}).items():
            for name, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar(f"gradients/{group}/{name}", value, step)
        self.writer.flush()

    def render(self):
        return render_dashboard(self.run_dir, self.cfg["n_steps"])

    def close(self):
        if self.writer:
            self.writer.close()


def render_dashboard(run_dir, total_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    destination = root / "diagnostics"
    destination.mkdir(exist_ok=True)
    train = [r for r in read_records(root / "train.jsonl") if "loss" in r]
    validation = read_records(root / "validation.jsonl")
    samples = read_records(root / "samples.jsonl")
    summary = convergence_summary(train, validation, total_steps)
    (destination / "convergence.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    axes = axes.ravel()

    def lines(axis, rows, keys):
        for key in keys:
            points = [(r["step"], r[key]) for r in rows if key in r]
            if points:
                axis.plot(*zip(*points), label=key, marker=".", markersize=4, linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_xlabel("optimizer step")
        if axis.lines:
            axis.legend(fontsize=8)

    lines(axes[0], train, ["loss", "edge_loss", "diag_loss"])
    axes[0].set_title("Training losses (window means)")
    lines(axes[1], validation, ["loss", "edge_loss", "diag_loss"])
    axes[1].set_title("Held-out losses: fixed data, t and noise")
    lines(axes[2], train, ["grad_norm", "grad_norm_postclip"])
    axes[2].set_yscale("symlog", linthresh=1e-7)
    axes[2].set_title("Global gradient norms, before/after clipping")
    lines(axes[3], train, ["clip_fraction"])
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Fraction of training steps clipped")
    lines(axes[4], train, ["lr"])
    axes[4].set_title("Learning rate")
    lines(axes[5], train, ["jvp_rms"])
    axes[5].set_yscale("symlog", linthresh=1e-7)
    axes[5].set_title("JVP RMS")
    for group in sorted({g for r in train for g in r.get("gradient_groups", {})}):
        points = [(r["step"], r["gradient_groups"][group]["update_to_weight"])
                  for r in train if group in r.get("gradient_groups", {})]
        axes[6].plot(*zip(*points), label=group, marker=".", markersize=3, linewidth=1)
    axes[6].set_yscale("symlog", linthresh=1e-8)
    axes[6].set_title("Measured optimizer update / weight by group")
    axes[6].set_xlabel("optimizer step")
    if axes[6].lines:
        axes[6].legend(fontsize=6, ncol=2)
    lines(axes[7], samples, ["pixel_std", "between_samples_std"])
    axes[7].set_title("Fixed-noise sample spread (not a quality score)")
    path = destination / "dashboard.png"
    temporary_plot = path.with_suffix(".png.tmp")
    fig.savefig(temporary_plot, dpi=120, format="png")
    temporary_plot.replace(path)
    plt.close(fig)
    diagnostic_rows = [r for r in train if r.get("gradient_groups")]
    if diagnostic_rows:
        groups = sorted({group for row in diagnostic_rows for group in row["gradient_groups"]})
        values = [[np.log10(max(row["gradient_groups"].get(group, {}).get("grad_rms", 0), 1e-12))
                   for row in diagnostic_rows] for group in groups]
        fig, ax = plt.subplots(figsize=(13, max(4, len(groups)*0.4)), constrained_layout=True)
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(groups)), groups)
        ticks = np.linspace(0, len(diagnostic_rows)-1, min(8, len(diagnostic_rows)), dtype=int)
        ax.set_xticks(ticks, [diagnostic_rows[index]["step"] for index in ticks])
        ax.set_xlabel("optimizer step")
        ax.set_title("Per-group pre-clip gradient RMS: log10")
        fig.colorbar(image, ax=ax)
        fig.savefig(destination / "gradient_heatmap.png", dpi=120)
        plt.close(fig)
    flat_rows = []
    for record in train:
        flat = {key: value for key, value in record.items() if key != "gradient_groups"}
        for group, values in record.get("gradient_groups", {}).items():
            flat.update({f"{group}/{key}": value for key, value in values.items()})
        flat_rows.append(flat)
    if flat_rows:
        with (destination / "training.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat_rows for key in row}))
            writer.writeheader()
            writer.writerows(flat_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    lines(axes[0], train, ["steps_per_second"])
    axes[0].set_title("Measured throughput, including monitoring overhead")
    lines(axes[1], train, ["eta_seconds"])
    axes[1].set_title("Estimated remaining seconds")
    lines(axes[2], train, ["gpu_allocated_gb", "gpu_reserved_gb", "gpu_peak_allocated_gb"])
    axes[2].set_title("GPU allocated / reserved / peak GiB")
    fig.savefig(destination / "runtime.png", dpi=120)
    plt.close(fig)
    return path


def run_with_dashboard(command, cwd, env, run_dir, total_steps, refresh_seconds=10):
    """Notebook-native widget progress; subprocess drains stdout independently."""
    import collections
    import html
    import subprocess
    import threading
    import ipywidgets as widgets
    from IPython.display import display, Image

    progress = widgets.IntProgress(value=0, min=0, max=total_steps, description="Training")
    status = widgets.HTML("Starting trainer…")
    log_output = widgets.HTML()
    plot_output = widgets.Output()
    display(widgets.VBox([progress, status, log_output, plot_output]))
    logs = collections.deque(maxlen=200)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    def drain():
        for line in process.stdout:
            logs.append(line.rstrip())
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    previous_plot = None
    try:
        while process.poll() is None:
            state = read_records(Path(run_dir) / "progress.json")
            if state:
                row = state[-1]
                progress.value = min(row["step"], total_steps)
                eta = row.get("eta_seconds")
                eta_text = f"{eta/3600:.2f} h" if eta is not None else "estimating"
                status.value = (f"Step {row['step']:,}/{total_steps:,} · "
                    f"{row.get('steps_per_second', 0):.3f} step/s · "
                    f"ETA {eta_text} · {html.escape(row.get('phase', 'training'))}")
            log_output.value = "<pre>" + html.escape("\n".join(list(logs)[-8:])) + "</pre>"
            plot = Path(run_dir) / "diagnostics/dashboard.png"
            if plot.exists() and plot.stat().st_mtime_ns != previous_plot:
                previous_plot = plot.stat().st_mtime_ns
                with plot_output:
                    plot_output.clear_output(wait=True)
                    display(Image(filename=str(plot)))
            time.sleep(refresh_seconds)
        reader.join(timeout=5)
        if process.returncode:
            progress.bar_style = "danger"
            status.value = "Trainer failed; check the traceback and failure.json on the Volume"
            log_output.value = "<pre>" + html.escape("\n".join(logs)) + "</pre>"
            raise subprocess.CalledProcessError(process.returncode, command, output="\n".join(logs))
        state = read_records(Path(run_dir) / "progress.json")
        progress.value = min(state[-1]["step"], total_steps) if state else total_steps
        progress.bar_style = "success" if progress.value == total_steps else "info"
        status.value = "Training complete; preparing Hub upload" if progress.value == total_steps else "Checkpoint saved; ready to resume"
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
