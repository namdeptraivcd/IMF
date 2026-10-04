"""Export completed Trace-iMF runs and upload a loadable model bundle to the Hub."""
import json
import shutil
from pathlib import Path

import torch


def prepare_repository(repo_id, token, private=True):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    else:
        if info.private != private:
            raise ValueError("Existing repo visibility differs from HF_PRIVATE; set HF_PRIVATE to match")
    return api


def export_model(checkpoint_path, output_dir, source_dir=None):
    """Export a trusted LOCAL trainer checkpoint; require a completed real-data run."""
    from safetensors.torch import load_file, save_file
    from models import build_backbone

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if checkpoint["step"] != cfg["n_steps"]:
        raise ValueError("Refusing final model export before all configured training steps finish")
    if checkpoint.get("synthetic_data") is not False:
        raise ValueError("Final Hub export requires a real-data checkpoint with synthetic_data=False")
    source_dir = Path(source_dir or Path(__file__).parent).resolve()
    output_dir = Path(output_dir).resolve()
    run_dir = checkpoint_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    source = "ema" if checkpoint.get("ema") is not None else "model"
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in checkpoint[source].items()}
    save_file(weights, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    model = build_backbone(cfg)
    model.load_state_dict(load_file(str(output_dir / "model.safetensors")), strict=True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != checkpoint["trainable_parameters"]:
        raise ValueError("Checkpoint parameter count does not match the exported architecture")
    objective = cfg.get("objective", "trace_imf")
    model_config = {"architecture": cfg.get("architecture", "TraceDiT"), "model": cfg["model"],
        "objective": objective, objective: cfg[objective], "dataset": cfg["dataset"],
        "weights_source": source, "max_nfe": cfg.get("multi_trace_imf", {}).get("max_nfe", 1),
        "training_steps": checkpoint["step"], "trainable_parameters": count,
        "pixel_range": [0, 1], "sampling_steps": 1}
    (output_dir / "config.json").write_text(json.dumps(model_config, indent=2))
    (output_dir / "training_config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("models/dit.py", "models/unet.py", "models/__init__.py", "trace_imf.py",
                 "multi_trace_imf.py", "objectives.py", "sample.py",
                 "LICENSE", "third_party/MeanFlow.LICENSE", "third_party/README.md"):
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / name, destination)
    (output_dir / "requirements.txt").write_text(
        "torch>=2.2\nnumpy>=1.24\nsafetensors>=0.4\npillow>=9\nhuggingface_hub>=0.21\n")
    if (run_dir / "train.jsonl").exists():
        shutil.copyfile(run_dir / "train.jsonl", output_dir / "training_log.jsonl")
        records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines() if line.strip()]
        metrics = next((record for record in reversed(records) if record.get("step") == checkpoint["step"]), {})
        (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    sample_path = run_dir / "images" / f"step_{checkpoint['step']:07d}.png"
    if sample_path.exists():
        shutil.copyfile(sample_path, output_dir / "sample_grid.png")
    for nfe in range(2, model_config["max_nfe"] + 1):
        sample_path = run_dir / "images" / f"step_{checkpoint['step']:07d}_nfe{nfe}.png"
        if sample_path.exists():
            shutil.copyfile(sample_path, output_dir / f"sample_grid_nfe{nfe}.png")
    if (run_dir / "source_manifest.json").exists():
        shutil.copyfile(run_dir / "source_manifest.json", output_dir / "source_manifest.json")
    for name in ("validation.jsonl", "samples.jsonl", "best_validation.json", "best_validation.safetensors", "fid.json"):
        if (run_dir / name).is_file():
            shutil.copyfile(run_dir / name, output_dir / name)
    for name in ("diagnostics", "tensorboard"):
        if (run_dir / name).is_dir():
            shutil.copytree(run_dir / name, output_dir / name, dirs_exist_ok=True)
    (output_dir / "README.md").write_text(model_card(model_config))
    return output_dir


def model_card(cfg):
    if cfg.get("objective") == "multi_trace_imf":
        return f'''---
library_name: pytorch
license: mit
tags: [meanflow, multi-trace, image-generation, class-conditional, cifar10]
---

# Multi-Trace U-Net — {cfg['trainable_parameters']:,} parameters

U-Net adapted from the user-provided Untitled0.ipynb and widened to approximately
22M parameters. Trained for {cfg['training_steps']:,} optimizer updates on CIFAR-10.
Training samples K uniformly from 1..{cfg['max_nfe']} and a grid interval j/K;
the same u head fits the interval and diagonal, with a detached directional JVP.
Default inference weights are **{cfg['weights_source']}**. Model input/output pixels
are 32x32 RGB; sampling maps the final [-1,1] image into [0,1].

This is a custom PyTorch model. Download with huggingface_hub.snapshot_download,
install the included requirements.txt, then run:

```bash
python sample.py --model-dir . --nfe 1 --output samples_1nfe.png
python sample.py --model-dir . --nfe 5 --output samples_5nfe.png
```

config.json describes architecture/objective; training_config.json records the
effective batch, accumulation, LR schedule, clipping and EMA decay.
Optimizer/EMA/RNG resume checkpoints stay on the Modal Volume.
best_validation.safetensors contains the selected EMA validation snapshot;
its step and fixed held-out losses are recorded in best_validation.json.
Diagnostics include gradient RMS, actual optimizer updates, losses by K,
clipping frequency, samples, throughput, ETA and TensorBoard events.
Plateau flags are heuristics, not proof of convergence or image quality.

When present, fid.json reports pytorch-fid pool3 FID against 50,000 unaugmented
CIFAR-10 training images, with the generated sample count and each NFE explicitly
recorded. This protocol differs from the reference notebook's ImageNet-feature
FID; the values must not be compared directly. No quality claim is made here.
The test split was used as fixed validation during training.

Labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The U-Net and interval schedule follow the user-provided notebook; repository
training flow and DiT baseline are adapted from haidog-yaqub/MeanFlow (MIT).
See third_party/ for attribution.
'''
    return f'''---
library_name: pytorch
license: mit
tags:
  - trace-imf
  - meanflow
  - image-generation
  - class-conditional
  - cifar10
---

# Trace-iMF - {cfg['trainable_parameters']:,} trainable parameters

Class-conditioned DiT trained on `{cfg['dataset']}` for {cfg['training_steps']:,}
steps with the two-trace quadratic Trace-iMF objective. One shared u head fits
the left edge r=0 and diagonal r=t. Time is uniform; the JVP outcome is detached.
Diagonal loss weight: {cfg['trace_imf']['lambda_diag']}.

This is a custom PyTorch model. Use the included source and `sample.py`;
it is not an AutoModel or a Diffusers pipeline. No FID or comparative quality
claim has been established. The training logs and sample grid are included
when available. Images are 32x32 CIFAR-10; use the model for research.

## Download and sample

```python
from huggingface_hub import snapshot_download
folder = snapshot_download(repo_id="YOUR_USERNAME/YOUR_MODEL")
```

```bash
python -m pip install -r /path/to/download/requirements.txt
python /path/to/download/sample.py --model-dir /path/to/download --output samples.png
```

For a private repository, provide HF_TOKEN in your environment before download.
`model.safetensors` includes model weights and fixed positional buffers.
`config.json` describes the architecture; `training_config.json` records the
training hyperparameters. Optimizer checkpoints remain on the Modal Volume.
They are not included in this inference bundle.

When available, diagnostics/ contains dashboards, gradient heatmaps, runtime
plots, CSV and a convergence report. TensorBoard events are in tensorboard/.
best_validation.safetensors is an optional alternate set of weights; its
actual selection step and held-out metrics are recorded in best_validation.json.
Generate from those weights with `sample.py --weights best_validation.safetensors`.
Fixed-noise validation and plateau flags are diagnostic evidence, not proof of
convergence or image quality. The CIFAR-10 test split is used for validation;
reported final quality would require a separate evaluation protocol.

Generation uses X = E-u(E,t=1,r=0), one evaluation, then clamps to [-1,1]
and maps back to [0,1]. CIFAR-10 label order: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck.

## Attribution

DiT and training flow are adapted from haidog-yaqub/MeanFlow, MIT license;
see third_party/MeanFlow.LICENSE. Trace-iMF follows the user's theory note.
'''


def upload_model(api, repo_id, folder):
    """Publish only the inference bundle, then verify required files at its commit."""
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
        commit_message="Upload completed Trace-iMF training run",
        allow_patterns=["model.safetensors", "config.json", "README.md", "sample.py",
            "trace_imf.py", "multi_trace_imf.py", "objectives.py", "models/*.py", "third_party/*", "LICENSE", "requirements.txt",
            "training_config.json", "training_metrics.json", "training_log.jsonl",
            "sample_grid*.png", "source_manifest.json", "validation.jsonl", "samples.jsonl",
            "best_validation.json", "best_validation.safetensors", "fid.json", "diagnostics/*", "tensorboard/**"])
    required = {"model.safetensors", "config.json", "sample.py", "models/dit.py", "trace_imf.py"}
    if (Path(folder) / "config.json").exists():
        required |= {"models/__init__.py", "models/unet.py", "objectives.py", "multi_trace_imf.py"}
    present = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit.oid))
    if not required <= present:
        raise RuntimeError(f"Hub commit is missing required files: {sorted(required - present)}")
    return f"https://huggingface.co/{repo_id}/tree/{commit.oid}"
