"""Final EMA FID with the standard pytorch-fid Inception pool3 feature extractor."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def feature_statistics(extractor, batches, count, device, description):
    features = np.empty((count, 2048), dtype=np.float64)
    position = 0
    with tqdm(total=count, desc=description, unit="image", dynamic_ncols=True) as progress:
        for images in batches:
            values = extractor(images.to(device).float())[0].flatten(1).cpu().numpy()
            end = position + len(values)
            if end > count or values.shape[1] != 2048 or not np.isfinite(values).all():
                raise ValueError("Invalid FID feature batch")
            features[position:end] = values
            progress.update(len(values))
            position = end
    if position != count:
        raise ValueError(f"Expected {count} FID images, got {position}")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def evaluate_checkpoint(checkpoint_path, cache_dir):
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    from data import build_dataset
    from models import build_backbone
    from objectives import build_objective

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if checkpoint.get("synthetic_data") is not False or checkpoint["step"] != cfg["n_steps"]:
        raise ValueError("FID requires a completed real-data training checkpoint")
    settings = cfg["evaluation"]
    count, batch_size = settings["num_generated"], settings["batch_size"]
    nfes = list(settings["nfes"])
    if count < 2 or batch_size < 1:
        raise ValueError("Invalid evaluation sample count or batch size")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(cfg).to(device).eval()
    source = "ema" if checkpoint.get("ema") is not None else "model"
    model.load_state_dict(checkpoint[source], strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(checkpoint[source].items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    weight_hash = digest.hexdigest()
    del checkpoint
    objective = build_objective(cfg)
    if objective.num_classes is None or count % objective.num_classes:
        raise ValueError("Balanced FID requires generated_count divisible by num_classes")
    if any(k < 1 or k > getattr(objective, "max_nfe", 1) for k in nfes):
        raise ValueError("Evaluation NFE outside the trained range")
    extractor = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
                            use_fid_inception=True).to(device).eval()
    # pytorch-fid performs bilinear resize to 299 and [0,1] -> [-1,1].
    # Its converted TensorFlow Inception weights differ from torchvision ImageNet weights.
    dataset = build_dataset(cfg, train=True, augment=False)
    dataset_hash = hashlib.sha256(dataset.data.tobytes()).hexdigest()
    protocol = dict(implementation="pytorch-fid==0.3.0", features="FID Inception pool3 (2048)",
        input_range=[0, 1], real_split="CIFAR-10 train, unaugmented", real_count=len(dataset),
        real_data_sha256=dataset_hash, generated_count=count, seed=settings["seed"],
        batch_size=batch_size, class_schedule=f"balanced labels: sample index modulo {objective.num_classes}",
        sampling_precision="bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32",
        weights_source=source, weights_sha256=weight_hash, training_step=cfg["n_steps"], nfes=nfes,
        reference_notebook_comparable=False)
    report_path = path.parent.parent / "fid.json"
    result = dict(protocol=protocol, fid={})
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["protocol"] == protocol:
            result = previous
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    real_path = cache_dir / f"cifar10_train_{dataset_hash}_fid_pool3_v0.3.0.npz"
    if real_path.exists():
        with np.load(real_path) as stats:
            real_mu, real_cov = stats["mu"], stats["cov"]
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=cfg.get("num_workers", 0), pin_memory=device.type == "cuda")
        real_mu, real_cov = feature_statistics(extractor, (x for x, _ in loader),
                                               len(dataset), device, "FID real features")
        temporary = real_path.with_suffix(".tmp.npz")
        np.savez(temporary, mu=real_mu, cov=real_cov)
        os.replace(temporary, real_path)
    for nfe in nfes:
        if str(nfe) in result["fid"]:
            print(f"Reuse completed FID {nfe} NFE: {result['fid'][str(nfe)]}", flush=True)
            continue
        generator = torch.Generator(device=device).manual_seed(settings["seed"])

        def generated_batches():
            for offset in range(0, count, batch_size):
                size = min(batch_size, count - offset)
                labels = torch.arange(offset, offset + size, device=device) % objective.num_classes
                noise = torch.randn(size, objective.channels, objective.image_size,
                    objective.image_size, device=device, generator=generator)
                with torch.autocast(device_type=device.type,
                    enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(), dtype=torch.bfloat16):
                    images = objective.sample(model, labels=labels, device=device, noise=noise,
                        **({"nfe": nfe} if hasattr(objective, "max_nfe") else {}))
                yield images.float()

        mu, cov = feature_statistics(extractor, generated_batches(), count, device, f"FID {nfe} NFE")
        score = float(calculate_frechet_distance(real_mu, real_cov, mu, cov))
        if not np.isfinite(score):
            raise RuntimeError("Non-finite FID")
        result["fid"][str(nfe)] = score
        _write_json(report_path, result)
        print(f"FID {nfe} NFE ({count:,} generated): {score:.4f}", flush=True)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.cache_dir)
