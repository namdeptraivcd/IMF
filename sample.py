"""Sample the exported custom TraceDiT inference bundle without loading pickle."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from models import build_backbone
from objectives import build_objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="samples.png")
    parser.add_argument("--n-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="model.safetensors")
    parser.add_argument("--nfe", type=int, default=1)
    args = parser.parse_args()
    if args.n_per_class <= 0:
        parser.error("n-per-class must be positive")
    folder = Path(args.model_dir)
    cfg = json.loads((folder / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = build_backbone(cfg).to(device)
    model.load_state_dict(load_file(str(folder / args.weights), device=device), strict=True)
    objective = build_objective(cfg)
    if args.nfe != 1 and not hasattr(objective, "max_nfe"):
        parser.error("This model supports only one NFE")
    if objective.num_classes is None:
        labels, columns = None, 10
    else:
        columns = objective.num_classes
        labels = torch.arange(columns, device=device).repeat(args.n_per_class)
    images = objective.sample(model, n_samples=columns * args.n_per_class,
                              labels=labels, device=device,
                              **({"nfe": args.nfe} if hasattr(objective, "max_nfe") else {}))
    pixels = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    rows = args.n_per_class
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    grid = np.concatenate([np.concatenate(pixels[row*columns:(row+1)*columns], axis=1)
                           for row in range(rows)], axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output)
    print(f"{args.nfe}-NFE samples saved: {output}")


if __name__ == "__main__":
    main()
