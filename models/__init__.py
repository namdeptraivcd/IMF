from .dit import TraceDiT
from .unet import ConditionalMultiTraceUNet


def build_backbone(cfg):
    architecture = cfg.get("architecture", "TraceDiT")
    classes = {"TraceDiT": TraceDiT, "ConditionalMultiTraceUNet": ConditionalMultiTraceUNet}
    if architecture not in classes:
        raise ValueError(f"Unknown architecture: {architecture}")
    return classes[architecture](**cfg["model"])
