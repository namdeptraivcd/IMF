"""Objective selection shared by training and the standalone inference bundle."""
from trace_imf import TraceIMF
from multi_trace_imf import MultiTraceIMF


def build_objective(cfg):
    common = dict(channels=cfg["model"]["in_channels"],
                  image_size=cfg["model"]["input_size"],
                  num_classes=cfg["model"]["num_classes"])
    name = cfg.get("objective", "trace_imf")
    if name == "multi_trace_imf":
        return MultiTraceIMF(**common, **cfg["multi_trace_imf"])
    if name == "trace_imf":
        return TraceIMF(**common, **cfg["trace_imf"])
    raise ValueError(f"Unknown objective: {name}")
