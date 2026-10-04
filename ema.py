"""EMA updates once per successful optimizer update, never per microbatch."""
import copy
import torch


class ModelEMA:
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        current = dict(model.named_parameters())
        for name, value in self.model.named_parameters():
            value.lerp_(current[name].detach(), 1 - self.decay)
        for name, value in self.model.named_buffers():
            value.copy_(dict(model.named_buffers())[name])
        self.updates += 1
