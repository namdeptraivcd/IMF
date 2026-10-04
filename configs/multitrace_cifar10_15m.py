"""15M U-Net profile for the H200 Modal notebook."""
from copy import deepcopy

from configs.multitrace_cifar10_22m import config as _config_22m
from configs.multitrace_cifar10_22m import scale_training_budget


config = deepcopy(_config_22m)
config.update(
    name="multitrace_unet_cifar10_15m",
    parameter_target=15_000_000,
    **scale_training_budget(micro_batch=512, accumulation=1),
)
config["model"].update(base_channels=69, cond_dim=312)
