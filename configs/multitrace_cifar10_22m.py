"""22M U-Net profile derived from the supplied 5.947M notebook."""
import math


def scale_training_budget(micro_batch=128, accumulation=4, world_size=1):
    effective = micro_batch * accumulation * world_size
    if min(micro_batch, accumulation, world_size) <= 0:
        raise ValueError("Effective batch must be positive")
    return dict(batch_size=micro_batch, gradient_accumulation_steps=accumulation,
                n_steps=math.ceil(51_200_000 / effective),
                warmup_steps=math.ceil(2_560_000 / effective),
                ema_decay=0.9999 ** (effective / 512))


config = dict(
    name="multitrace_unet_cifar10_22m", architecture="ConditionalMultiTraceUNet",
    objective="multi_trace_imf", seed=42, dataset="cifar10", data_root="data/cifar10",
    image_size=32, num_workers=4, lr=1e-4, min_lr_ratio=0.1,
    betas=(0.9, 0.99), weight_decay=0.0, grad_clip=1.0,
    mixed_precision="bf16", channels_last=True, allow_tf32=True,
    fused_optimizer=True, preload_gpu=True,
    log_step=50, sample_step=1000, checkpoint_step=1000,
    keep_last_checkpoints=3, sample_nrow=10,
    parameter_target=22_000_000, parameter_tolerance=0.01,
    model=dict(input_size=32, in_channels=3, num_classes=10,
               base_channels=84, channel_mult=(1, 2, 2, 4), num_res_blocks=1,
               cond_dim=356, fourier_dim=64, attn_resolutions=(8,)),
    multi_trace_imf=dict(lambda_diag=1.0, max_nfe=5, jvp_precision="bf16", channels_last=True),
    monitoring=dict(enabled=True, tensorboard=True, save_best=True, validate_raw=True,
        plot_every=500, validation_every=1000, validation_batches=5,
        sample_n_per_class=2),
    evaluation=dict(num_generated=10_000, batch_size=128, nfes=(1, 2, 3, 4, 5), seed=2026),
    **scale_training_budget(),
)
