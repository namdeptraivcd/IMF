# Cấu hình hiện tại — Multi-Trace U-Net 15M trên Modal

Snapshot này mô tả `notebooks/multitrace_imf_modal.ipynb` tại commit
`dc65c64c0e7237675b6584cff7c009f770d3860b` trên nhánh
`codex/modal-training`. Checkpoint 22M/DiT không tương thích với cấu hình này.

## Hạ tầng Modal

- GPU: 1 × Nvidia H200; code chỉ dùng `CUDA_VISIBLE_DEVICES=0`.
- Notebook source: `https://github.com/namdeptraivcd/IMF.git`.
- Volume: `imf-training`, mount tại `/mnt/imf-training`.
- Secret bắt buộc: `HF_TOKEN` có quyền write.
- CPU/RAM nằm trong Compute profile, không được lưu trong notebook. Notebook mặc
  định `NUM_WORKERS=2` và `OMP_NUM_THREADS=2`.

## Run

```python
REPO_REF = "codex/modal-training"
RUN_NAME = "multitrace_unet15m_h200_b512_v1"
STOP_AFTER_UPDATES = 1_000  # pilot; đổi thành None để train đủ schedule
AUTO_RESUME = True
RESUME_CHECKPOINT = None    # tự chọn checkpoint mới nhất
```

Pilot đã tạo checkpoint `step_0001000.pt`. Khi resume phải giữ nguyên
`RUN_NAME`, model, batch, accumulation và training schedule.

## Model

| Trường | Giá trị |
|---|---:|
| Architecture | `ConditionalMultiTraceUNet` |
| Trainable parameters | **14.992.182** |
| Input | CIFAR-10, RGB 32×32, 10 classes |
| Base channels | 69 |
| Channel multipliers | `(1, 2, 2, 4)` |
| Channels theo level | `69 / 138 / 138 / 276` |
| Residual blocks/level | 1 |
| Condition dimension | 312 |
| Fourier dimension | 64 |
| Attention resolutions | `(8,)`, kèm middle attention ở 4×4 |
| Output | Một shared velocity head `u` |

Parameter guard dùng target 15.000.000 và tolerance 1%; số đếm thực lệch mục
tiêu khoảng 0,0521%.

## Multi-Trace objective

```python
objective = "multi_trace_imf"
lambda_diag = 1.0
max_nfe = 5
jvp_precision = "bf16"
channels_last = True
```

- `K ~ Uniform{1,2,3,4,5}`.
- `j ~ Uniform{0,...,K-1}` và interval `[j/K, (j+1)/K]`.
- Một K/j được dùng cho cả optimizer update.
- Edge và diagonal dùng chung head; directional JVP được detach theo
  semi-gradient.
- Sampling hỗ trợ equal-grid reverse với 1–5 NFE.

## Batch và image budget

```python
BATCH_SIZE = 512
ACCUMULATION_STEPS = 1
WORLD_SIZE = 1
EFFECTIVE_BATCH = 512
TRAIN_STEPS = 100_000
```

Mỗi update dùng 512 ảnh và ghép edge + diagonal thành 1.024 trace inputs trong
forward. Tổng training budget là 51,2 triệu lượt ảnh. GPU smoke test chạy đúng
physical batch 512 trước khi tạo run thật.

Nếu đổi effective batch `B`, notebook tính:

```text
updates       = ceil(51.200.000 / B)
warmup        = ceil(2.560.000 / B)
ema_decay     = 0.9999 ** (B / 512)
```

## Optimizer và LR schedule

| Trường | Giá trị |
|---|---:|
| Optimizer | AdamW, fused trên CUDA |
| Betas | `(0.9, 0.99)` |
| Weight decay | `0.0` |
| Peak LR | `1e-4` |
| Minimum LR | `1e-5` (`min_lr_ratio=0.1`) |
| Warmup | 5.000 optimizer updates |
| Sau warmup | cosine decay tới `1e-5` |
| Global gradient clipping | **1.0** |
| EMA | `0.9999` mỗi optimizer update |
| Precision | BF16 AMP; JVP BF16 trên H200 |
| TF32 | bật |
| Memory layout | channels-last |
| CIFAR preload | bật trên GPU một process |

`grad_clip=1.0` theo notebook tham chiếu. Pilot step 1–1.000 có
`clip_fraction=1.0`; raw train/validation loss vẫn giảm rõ. Không đổi clipping
giữa run hiện tại vì đây là resume invariant.

## Monitoring

```python
LOG_EVERY = 50
PLOT_EVERY = 500
VALIDATION_EVERY = 1_000
VALIDATION_BATCHES = 5
SAMPLE_EVERY = 1_000
REFRESH_SECONDS = 10
```

- Ghi train loss, edge/diagonal loss, loss theo K, JVP RMS, LR, gradient trước và
  sau clip, clip fraction, update/weight theo nhóm, throughput, ETA và GPU memory.
- Fixed validation dùng raw và EMA weights, cố định data/time/noise.
- Fixed-noise samples dùng EMA, cùng labels/noise cho NFE 1–5.
- TensorBoard, dashboard, gradient heatmap, runtime plot và convergence report
  được lưu trên Volume.

## Checkpoint và Hugging Face

```python
CHECKPOINT_EVERY = 1_000
KEEP_LAST_CHECKPOINTS = 3
HF_CHECKPOINT_EVERY = 10_000
HF_REPO_ID = None
HF_PRIVATE = None
```

- Checkpoint local mỗi 1.000 updates, ghi file tạm rồi atomic rename.
- Checkpoint pilot, mỗi 10.000 updates và final được upload vào
  `training-checkpoints/<RUN_NAME>/` trên Hugging Face.
- Mỗi checkpoint chứa raw model, EMA, optimizer, scheduler, AMP scaler, RNG,
  data cursor, config, step và source manifest.
- Upload retry ba lần; lỗi Hub không dừng training và được ghi trong
  `hub_checkpoints.jsonl`.
- `HF_PRIVATE=None` giữ visibility của repo đã có; repo mới mặc định private.

## Final evaluation và export

```python
FID_NUM_GENERATED = 10_000
FID_BATCH_SIZE = 128
FID_NFES = [1, 2, 3, 4, 5]
FID_SEED = 2026
```

FID dùng `pytorch-fid==0.3.0`, real stats từ 50.000 CIFAR-10 train images không
augmentation, generated labels cân bằng và cùng noise/labels giữa các NFE. Sau
khi đủ 100.000 updates, notebook export final EMA `model.safetensors`, config,
source inference, logs, diagnostics, samples, best validation weights và FID lên
Hugging Face.

## Resume sau pilot

Giữ nguyên mọi setting, chỉ đổi:

```python
STOP_AFTER_UPDATES = None
```

Sau đó chạy lại cell Settings và cell pipeline. Trainer tự chọn
`/mnt/imf-training/runs/multitrace_unet15m_h200_b512_v1/ckpts/step_0001000.pt`
và tiếp tục từ update 1.001.
