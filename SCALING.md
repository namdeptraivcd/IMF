# U-Net Multi-Trace 15M từ Untitled0.ipynb

Nguồn tham chiếu là notebook người dùng cung cấp, không phải config mặc định
DiT của backbone_IMF. Profile chạy mặc định nằm ở `configs/multitrace_cifar10_15m.py`;
workflow Modal là `notebooks/multitrace_imf_modal.ipynb`. Baseline DiT cũ vẫn dùng
`configs/cifar10_22m.py`; profile U-Net 22M trước đó được giữ tại
`configs/multitrace_cifar10_22m.py`. Checkpoint 15M/22M/DiT không tương thích.

Đây là **Multi-Trace mở rộng cho 1–5 NFE**, không phải objective chuyên one-step
hai trace `r=0/r=t` trong mục10/14 của idea.pdf. K=1 vẫn có xác suất1/5 nên giữ
một liên hệ với one-step residual control; composition nhiều bước cần phân
tích riêng. Đổi DiT thành U-Net không thay lập luận này. Chi tiết đối chiếu
formulation ở `DIT_PROPOSAL_ALIGNMENT.md`; lựa chọn tiếp tục hiện tại là U-Net.

## So sánh cấu hình

| Thành phần | Notebook tham chiếu | U-Net 15M mặc định | Lý do |
|---|---:|---:|---|
| Trainable parameters | 5.946.579 | **14.992.182** | Đếm trực tiếp; 2,52× reference, lệch mục tiêu 0,0521% |
| Base channels | 44 | 69 | Widen backbone, giữ bốn levels và skip topology |
| Channels ở các levels | 44/88/88/176 | 69/138/138/276 | Multipliers vẫn (1,2,2,4) |
| Condition dimension | 160 | 312 | Tăng khả năng conditioning và đạt budget 15M |
| Residual blocks/level | 1 | 1 | Decoder vẫn hai blocks/level |
| Attention resolutions | 8 + middle 4 | 8 + middle 4 | Giữ attention math, hỗ trợ forward-mode JVP |
| Fourier dimension / classes / image | 64 / 10 / 32×32 RGB | Giữ nguyên | Cùng dataset và conditioning |
| Physical batch | 512 | 512 | Dùng H200; GPU smoke kiểm tra đúng physical batch trước run |
| Accumulation | 1 | 1 | Effective image batch vẫn **512** |
| Optimizer updates | 100.000 | 100.000 | Cùng budget **51,2M ảnh** |
| Warmup updates | 5.000 | 5.000 | Cùng warmup **2,56M ảnh** |
| LR → minimum | 1e-4 → 1e-5 | Giữ nguyên | Không có quy tắc LR tuyến tính theo parameters |
| AdamW betas / weight decay | (0.9,0.99) / 0 | Giữ nguyên | Theo config thực thi trong notebook |
| Gradient clipping | 1.0 | **1.0** | Giữ điểm khởi đầu, theo dõi clip_fraction để tune |
| EMA decay | 0.9999/update | 0.9999/optimizer update | Không update EMA mỗi microbatch |
| Diagonal loss weight | 1.0 | Giữ nguyên | Cả hai MSE đều mean theo pixel và batch |
| K / interval | K~Uniform{1..5}, j~Uniform{0..K−1} | Giữ nguyên | Một K/j cho cả effective batch |
| Training precision | BF16 AMP, TF32 | Giữ nguyên trên GPU BF16 | CPU FP32; GPU không BF16 dùng FP16 forward + FP32 JVP |
| Input layout / preload | channels-last / toàn bộ CIFAR trên GPU | Giữ nguyên trên 1 GPU | CIFAR float32 ~0,57 GiB, tránh CPU data bottleneck |
| Horizontal flip | Một coin-flip cho cả batch | Coin-flip riêng từng ảnh | Tránh tương quan augmentation giữa các ảnh |
| FID feature extractor | torchvision ImageNet Inception | pytorch-fid 0.3.0 FID Inception pool3 | Dùng protocol chuẩn, **không so trực tiếp số FID cũ** |
| FID real/generated | 10k train / 10k generated | 50k unaugmented train / 10k generated | Giảm nhiễu real stats, vẫn ghi rõ FID10k |
| FID batch | 512 | 128 | Batch chỉ ảnh hưởng memory/throughput của evaluation |

Actual `total_steps` trong notebook là 100.000; comment nhắc 300.000 không phải
giá trị được chạy. Mọi "step" của profile mới là một optimizer update với một
physical batch. Forward có gradient gộp edge+diagonal thành 1024 trace inputs
khi image batch=512; effective image batch vẫn là 512.
JVP không xây graph cho parameter gradient, tangent diagonal và kết quả được
detach đúng semi-gradient. Notebook chạy GPU smoke test bằng đúng batch512;
local CPU test không xác nhận CUDA forward AD, tốc độ hoặc memory H200.

## Scale khi thay effective batch

Với `B = micro_batch × accumulation × world_size`:

```text
updates = ceil(51_200_000 / B)
warmup_updates = ceil(2_560_000 / B)
ema_decay = 0.9999 ** (B / 512)
```

EMA giữ cùng độ dài nhớ theo số ảnh: khoảng 5,12M ảnh cho decay 1/e, khoảng
3,55M ảnh cho half-life. `scale_training_budget()` áp dụng công thức trên;
notebook gọi helper trước khi ghi runtime config. Batch64×accum8 giữ nguyên
100k updates/5k warmup/EMA .9999. Batch64×accum4 thành 200k updates/10k warmup
và EMA sqrt(.9999). Việc làm tròn có thể tăng image budget dưới một batch.
CLI `--batch-size`/`--steps` là override để smoke/debug; không tự scale image
budget. Khi train thật bằng CLI, gọi helper trong config trước khi bắt đầu.

LR, clipping, lambda_D và max_nfe không tự tăng theo số parameters. Clipping
1.0 là giả thiết thực nghiệm của bản tham chiếu: đọc norm trước/sau clip,
clip_fraction, actual update/weight và EMA validation trước khi đổi. Profile
mới giữ budget dữ liệu để so sánh; tăng parameters không đảm bảo chất lượng
hoặc cần đúng một hệ số bước học nào đó. Dataset vẫn CIFAR-10 32×32.

## Monitoring, resume và inference

- Log mỗi 50 updates: mean loss/gradient/clip stats, loss và số quan sát theo K,
  LR thực tế, images_seen, image/s, ETA và CUDA memory. Heatmap tách encoder /
  decoder levels; update/weight đo optimizer update thật, không dùng LR×gradient.
- Validation mỗi 1000: 2560 test images ở defaults, cùng noise/quantiles qua các
  lần đánh giá. Mỗi K có một dòng loss riêng; tổng loss là mean đều theo K.
  Raw và EMA validation đều được ghi để phân biệt EMA lag với loss plateau;
  EMA .9999 có thể làm đường validation phản ứng chậm ở đầu run.
  Năm batches đi qua mọi interval cho K≤5. Nếu đổi physical batch, số ảnh cố
  định thay đổi theo `VALIDATION_BATCHES × BATCH_SIZE`; dùng RUN_NAME mới.
- Samples EMA với cùng labels/noise, 1..5 NFE; TensorBoard/PNG/CSV/report plateau
  là bằng chứng chẩn đoán, không phải chứng minh hội tụ hoặc FID.
- Resume phục hồi raw model, optimizer/scheduler/scaler, EMA/update count,
  RNG và GPU resident batcher permutation/cursor. CPU DataLoader shuffle mới.
  Checkpoint ở ranh giới optimizer update; không checkpoint giữa microbatches.
- Modal lưu checkpoint cục bộ mỗi 1000 updates và backup full resume checkpoint
  lên `training-checkpoints/<RUN_NAME>/` trong Hugging Face repo mỗi 10000 updates.
  Điểm dừng pilot và final luôn được upload. Upload lỗi retry ba lần, ghi
  `hub_checkpoints.jsonl` và không làm dừng training; checkpoint Volume vẫn còn.
- Final FID dùng EMA và ghi protocol, sample counts, weights SHA256, seed, batch,
  từng NFE; cache real stats trên Volume. Fixed noise/labels dùng lại giữa NFE.
  FID10k có sampling variance và không tương đương FID50k. Không có metric
  chất lượng thực tế nào được đo trong workspace này.
- HF inference bundle mặc định chứa final EMA `model.safetensors`; full optimizer
  checkpoint được upload riêng khỏi inference bundle. `sample.py --nfe 1..5` dùng đúng reverse equal grid;
  `best_validation.safetensors` là EMA snapshot được chọn theo validation loss.

## Chạy trên Modal

Upload `notebooks/multitrace_imf_modal.ipynb`, dùng RUN_NAME mới.
Nếu chuyển từ notebook DiT trong cùng kernel, restart kernel trước để xóa
Python module cache. Clone guard báo lỗi khi module IMF đang đến từ source khác.

Mount `/mnt/imf-training` có thể là symlink tới `/__modal/volumes`; guard kiểm tra
mount path mà không yêu cầu resolved path còn trong `/mnt`.

Default `STOP_AFTER_UPDATES=1000` chạy pilot có checkpoint với **full 100k LR
schedule**, không nén warmup. Đọc throughput/ETA sau pilot; để train hết, đổi
`STOP_AFTER_UPDATES=None`, chạy settings + pipeline lần nữa. Sau restart kernel,
Run all với cùng RUN_NAME/config; notebook tự pin source commit cũ. Pilot chưa
được xuất thành final model, nhưng checkpoint resume pilot được backup lên Hub.
Không suy ra 100k updates có đủ $30
credit từ tên GPU; FID và CPU/RAM cũng tốn thời gian/chi phí.

Khi đã hoàn tất train, chạy pipeline lại để retry FID/upload mà không train lại.
`HF_TOKEN` đặt trong Modal Secret; Git clone dùng `codex/modal-training`, không
cần merge vào main. GPU-resident profile hỗ trợ một GPU/process.

## Kiểm chứng implementation

**39 unittest tests qua** trong môi trường local dưới đây.

- Đếm bằng PyTorch: reference5.946.579, default15M=14.992.182 và
  retained22M=22.002.655 trainable parameters.
- Model15M chạy hai bước loss/backward/AdamW và sampling trên CPU; loss và
  mọi parameter gradients hữu hạn. Train FakeData với micro1×accum2 đến step2,
  resume đến step4: optimizer/scheduler/EMA update count4, strict EMA reload,
  JSONL đúng steps1..4 và validation0/2/4, không missing gradient tensors.
- PNG dashboard/heatmap/runtime/loss theoK và samples1..5 NFE được kiểm tra.
  Đây là ảnh/log dữ liệu giả, không dùng để kết luận hội tụ hoặc chất lượng.
- Tests kiểm tra analytic nonzero-interval JVP/semi-gradient, finite difference
  với channels-last U-Net, accumulation tương đương full batch, EMA sau resume,
  GPU resident batcher state (kiểm tra tensor path trên CPU), safetensors export
  và inference bundle chạy bằng Python process riêng.
- FID orchestration dùng mocked features/scores để kiểm tra EMA selection,
  fixed noise/labels giữaNFE, cache real stats và retry kết quả từngNFE. Không
  tải Inception weights hoặc đo FID thật trong những tests này.
- Notebook schema/cell syntax, pilot không upload partial model, completed-run
  eval→export→upload và Modal volume symlink guard được kiểm tra. HF API trong
  tests được mock; chưa upload model thật hoặc chạy trên tài khoản Modal/GPU.

Môi trường local: PyTorch2.2.2, Accelerate1.15.0, pytorch-fid0.3.0 trong venv
tạm `/private/tmp/imf-verify-env`. CUDA BF16/JVP, microbatch512 memory, tốc độ
H200 và final model quality phải được đo ở run Modal thật.

Nguồn API: [Accelerate gradient synchronization](https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization),
[pytorch-fid](https://github.com/mseitzer/pytorch-fid),
[Modal Notebooks](https://modal.com/docs/guide/notebooks).
