# IMF - Multi-Trace U-Net 15M trên Modal

Profile mới tham chiếu notebook **U-Net Multi-Trace** của người dùng:
[SCALING.md](SCALING.md) có bảng so sánh config và công thức scale theo image budget.
Backbone mặc định **14.992.182 tham số**, base channels69, cond312; K=1..5,
BF16, AdamW betas(0.9,0.99), clipping1.0, effective batch512
(micro512 × accumulation1 trên một H200),
100k optimizer updates, warmup5k, EMA .9999. LR1e-4→1e-5 giữ theo bản tham chiếu.

Notebook mới: [notebooks/multitrace_imf_modal.ipynb](notebooks/multitrace_imf_modal.ipynb).
Clone nhánh `codex/modal-training`; cần RUN_NAME mới, Volume và HF_TOKEN Secret.
Mặc định pilot1000 updates có resume/ETA/gradient/raw+EMA validation; đổi
`STOP_AFTER_UPDATES=None` để train hết schedule rồi chạy FID1..5 NFE và upload
final EMA model lên Hugging Face. Full resume checkpoint được backup lên cùng
Hub repo tại pilot/final và mỗi 10000 updates. FID dùng pytorch-fid, khác protocol FID trong
notebook nguồn nên không so trực tiếp. Local tests không xác nhận CUDA memory,
thời gian hoàn tất hay chất lượng model trên Modal.

```bash
python train.py --config configs/multitrace_cifar10_15m.py --check-model
python train.py --config configs/multitrace_cifar10_15m.py --smoke-test
python tools/build_multitrace_notebook.py
```

Profile U-Net 22M cũ vẫn nằm ở `configs/multitrace_cifar10_22m.py` để đối chiếu;
checkpoint 22M không tương thích với profile 15M. Phần dưới mô tả baseline
**DiT hai trace** cũ; config/workflow U-Net nằm trong SCALING.md.

Triển khai PyTorch theo idea Trace-iMF của người dùng, kế thừa DiT và training
flow từ `../backbone_IMF`. Đọc [REPO_PLAN.md](REPO_PLAN.md) để xem khảo sát repo,
công thức loss, khác biệt với implementation nguồn và phạm vi lý thuyết.

Backbone mặc định có **22.082.956 tham số trainable** (22.083M, lệch 0,38% so với
22M), hidden 384, depth 8, 6 attention heads, MLP ratio 4, patch 2 cho ảnh RGB
32x32. Positional embedding sin-cos là buffer cố định gồm 98.304 phần tử, không
được tính vào parameter budget. Head `u` dùng chung trên edge và diagonal.
Không có head `v` riêng hoặc embedding CFG.

## Cài đặt và kiểm tra

Python 3.10+, môi trường PyTorch/torchvision tương thích. Cấu hình CUDA dùng
bf16; CPU tự chuyển sang fp32. Chọn PyTorch wheel phù hợp với CUDA của máy train.

```bash
python -m pip install -r requirements.txt
python train.py --config configs/cifar10_22m.py --check-model
python -m unittest discover -s tests -v
python train.py --config configs/cifar10_22m.py --smoke-test
```

`--check-model` đếm tham số trước khi tải dữ liệu. `--smoke-test` chạy hai bước
forward/backward/AdamW và one-step sampling bằng model 22M trên CPU với ảnh giả,
không tải dataset. Budget cho phép sai lệch 1%; cấu hình vượt budget sẽ báo lỗi.

## Train

CIFAR-10 là cấu hình ban đầu khi chưa có dataset được chỉ định. Dataset tự tải
về `data/cifar10`. Batch size mặc định 64 **mỗi process**; cả hai trace dùng cùng
batch ảnh, forward có gradient gộp thành 128 trace inputs mỗi process.

```bash
python train.py --config configs/cifar10_22m.py

# Hai GPU; global image batch = 128
accelerate launch --num_processes 2 train.py --config configs/cifar10_22m.py

# Kiểm tra toàn bộ trainer bằng FakeData, không tải CIFAR-10
python train.py --fake-data --steps 2 --batch-size 2 --num-workers 0
```

Trainer lưu `logs/<run>/config.json`, `train.jsonl`, `images/` và `ckpts/`.
Logs chứa loss edge/diagonal, JVP RMS, gradient norm và learning rate, trung bình
giữa các process. Checkpoint mỗi 5000 bước và cuối run chứa model, optimizer,
scheduler, config, step và RNG mỗi process. Ghi checkpoint bằng file tạm rồi rename.
Resume dùng `--resume logs/<run>/ckpts/step_0005000.pt` với cùng config và world
size. Model, optimizer, scheduler và RNG được khôi phục; DataLoader bắt đầu shuffle
mới, nên thứ tự dữ liệu không tái hiện chính xác lượt chạy liên tục.
Các ảnh lưu là one-step samples; ảnh từ smoke/FakeData không đánh giá chất lượng.

## Modal Notebook và Hugging Face

Notebook: [notebooks/trace_imf_modal.ipynb](notebooks/trace_imf_modal.ipynb).
Upload vào [Modal Notebooks](https://modal.com/notebooks), chọn 1 GPU L4 hoặc
A100, attach Volume `imf-training` tại `/mnt/imf-training` và attach Secret có
`HF_TOKEN` quyền write. Chỉnh cell cấu hình rồi Run all.

Notebook clone `https://github.com/namdeptraivcd/IMF.git`, mặc định nhánh
`codex/modal-training`. Có thể đặt `REPO_REF` là Git commit SHA. Nó kiểm tra
GPU và Hub trước khi train, chạy GPU smoke test 3 bước, train CIFAR-10 22M,
resume tự động từ checkpoint trên Volume, rồi upload model khi đủ số bước.
Mặc định 200000 bước, batch 64, checkpoint mỗi 1000 bước và giữ 3 checkpoint gần
nhất. Đặt `TRAIN_STEPS=20` để thử luồng ngắn trước lượt train dài có tính phí GPU.

`HF_REPO_ID=None` dùng `<tài-khoản-token>/trace-imf-cifar10-22m`; hoặc điền repo
riêng. Profile Multi-Trace dùng `HF_PRIVATE=None`: giữ nguyên visibility của repo
đã tồn tại và tạo repo mới ở chế độ private. Đặt `True` hoặc `False` chỉ khi muốn
kiểm tra visibility khớp chính xác. Không ghi token vào config hay artifact.
Trong khi train, notebook backup full resume checkpoint tại pilot/final và mỗi
10000 updates vào `training-checkpoints/<RUN_NAME>/` trên Hub; checkpoint cục bộ
vẫn lưu mỗi 1000 updates và giữ ba bản gần nhất. Mỗi remote checkpoint chứa raw
model, EMA, optimizer, scheduler, scaler, RNG và data cursor. Upload retry ba lần;
lỗi Hub được ghi vào `hub_checkpoints.jsonl` nhưng không dừng GPU training.
Khi train xong, notebook export `model.safetensors`, architecture config, source
sampling, model card, log, ảnh, đồ thị, TensorBoard events và best validation
weights khi có.
Upload thất bại có thể chạy lại cell pipeline để thử lại mà không train lại.

Hướng dẫn nền tảng: [Modal Notebook setup](https://modal.com/docs/guide/notebooks),
[Volume persistence](https://modal.com/docs/guide/volumes),
[Hub folder uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).
Chưa chạy notebook trên tài khoản Modal/GPU hoặc upload một model thật trong
workspace này; token/account được người dùng cung cấp khi mở notebook.

Tạo lại notebook sau khi sửa các cell trong generator:

```bash
python tools/build_modal_notebook.py
```

Code cho notebook nằm trên nhánh [`codex/modal-training`](https://github.com/namdeptraivcd/IMF/tree/codex/modal-training).
Nếu sửa code riêng, commit/push ref đó trước khi chạy notebook clone. Khi resume,
notebook checkout Git commit lưu trong run trước và kiểm tra source manifest/config. `AUTO_RESUME`
chọn checkpoint mới nhất; `RESUME_CHECKPOINT` có thể chỉ định checkpoint của run.

## Theo dõi hội tụ và thời gian chạy

Notebook có progress widget trực tiếp: step/total, step/s, ETA theo giờ, phase và
stdout của trainer. `tqdm` cũng hiển thị tiến độ/ETA/loss/gradient/LR. ETA dùng
wall-clock thực tế nên tính cả evaluation, plotting và checkpoint trước đó;
các ước lượng đầu run có thể dao động. Default log mỗi 50 bước, plot mỗi 200,
validation/sample/checkpoint mỗi 1000. Các interval nằm trong cell settings.

- `train.jsonl`: loss total/edge/diagonal, JVP RMS, LR thực sự dùng ở bước đó,
  norm gradient trước/sau clip, clipping fraction, throughput/ETA, GPU memory.
- Gradient snapshot từng nhóm embedding, từng block và head: norm, RMS,
  gradient/weight, **actual optimizer update/weight**, missing/zero gradient
  tensors. Snapshot đo một bước tại thời điểm log; loss/clip stats là mean trong
  cửa sổ. Ratios của weight đang bằng 0 được ghi null để tránh spike giả do adaLN-zero.
- `validation.jsonl`: cùng test images, thời gian và noise qua các lần đánh giá,
  với batch/số batch cố định. Được dùng để tune nên cần protocol evaluation riêng
  khi báo cáo metric cuối.
- `images/` và `samples.jsonl`: fixed-noise/class samples để so thay đổi theo step.
- `diagnostics/`: dashboard 8 panels, gradient heatmap, runtime/ETA/memory plots,
  CSV và convergence report. Ít nhất 5 validation points mới xét plateau candidate;
  <1% cải thiện trong cửa sổ được gắn early/middle/late theo budget. Đây là gợi ý,
  không tự early-stop và không chứng minh hội tụ/chất lượng. MSE có noise floor.
- `tensorboard/`: scalar/image events; mỗi resume mở segment mới.
- `best_validation.safetensors` + `.json`: weights và step/metrics của best
  available validation snapshot, có thể so với final weights.
- `failure.json`: step/loss/grad norm và parameter có gradient NaN/Inf khi trainer dừng.

Resume loại metrics sau checkpoint trước khi nối log, giữ schedule/optimizer/scaler
và global step. DataLoader shuffle mới, không replay bit-for-bit. Có thể bật
diagnostics cho CLI thông thường bằng `monitoring=dict(enabled=True, tensorboard=True,
validation_every=1000, validation_batches=4, plot_every=200, save_best=True)` trong
config và cài `requirements-notebook.txt`.

Để tạm dừng có checkpoint theo global step mà vẫn giữ LR schedule đầy đủ:

```bash
python train.py --config configs/cifar10_22m.py --run-dir logs/my_run --stop-after 1000
python train.py --config configs/cifar10_22m.py --resume logs/my_run/ckpts/step_0001000.pt
```

## Objective

```text
t ~ Uniform(0,1), E ~ N(0,I), X = 2*image - 1
z = (1-t)*X + t*E, C = E-X
v_diag = u(z,t,t)
D_edge = partial_t u(z,0,t) + partial_z u(z,0,t) * stopgrad(v_diag)
V_edge = u(z,0,t) + t*stopgrad(D_edge)
loss = MSE(V_edge,C) + lambda_diag*MSE(v_diag,C)
```

`lambda_diag=1.0` là điểm bắt đầu thực nghiệm. Trong note, gợi ý `M²/3` chỉ có
ý nghĩa khi biết bound M của spatial Jacobian. Code không ước lượng M và không
tuyên bố đảm bảo FID/hội tụ. MSE dùng mean theo pixel, tương đương chuẩn L2
trong note sau khi chia cả hai nhánh cho cùng số chiều ảnh. JVP chạy fp32 qua
native attention, dùng `torch.func.jvp`; forward tối ưu dùng SDPA khi CUDA/dtype
hỗ trợ. JVP outcome bị detach; edge và diagonal đều có parameter gradient.

API model giữ thứ tự `(z,t,r,y)` của repo nguồn. Khi đọc ký hiệu toán học
`u(z,r,t)`, cần đổi đúng thứ tự tham số khi gọi Python.

## Sinh mẫu từ checkpoint

```python
import torch
from torchvision.utils import save_image
from models.dit import TraceDiT
from trace_imf import TraceIMF

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("logs/<run>/ckpts/step_0200000.pt", map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
model = TraceDiT(**cfg["model"]).to(device)
model.load_state_dict(checkpoint["model"])
objective = TraceIMF(channels=cfg["model"]["in_channels"],
                     image_size=cfg["model"]["input_size"],
                     num_classes=cfg["model"]["num_classes"], **cfg["trace_imf"])
labels = torch.arange(10, device=device)
samples = objective.sample(model, labels=labels, device=device)
save_image(samples, "samples.png", nrow=10)
```

Chỉ một evaluation: `X_gen = E-u(E,t=1,r=0)`, sau đó clamp và đổi về [0,1].
Chưa train interior intervals, CFG, adaptive L2, latent VAE hoặc few-step.

## Cấu trúc

```text
models/dit.py             # TraceDiT: transformer backbone, một head u
trace_imf.py              # loss hai trace + one-step sampling
data.py                   # loader kế thừa nguồn; cấu hình đầu dùng CIFAR-10
train.py                  # parameter guard, smoke test, Accelerate trainer
hub.py                    # export safetensors + Hub upload và verify commit
monitoring.py             # validation, gradients, charts, TensorBoard, live ETA
sample.py                 # sampling từ inference bundle, không dùng pickle
configs/cifar10_22m.py    # cấu hình model và train
configs/multitrace_cifar10_15m.py  # U-Net Multi-Trace mặc định cho H200
configs/multitrace_cifar10_22m.py  # profile U-Net 22M cũ để đối chiếu
tests/test_trace_imf.py   # analytic JVP/semigradient, sampling, budget, backward
tests/test_notebook_pipeline.py  # resume, export guards, Hub visibility/upload
tests/test_monitoring.py  # gradient/update math, fixed validation, rollback, clone pin
notebooks/trace_imf_modal.ipynb  # clone repo + Modal GPU notebook
tools/build_modal_notebook.py   # tạo notebook
third_party/              # nguồn code và MIT notice
REPO_PLAN.md              # khảo sát và thiết kế trước triển khai
```

Nguồn code và license được ghi tại [third_party/README.md](third_party/README.md).
