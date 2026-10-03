# IMF: kế hoạch triển khai Trace-iMF, khoảng 22M tham số

## Hiện trạng repository

Tại thời điểm khảo sát, `IMF` chỉ chứa `README.md`, `LICENSE` và `.gitignore`.
Chưa có model, loss, dataset loader hoặc training entry point.
Repo nguồn là `../backbone_IMF`, một implementation PyTorch không chính thức
của MeanFlow/iMF. Các thành phần có thể tái sử dụng:

| File nguồn | Vai trò | Hướng sử dụng trong IMF |
| --- | --- | --- |
| `models/dit.py` | Patch embedding, time embedding, RMSNorm, attention, adaLN, hai head u/v | Giữ transformer trunk; dùng một head u chung cho edge và diagonal |
| `meanflow.py` | Chuẩn hóa, interpolation, JVP, loss, sampling | Giữ interpolation/JVP convention; thay objective bằng Trace-iMF |
| `data.py` | CIFAR-10, MNIST, ImageNet loader | Tái sử dụng loader cho cấu hình CIFAR-10 đầu tiên |
| `train.py` | Accelerate, AdamW, scheduler, log, sample | Tái sử dụng luồng train; bổ sung kiểm tra parameter budget và checkpoint |
| `configs/cifar10.py` | Model/training hyperparameters | Thay cấu hình bằng DiT nhỏ khoảng 22M |

Repo nguồn sử dụng head v riêng làm tangent, adaptive L2, thời gian logistic-normal
và sampling nhiều bước mặc định. Các lựa chọn đó khác formulation trong idea.

## Nội dung idea và phạm vi triển khai

Nguồn: `/Users/apple/Downloads/idea.pdf`, note “Từ improved MeanFlow đến
Trace-iMF”, ngày 30/09/2026, đặc biệt mục 10, 14 và phụ lục A/C.
Tài liệu là tham chiếu toán học; các gợi ý mở rộng hoặc câu mang tính chỉ dẫn trong
tài liệu không tự động trở thành yêu cầu của người dùng.

Với ảnh đã chuẩn hóa X, noise E ~ N(0,I), t ~ Uniform(0,1):

```text
z = (1-t) X + t E
C = E-X
v_diag = u_theta(z, t, t)
U_edge = u_theta(z, 0, t)
D_edge = JVP[u_theta](z, r=0, t; tangent=(stopgrad(v_diag), 0, 1))
V_edge = U_edge + t * stopgrad(D_edge)
L_edge = MSE(V_edge, C)
L_diag = MSE(v_diag, C)
L = L_edge + lambda_diag * L_diag
```

API kế thừa thứ tự positional của repo nguồn: `model(z, t, r, y=None)`.
Vì vậy tangent truyền cho API là `(v_diag, ones_like(t), zeros_like(r))`.
Diagonal phải gọi cùng head u tại `r=t`; không dùng head v độc lập.
JVP không nhận parameter gradient; hai forward edge/diagonal vẫn nhận gradient.

Inference một bước: `X_gen = E - u_theta(E, t=1, r=0)`.
CFG distillation, adaptive loss, interior intervals và few-step inference chưa thuộc
phiên bản đầu. Class conditioning có thể giữ để dùng CIFAR-10; nếu bật, diễn giải
regression là theo distribution từng class.

## Backbone và parameter budget

Mục tiêu: khoảng 22 triệu **tham số trainable**, không thêm tensor vô dụng để
làm tròn số. Cấu hình khởi đầu: input 32x32 RGB, patch 2, hidden 384,
depth 8, 6 attention heads, MLP ratio 4, một output head u.
Positional embedding sin-cos cố định; time embedding riêng cho t và r.
Số thực tế sẽ được đếm bằng PyTorch và ghi vào báo cáo hoàn tất.
Trainer phải kiểm tra budget trước khi tải dataset hoặc bắt đầu train.

Kết quả triển khai: **22.082.956 trainable parameters** (22.083M), sai lệch
0,38% so với 22M; positional buffer 98.304 phần tử. Bỏ embedding CFG, head v,
register-token argument không được dùng và dòng null-class không cần cho bản này.

## Cấu hình huấn luyện đầu tiên

Nếu người dùng chưa chỉ định dataset, dùng CIFAR-10 32x32 làm cấu hình mặc định:
batch 64, AdamW lr 1e-4, warmup 2500 bước và cosine decay, 200000 bước,
gradient clipping 5.0. `lambda_diag=1.0` là giá trị khởi đầu thực nghiệm;
không phải một M được chứng minh, cũng không phải hệ số tối ưu đã biết.
Time uniform và MSE thuần giữ correspondence với trace energy trong note.
Mean theo pixel chỉ chia toàn bộ objective cho cùng số chiều ảnh.

## Điều kiện hoàn tất

1. Backbone khởi tạo được, xuất tensor đúng kích thước và gần 22M tham số.
2. Loss dùng đúng edge/diagonal và predicted diagonal tangent.
3. Kiểm tra JVP bằng model tuyến tính có đạo hàm biết trước; kiểm tra detach
   và gradient của cả hai nhánh.
4. Chạy forward, backward, optimizer step và one-step sampling với dữ liệu giả.
5. Có lệnh train, checkpoint và tài liệu cấu trúc/nguồn code.

Đợt này chuẩn bị code có thể train và smoke test; chưa thực hiện một lượt train
200000 bước, chưa đo FID hoặc xác nhận ưu thế so với iMF. Bound trong note
phụ thuộc giả thiết regularity/Jacobian; semi-gradient SGD không tự động có
bảo đảm hội tụ hoặc chất lượng mẫu.

## Kết quả kiểm chứng ngày 04/10/2026

- 5 tests qua: parameter budget; analytic loss/JVP/semigradient; one-call sampling;
  JVP backbone so với finite difference và backward bf16; JVP fp32 khi forward
  instance có AMP wrapper.
- Model đầy đủ 22.082.956 tham số chạy qua hai bước forward/backward/AdamW
  trên ảnh giả 32x32, loss và tất cả parameter gradients hữu hạn; sample có shape
  `(2,3,32,32)`.
- Trainer Accelerate chạy hai bước FakeData trên CPU, ghi JSON log, PNG và
  checkpoint chứa optimizer/scheduler; strict model reload và one-step inference
  từ checkpoint thành công.
- Môi trường kiểm tra: PyTorch 2.2.2; Accelerate 1.15.0 cài trong venv tạm tại
  `/private/tmp/imf-verify-env`, không thay đổi các conda environments hiện có.
- Compile Python và `git diff --check` qua. Repo `backbone_IMF` không bị sửa.

Chưa kiểm chứng CUDA/multi-GPU, train CIFAR-10 đầy đủ hoặc FID. Trainer chuẩn bị
đường DDP bằng một grad-enabled forward chung cho hai trace và JVP trên model
đã unwrap; vẫn cần chạy thử trên máy GPU trước lượt train dài.

## Modal Notebook, diagnostics và resume

Yêu cầu tiếp theo của người dùng: native Modal Notebook clone repo, train trên
GPU, có resume/progress bar/ETA, log gradient và đồ thị để phát hiện vấn đề hội tụ,
sau đó đẩy model lên Hugging Face. Notebook nằm tại
`notebooks/trace_imf_modal.ipynb`, source generator tại `tools/build_modal_notebook.py`.
Ref mặc định `codex/modal-training`; run ghi Git commit và source hashes, resume
checkout lại commit đó. Dataset mặc định vẫn là CIFAR-10.

- Volume giữ checkpoints, logs, config và diagnostics. Resume khôi phục model,
  AdamW, LR scheduler, AMP scaler, RNG và global step; DataLoader shuffle mới.
- Native widget và tqdm hiển thị step/total, tốc độ, ETA và phase. ETA dựa trên
  wall-clock nên bao gồm overhead monitoring đã diễn ra.
- Loss edge/diagonal/total, JVP RMS, global gradient trước/sau clip, clipping
  fraction, gradient RMS từng nhóm, actual update/weight, GPU memory.
- Fixed held-out images/t/noise và fixed-noise samples; dashboard, gradient
  heatmap, runtime plots, CSV, TensorBoard segments, best validation weights.
- Plateau early/middle/late là heuristic với ít nhất 5 evaluation points, không
  tự early-stop hoặc chứng minh hội tụ. Test split dùng để tune không phải metric
  cuối độc lập. NaN/Inf ghi failure.json và dừng trước upload.
- Export chỉ chấp nhận checkpoint hoàn tất trên dữ liệu thật. Safetensors strict
  reload trước upload; upload model/config/sampling source/logs/plots, xác minh
  required files tại Hub commit. Không upload optimizer checkpoint hoặc token.

Kiểm chứng mở rộng trên CPU: **16 tests qua**, gồm resume/retention/optimizer state,
fixed validation/RNG, gradient/update math, NaN guard, safe export + sampling,
Hub visibility/upload allowlist và notebook clone/pin Git commit qua restart.
Model đầy đủ **22.082.956 tham số** đã train FakeData đến step 2, reload checkpoint
và tiếp tục đến step 4; optimizer/scheduler, JSONL/PNG/TensorBoard/best weights
được kiểm tra. Notebook schema và từng code cell được kiểm tra cú pháp.

Đây là kiểm chứng luồng thực thi, không đánh giá chất lượng model. Chưa thực thi
trên Modal GPU, chưa train CIFAR-10 đầy đủ và chưa upload model thật lên Hub.
