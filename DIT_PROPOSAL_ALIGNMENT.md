# DiT 22M, workflow notebook và mức độ khớp proposal

Ngày đối chiếu: 04/10/2026. Đây là bản thiết kế để xem xét, **chưa áp dụng một
notebook DiT mới**. Phần scale U-Net đã được tạm dừng để đối chiếu tài liệu này;
sau đó người dùng chọn tiếp tục **U-Net22M Multi-Trace**, mô tả ở SCALING.md.
Các bảng DiT bên dưới là phương án tham khảo, không phải config của run U-Net.

## 1. Kết luận

**DiT 22M + hyperparameters của repo + cách vận hành của notebook là một hướng
phù hợp với proposal, nếu loss vẫn train đúng hai trace `r=0` và `r=t`, cùng
một head `u`, và đánh giá chính ở 1 NFE.**

Nếu giữ nguyên cả cách lấy interval `K=1..5` và sampler 1–5 NFE của notebook,
thì đó là **DiT Multi-Trace**, một mở rộng có liên hệ với proposal, không phải
objective one-step chính trong mục 10/14 của proposal. Thay U-Net bằng DiT
không tự làm hai objective này trở thành giống nhau.

Nếu "config như repo" có nghĩa giữ nguyên **toàn bộ** `backbone_IMF` kể cả
`MeanFlow.loss`, hai head `u/v`, logistic-normal time, CFG và adaptive L2,
thì cũng **không khớp** formulation Trace-iMF của proposal.

## 2. Ba nguồn đang được đối chiếu

| Nguồn | Nội dung thực tế | Vai trò |
|---|---|---|
| [idea.pdf](/Users/apple/Downloads/idea.pdf) | Trace-iMF chuyên one-step; mục 7–10, 14, 15.3 và phụ lục A/C | Formulation và phạm vi của bound |
| [backbone config](/Users/apple/tower-challenge/backbone_IMF/configs/cifar10.py), [backbone loss](/Users/apple/tower-challenge/backbone_IMF/meanflow.py) | DiT512/depth12/heads8, hai output heads; iMF với adaptive loss và CFG | Nguồn kiến trúc/training hyperparameters, không phải objective cần chép nguyên |
| [IMF config hiện có](/Users/apple/tower-challenge/IMF/configs/cifar10_22m.py), [TraceDiT](/Users/apple/tower-challenge/IMF/models/dit.py) | DiT384/depth8/heads6, một head `u`, 22.082.956 trainable parameters | Backbone 22M đã có để triển khai proposal |
| [Untitled0.ipynb](/Users/apple/Downloads/Untitled0.ipynb) | U-Net5.946.579 parameters, Multi-Trace1..5, batch512, EMA, eager JVP | Tham chiếu cho quy trình train/eval và một objective mở rộng |

Mốc DiT trước khi làm U-Net là commit `4915892` trên `codex/modal-training`.
Notebook U-Net mới là artifact riêng; các config DiT dưới đây không mô tả nó.

## 3. Config DiT theo repo để kiểm tra proposal

Giữ **backbone DiT22M trong IMF**, còn các lựa chọn training dưới đây giữ theo
config CIFAR-10 của `backbone_IMF`. Proposal không quy định width, depth, batch,
LR, optimizer, clipping hay số training steps; đây là lựa chọn thực nghiệm.

| Thành phần | Giá trị | Nguồn / lý do |
|---|---|---|
| Backbone | `TraceDiT`, **22.082.956 trainable parameters** | IMF hiện có, lệch mục tiêu22M khoảng0,38% |
| Input | RGB32×32, CIFAR-10, 10 classes | Cùng dữ liệu nguồn và notebook |
| Patch size / tokens | 2 / 256 | `(32/2)²`; mỗi patch được đưa vào transformer |
| Hidden / depth / heads / MLP ratio | 384 / 8 / 6 / 4 | Cấu hình22M; không phải nguyên model512/12/8 của backbone repo |
| Output | Một head `u`; diagonal cũng gọi head này | Để `v_diag(z,t)=u(z,t,t)` đúng proposal |
| Position | Sin-cos cố định, buffer98.304 phần tử | Không tính buffer vào trainable budget |
| Conditioning | `t`, `r`, class `y`; không CFG | Conditional extension theo từng class; không thêm guidance target |
| Seed | 42 | Reproducibility setting, không được suy ra từ theorem |
| Effective image batch | **64** trên1 GPU | Giữ batch repo; edge/diag forward gộp thành128 trace inputs, không phải128 ảnh độc lập |
| Micro-batch / accumulation | 64 / 1 | Có thể đổi thành32×2 để giảm memory, giữ effective batch64 |
| Optimizer | AdamW, betas **(0.9,0.999)** | Repo gọi AdamW mặc định; notebook dùng(0.9,0.99), là một thay đổi khác |
| Weight decay | 0 | Repo |
| Peak LR / minimum LR | 1e-4 / 1e-5 | Repo; cosine minimum ratio0.1 |
| Updates / warmup | **200.000 / 2.500** | Repo; không phải giá trị mà proposal chứng minh tối ưu |
| Tổng ảnh / ảnh warmup | **12,8M / 160k** | Effective batch64 nhân số updates |
| Gradient clipping | **5.0** | Có trực tiếp trong `backbone_IMF/configs/cifar10.py`; không lấy từ proposal |
| Precision | BF16 forward; FP32 JVP, loss accumulation | DiT workflow hiện có; ưu tiên derivative check bằng FP32 |
| Time | `t ~ Uniform(0,1)` | Để MSE tương ứng trace energies tích phân đều theo thời gian |
| Left edge | `r=0` cho mọi training image | Proposal one-step |
| Diagonal | `r=t`, cùng `z,t,y` | Predicted marginal tangent từ shared head |
| Target | `C=E−X`, `X=2*image−1`, `E~N(0,I)` | Proposal interpolation / target |
| Loss | MSE thuần: `L0 + 1.0*LD` | Giữ quadratic surrogate; λD1 là điểm khởi đầu thực nghiệm |
| JVP tangent | Predicted diagonal, không dùng `C` làm tangent | `∂t u + (∇z u)*u_diag` |
| Stop-gradient | Detach JVP outcome; cả edge và diagonal vẫn nhận gradient | Proposal mục10.3/phụ lụcA |
| Main inference | **1 NFE**, `E−u(E,0,1)` | Proposal equation100 |

Class labels là một mở rộng: dùng cùng derivation cho distribution `p0(·|y)`.
Để diễn giải bound cho mixture nhiều classes cần lấy trung bình theo đúng class
prior và có các giả thiết regularity/Jacobian phù hợp cho từng class. Việc có
10-class embedding tự nó không chứng minh các giả thiết đó.

### EMA nếu muốn học cách vận hành của notebook

Repo DiT đã publish chưa có EMA; notebook tham chiếu dùng `.9999` mỗi update với
effective batch512. EMA có thể bổ sung vào workflow DiT mà không đổi hình thức
của objective. Tuy nhiên model EMA là bộ parameters khác model raw; phải đánh
giá loss/sample của chính EMA, không áp metric raw cho EMA.

Có hai cách chọn decay:

- Chép nguyên `.9999` ở batch64: hợp lệ về thực nghiệm, nhưng độ dài nhớ theo
  số ảnh ngắn hơn notebook **8 lần**.
- Giữ độ dài nhớ theo số ảnh: `decay=.9999**(64/512)` = **0.999987499453**.
  Đây là cách quy đổi rõ ràng nếu mục đích là giữ hành vi EMA theo image budget.

Trong bản thiết kế giữ batch repo, đề xuất ghi rõ lựa chọn thứ hai, update EMA
một lần sau mỗi successful optimizer update. EMA không phải điều kiện bắt buộc
của proposal và không tạo bảo đảm hội tụ.

## 4. Workflow giữ đúng objective của proposal

### Chuẩn bị và khởi động

1. Modal1 GPU; attach Volume `/mnt/imf-training`, attach `HF_TOKEN` quyền write.
2. Clone IMF ở ref đã publish; ghi/pin Git commit, source manifest và config.
3. Khởi tạo **TraceDiT22M**, kiểm tra parameter budget; tải CIFAR-10 train/test,
   chuẩn hóa đúng một lần. Train có random horizontal flip riêng từng ảnh.
4. Eager JVP smoke test: loss/grad finite, đúng shape, đúng tangent/order,
   optimizer step và sample. Chưa bật compile trước khi có proof tương thích.
5. Tạo optimizer/scheduler; tạo EMA nếu chọn dùng; resume đầy đủ state khi có
   checkpoint. Không resume U-Net checkpoint vào DiT hoặc đổi objective giữa run.

### Một optimizer update

Notation toán học là `u(z,r,t,y)`; **API Python DiT trong repo là `(z,t,r,y)`**.
Notebook gốc có thứ tự `(z,r,t,y)`. Phải đổi đúng thứ tự, không chép positional
calls nguyên xi. Hai time embeddings khác nhau nên việc đảo thứ tự không vô hại.

```text
sample image x, class y, E~N(0,I), t~Uniform(0,1)
X = 2*x - 1
z = (1-t)*X + t*E
C = E-X

diag = model(z, t, t, y)          # u(z,r=t,t,y)
edge = model(z, t, 0, y)         # u(z,r=0,t,y)
D = JVP(model, primals=(z,t,0),
               tangent=(stopgrad(diag),1,0))
V = edge + t*stopgrad(D)
L0 = mean((V-C)^2)
LD = mean((diag-C)^2)
L  = L0 + lambda_D*LD

backward(loss / accumulation) for each microbatch
measure parameter gradients; clip global gradient norm at5.0
AdamW.step(); LR scheduler.step(); EMA.update() if enabled
zero_grad(); increment optimizer update counter
```

Cùng model/head phải được dùng cho edge, diagonal và JVP. JVP có thể chạy riêng
trên model đã unwrap để tương thích Accelerate; không dùng EMA làm tangent khi
training raw model. Tangent theo `r` bằng0 vì đạo hàm giữ fixed start interval.

Mean theo pixel chia cả hai losses cho cùng `d=3*32*32=3072` so với squared
Euclidean norm của proposal. Tỉ lệ λD giữ nguyên; nếu muốn thay số vào bound dạng
norm tổng phải nhân residual energies dạng pixel-mean với3072. BF16 là xấp xỉ
số học; proof công thức/JVP nên kiểm tra bằng FP32.

### Logging, checkpoint và evaluation

| Việc | Interval đề xuất | Lý do |
|---|---:|---|
| Train log / gradient snapshot | 50 updates | Mean loss/norm/clip trong cửa sổ; snapshot actual update/weight tại bước log |
| Dashboard / heatmap | 500 updates | Theo dõi ổn định mà hạn chế I/O overhead |
| Fixed raw/EMA validation | 1.000 updates | 512 test images: batch64×8; cùng image/time/noise qua các lần eval |
| Fixed-noise/class samples | 1.000 updates | Đánh giá chính ở1 NFE; raw/EMA ghi nhãn riêng |
| Checkpoint | 1.000 updates; giữ3 checkpoint cuối | Có thể resume sau interruption, không chờ đến cuối run |
| Pilot | Dừng sau1.000 updates, giữ full200k LR schedule | Đo memory/throughput/ETA trước run dài; pilot nằm trong warmup, chưa đủ để kết luận hội tụ |
| Final quality evaluation | Sau khi train hoàn tất | 1 NFE; cùng protocol và số ảnh khi so giữa variants |
| HF upload | Sau evaluation hoàn tất | Export đúng raw/EMA đã đánh giá; kèm config/source/metrics/sample/diagnostics |

Logs cần có loss total/edge/diagonal, JVP RMS, gradient trước/sau clip,
clip_fraction, gradient RMS từng DiT block, actual update/weight, zero/missing
gradient tensors, LR dùng thực tế, image/s, peak GPU memory và ETA. Zero-init
adaLN/head có thể làm trunk gradients bằng0 lúc đầu; không tự coi là lỗi.
Raw+EMA validation giúp tránh nhầm EMA lag thành hội tụ sớm.

Resume lưu model raw, EMA nếu có, optimizer, scheduler, scaler, RNG và global
update. GPU-resident loader cần lưu permutation/cursor; DataLoader thông thường
shuffle mới và không đảm bảo replay bit-for-bit. Pin source/config để tránh
vừa resume vừa đổi thuật toán. Upload failure chỉ retry export/upload, không
train lại một run đã hoàn tất.

Chỉ mượn GPU preload cho CIFAR nếu muốn giảm CPU bottleneck; `channels_last` của
U-Net không tự tăng tốc DiT token attention. Compute settings thay throughput
và memory, không thay identity. ETA phải đo bằng run thật; chưa có benchmark DiT
22M trên H200 ở workspace này.

### FID và giới hạn kết luận

Notebook gốc dùng torchvision **ImageNet Inception weights**, thay `fc` bằng
Identity rồi resize và map pixels sang[-1,1]. Feature pipeline này khác FID
Inception của `pytorch-fid`. Số FID cũ chỉ dùng để so nội bộ cùng protocol;
không so trực tiếp với FID chuẩn hoặc một protocol khác.

Đề xuất dùng một protocol có version/feature extractor/preprocess rõ ràng,
50k CIFAR training images không augmentation làm real reference, 10k generated
images ở1 NFE làm lượt đánh giá ban đầu. Ghi rõ **FID10k**, weights raw/EMA,
seed và class schedule; không gọi đây là FID50k. Khi so raw/EMA hoặc variants,
dùng cùng real stats/noise/labels. FID gộp classes không tự đo độ đúng class;
cần xem class-labelled samples hoặc metric riêng nếu muốn đánh giá conditioning.

Wasserstein bound trong proposal không phải một FID guarantee. Held-out MSE có
irreducible noise floor; plateau không chứng minh đã đạt nghiệm tốt. Test split
dùng để tune đã đóng vai trò validation, không còn là đánh giá cuối độc lập.

## 5. Nếu giữ nguyên Multi-Trace của notebook thì khác ở đâu?

Notebook lấy một schedule và interval cho cả batch:

```text
K ~ Uniform{1,2,3,4,5}
j ~ Uniform{0,...,K-1}
r = j/K
s ~ Uniform[r,(j+1)/K]
V = u(z_s,r,s) + (s-r)*stopgrad(JVP using u(z_s,s,s))
L_multi = MSE(V,C) + lambda_D*MSE(u(z_s,s,s),C)
```

| Điểm so | Proposal mục10/14 | Notebook Multi-Trace | Đánh giá |
|---|---|---|---|
| Shared diagonal tangent | `u(z,t,t)` | `u(z_s,s,s)` | Khớp nguyên tắc |
| JVP / stop-gradient / quadratic MSE | Có | Có | Khớp cấu trúc forward residual |
| Main edge | Luôn `r=0` | `r=j/K`, có cả interior | **Objective khác** |
| Time law trên diagonal | Uniform nếu chọnπt đều | Marginal `s` vẫn Uniform(0,1) | Khớp; không được nhầm với conditional time law của left edge |
| Left-edge weighting | Đều trên toàn[0,1] | Tập trung nhiều hơn ở thời gian nhỏ | Không bằng L0 của proposal |
| Inference chính | Một map1→0, 1 NFE | GhépK maps, K=1..5 | Few-step là mục tiêu mở rộng |
| Bound cho compositionK>1 | Chưa triển khai thành theorem hoàn chỉnh | Không được chứng minh trong notebook | Cần phân tích local errors và Lipschitz propagation |

Marginal `s` vẫn uniform: với một K cố định, chia[0,1] thànhK intervals, chọn mỗi
interval với xác suất1/K rồi lấy đều trong nó. Vì vậy diagonal regression vẫn
lấy trung bình đều theo thời gian.

Nhưng xác suất một training batch có `r=0` là

```text
P(r=0) = (1/5)*(1 + 1/2 + 1/3 + 1/4 + 1/5) = 137/300 ≈ 45,67%.
```

Phần còn lại khoảng54,33% train interior. Joint density của phần `r=0` theo s
là `q0(s)=number_of_K_with(s<1/K)/5`: bằng1 trên(0,1/5), rồi0.8,0.6,0.4,
và0.2 trên(1/2,1). Density này chưa chuẩn hóa; tích phân là45,67%. Do đó không
thể đổi tên `L_multi` thành `L0+lambda_D*LD` chỉ vì cả hai đều có diagonal branch.

### Tuy nhiên Multi-Trace vẫn giữ một liên hệ one-step có thể chứng minh

K=1 xuất hiện với xác suất1/5; khi đó r=0 và s~Uniform(0,1), đúng branch L0.
Gọi `EA` là **population residual energy** của interval branch (đã bỏ noise
variance constant), và `E0,ED` là residual energies theo norm tổng của proposal.
Vì các residual energies không âm:

```text
EA >= (1/5)*E0
=> E0 <= 5*EA
=> W2 <= sqrt(5*EA) + (M/sqrt(3))*sqrt(ED)
```

Đây là **suy luận bổ sung từ bound của proposal và distribution notebook**,
không phải theorem đã viết trong notebook. Nó vẫn cần mọi giả thiết regularity /
Jacobian của proposal và không dùng trực tiếp minibatch MSE làm residual energy.
Với squared bound tương ứng: `W2² <= 10*EA + (2*M²/3)*ED`.

Như vậy Multi-Trace không mất hoàn toàn one-step control, nhờ có explicitK=1;
nhưng không phải cùng objective/weighting của proposal, và bound thô trên yếu
hơn theo hệ số. Không thể từ đó kết luận FID kém hơn hay cần5 lần updates.
Interior training có thể giúp shared representation/optimization/few-step,
đúng như mục12 của proposal cho phép; hiệu quả cần experiment.

SamplingK>1 gọi model ở các `r>0`. Nếu chỉ train two-trace objective, những
inputs đó không được ràng buộc trực tiếp; chạy sampler5 NFE bằng model two-trace
không tự biến nó thành một few-step model được huấn luyện đúng cách. Đánh giá
1 NFE trước; thêm Multi-Trace như một experiment riêng nếu cần1–5 NFE.

## 6. Config repo và notebook không có cùng image budget

| Budget | Repo DiT | Notebook gốc |
|---|---:|---:|
| Effective batch | 64 | 512 |
| Optimizer updates | 200k | 100k |
| Training images processed | **12,8M** | **51,2M** |
| Warmup updates | 2.500 | 5.000 |
| Warmup images | **160k** | **2,56M** |

Giữ repo config là một baseline hợp lý về nguồn gốc, nhưng không phải so sánh
cùng lượng dữ liệu với notebook. Notebook thấy nhiều ảnh hơn4 lần và warmup
theo ảnh dài hơn16 lần. Comment "300k" trong notebook không khớp actual
`total_steps=100_000`; bảng dùng giá trị thực thi.

Nếu ưu tiên giữ image budget của notebook, có thể dùng DiT22M với micro128×
accumulation4, effective512, 100k updates, warmup5k và EMA.9999. **Loss vẫn có
thể giữ two-trace**, nên lựa chọn batch/budget này không buộc phải đổi sang
Multi-Trace. Nếu giữ effective64 mà muốn cùng image budget, cần800k updates
và40k warmup. Không tăng LR, clipping hoặc lambda_D theo số parameters một
cách tự động; không có quy tắc đó trong proposal.

## 7. Vì sao hợp lý, và điều gì chưa được bảo đảm?

**Hợp lý để kiểm tra proposal:** giữ DiT22M, one shared head, time uniform,
two-trace MSE và1 NFE; mượn eager training, EMA nếu chọn dùng, checkpoint/resume,
progress, diagnostics và evaluation workflow từ notebook. Những phần vận hành
đó không thêm interior term vào objective.

**Không phải bản tái hiện nguyên proposal:** chép raw MeanFlow loss/CFG/adaptive
L2 của backbone repo, hoặc giữ Multi-Trace interval sampling1..5 mà gọi loss
là two-trace one-step. Có thể làm experiment mở rộng, nhưng phải đặt tên và
mô tả phạm vi đúng.

Gradient clipping5.0 là clipping **parameter gradient `dL/dtheta`**. M trong
proposal bound **spatial Jacobian `du/dz`**; đây là hai đại lượng khác nhau.
Không thể nói clipping5 làm M=5 hoặc suy ra lambda_D=25/3. JVP RMS đo một
direction cũng không phải upper bound của Jacobian operator norm.

Lambda_D1 là một hyperparameter; `M²/3` chỉ là gợi ý khi biết boundM. MSE mean
giữ quadratic structure, còn adaptive weighting phụ thuộc sample error trong
source repo làm mất decomposition đơn giản thành residual energy + constant.

Các theorem nói residuals nhỏ thì map/distribution error bị chặn dưới giả thiết
đã nêu. Phụ lụcA chỉ rõ semi-gradient qua detachedJVP không phải full gradient
của residual functional. Vì thế proposal **không chứng minh SGD/AdamW sẽ hội
tụ**, DiT sẽ tốt hơn U-Net, 22M tốt hơn6M,200k bước đủ, hay một giá trị FID nào.

Để cô lập câu hỏi nghiên cứu, baseline đầu tiên nên là **DiT22M two-trace1NFE**
với budget/optimizer ghi rõ; sau đó so với **chính DiT22M Multi-Trace1..5** ở
cùng image budget và cùng FID protocol. So trực tiếp U-Net6M ở notebook với
DiT22M và thay cả loss/batch/EMA/FID cùng lúc không xác định được cải thiện
đến từ thành phần nào.
