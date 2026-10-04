# Nguồn code

`models/dit.py`, `data.py` và cấu trúc training được kế thừa/adapt từ
`../backbone_IMF`, nguồn [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow),
commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`.
MIT notice được giữ trong `MeanFlow.LICENSE`.

Model giữ attention, adaLN-zero, embedding và unpatchify của nguồn; thay hai head
bằng một head u, bỏ CFG embedding, cố định positional embedding, thay lớp timm
bằng PyTorch tương đương và RMSNorm với epsilon 1e-6 để hỗ trợ PyTorch 2.2.
Loss Trace-iMF được viết theo tài liệu idea của người dùng.

`models/unet.py` giữ U-Net topology/shared head và `multi_trace_imf.py` giữ
equal-grid interval schedule từ notebook `Untitled0.ipynb` do người dùng cung
cấp. Bản 22M widen base44→84 và cond160→356; thứ tự API được chuẩn hóa thành
`(z,t,r,y)`. GroupNorm nhận contiguous input để hỗ trợ forward AD trên PyTorch2.2.
EMA/accumulation/resume/monitoring là phần tích hợp của IMF; không thực thi
các cell Colab/unzip/Drive từ notebook tham chiếu.
