# Nguồn code

`models/dit.py`, `data.py` và cấu trúc training được kế thừa/adapt từ
`../backbone_IMF`, nguồn [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow),
commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`.
MIT notice được giữ trong `MeanFlow.LICENSE`.

Model giữ attention, adaLN-zero, embedding và unpatchify của nguồn; thay hai head
bằng một head u, bỏ CFG embedding, cố định positional embedding, thay lớp timm
bằng PyTorch tương đương và RMSNorm với epsilon 1e-6 để hỗ trợ PyTorch 2.2.
Loss Trace-iMF được viết theo tài liệu idea của người dùng.
