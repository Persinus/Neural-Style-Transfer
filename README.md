# Neural-Style-Transfer
# Neural Style Transfer — README

## Tóm tắt dự án
Một ứng dụng Neural Style Transfer (NST) cho phép người dùng tải lên **ảnh nội dung (content image)** và **ảnh phong cách (style image)**, sau đó hệ thống kết hợp nội dung của ảnh thứ nhất với phong cách của ảnh thứ hai để tạo ra một tác phẩm nghệ thuật mới.

Mục tiêu của repository này là cung cấp một hướng dẫn, mã mẫu và ứng dụng demo (CLI / Web) để thực hiện NST theo phương pháp tối ưu hóa truyền thống (Gatys et al.) và chỉ ra các hướng mở để áp dụng các phương pháp "fast style transfer" nếu cần tăng tốc.

---

## Tính năng
- Cho phép tải ảnh nội dung và ảnh phong cách từ máy người dùng.
- Triển khai phương pháp Neural Style Transfer dựa trên mạng VGG19 pretrained (không fine‑tune).
- Hỗ trợ điều chỉnh trọng số `content_weight`, `style_weight`, `tv_weight` (total variation) và số bước tối ưu hóa.
- Lưu ảnh kết quả ở nhiều kích thước (preview + high-res) và hỗ trợ rescaling/center‑crop.
- Cấu hình bằng file YAML/JSON hoặc tham số dòng lệnh.
- Có ví dụ chạy nhanh bằng CPU/GPU (nếu có CUDA).
- (Tùy chọn) Demo web bằng Streamlit hoặc Flask + simple frontend.

---

## Kiến trúc & Ý tưởng chính
1. **Mạng trích xuất đặc trưng**: Sử dụng VGG19 pretrained trên ImageNet. Cố định trọng số, chỉ dùng để tính feature maps.
2. **Content features**: Lấy từ một hoặc vài layers nông (ví dụ `conv4_2` trong VGG19).
3. **Style features**: Lấy từ nhiều layers (ví dụ `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`). Với mỗi layer, tính ma trận Gram của feature maps để biểu diễn phong cách.
4. **Image to optimize**: Bắt đầu từ ảnh nhiễu trắng, hoặc copy từ ảnh nội dung rồi tối ưu trực tiếp pixel của ảnh này để giảm tổng loss.
5. **Loss function**:
   - Content loss: MSE giữa content features của ảnh đang tối ưu và content features của ảnh nội dung.
   - Style loss: Tổng MSE giữa Gram matrices của ảnh đang tối ưu và ảnh phong cách (cân bằng theo layer weights).
   - Total variation (TV) loss: Giúp ảnh mượt mà, giảm nhiễu cục bộ.
6. **Optimization**: Sử dụng optimizer (LBFGS hoặc Adam); thực hiện N bước tối ưu hóa và lưu ảnh trung gian nếu cần.

---

## Dependencies (gợi ý)
- Python 3.8+
- torch, torchvision (nếu dùng PyTorch)
- pillow (PIL)
- numpy, matplotlib (tuỳ chọn để hiển thị)
- tqdm
- pyyaml (nếu dùng file cấu hình)
- streamlit hoặc flask (cho demo web)

Cài nhanh (ví dụ dùng pip):

```bash
pip install torch torchvision pillow numpy matplotlib tqdm pyyaml streamlit
```

> Nếu sử dụng GPU, cài torch phù hợp với CUDA phiên bản máy bạn: https://pytorch.org/get-started/locally/

---

## Hướng dẫn cài đặt & chạy (Quick start)
### 1) Clone repo

```bash
git clone <this-repo-url>
cd neural-style-transfer-app
```

### 2) Chạy ví dụ PyTorch (CLI)
File chính: `run_style_transfer.py` (mô tả API dưới đây)

```bash
python run_style_transfer.py \
  --content images/content.jpg \
  --style images/style.jpg \
  --output results/out.jpg \
  --content-weight 1e5 \
  --style-weight 1e10 \
  --tv-weight 1e-6 \
  --steps 500
```

Kết quả sẽ được lưu ở `results/out.jpg`.

### 3) Chạy demo nhanh bằng Streamlit

```bash
streamlit run app/streamlit_app.py
```
Mở trình duyệt tới http://localhost:8501 để upload ảnh và xem kết quả.

---

## File & cấu trúc thư mục gợi ý
```
neural-style-transfer-app/
├─ README.md
├─ requirements.txt
├─ run_style_transfer.py
├─ style_transfer/
│  ├─ model.py           # VGG wrapper, feature extractors
│  ├─ losses.py          # content loss, style loss, tv loss
│  ├─ utils.py           # load image, preprocess, postprocess
│  └─ optimizer.py       # chạy vòng lặp tối ưu
├─ app/
│  ├─ streamlit_app.py   # frontend demo nhanh
│  └─ flask_app.py       # (tùy chọn) API endpoint
├─ images/               # ví dụ content/style
└─ results/              # ảnh output
```

---

## Chi tiết kỹ thuật (recommendations & snippets)
### Layers được dùng cho VGG19
- **Content layer(s)**: `conv4_2` (hoặc `relu4_2`).
- **Style layers** (thường dùng): `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`.

### Tính Gram matrix (PyTorch snippet)
```python
# x: Tensor shape (B=1, C, H, W)
features = x.view(C, H*W)
G = torch.mm(features, features.t())  # C x C
# thường normalize by (C*H*W)
G_normalized = G / (C * H * W)
```

### Content loss
```python
content_loss = torch.nn.functional.mse_loss(target_features, content_features)
```

### Style loss
```python
style_loss = 0.0
for t_feat, s_gram in zip(target_feats, style_grams):
    t_gram = gram_matrix(t_feat)
    style_loss += torch.nn.functional.mse_loss(t_gram, s_gram) * layer_weight
```

### Total variation loss (L_tv)
```python
def tv_loss(img):
    x_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
    y_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
```

### Optimizer
- LBFGS thường cho kết quả tốt với số bước ít (nhưng cần wrapper callback).  
- Adam dễ dùng và linh hoạt (cần chọn learning rate ~0.01–0.1 tuỳ thang pixel normalization).

---

## Thông số khuyến nghị (hyperparameters)
- `content_weight`: 1e3 — 1e6 (ví dụ 1e5)
- `style_weight`: 1e6 — 1e11 (ví dụ 1e9 — 1e10)
- `tv_weight`: 1e-6 — 1e-2 (ví dụ 1e-6)
- `steps`: 300 — 2000 (tùy độ phân giải và thời gian chờ)
- `image_size` during dev: 256–512; for high quality: 1024+

> Lưu ý: Scale của ảnh và normalization (mean/std) của VGG ảnh hưởng tới learning rate và cân bằng loss — thường dùng mean/std giống preprocessing ImageNet.

---

## Tối ưu hoá tốc độ & mẹo thực nghiệm
- **Multi-scale / pyramidal approach**: tối ưu ảnh ở độ phân giải thấp rồi upsample và tiếp tục tối ưu ở các độ phân giải lớn hơn.
- **Use content init**: bắt đầu từ ảnh content (thay vì nhiễu) thường giữ được bố cục nhanh hơn.
- **Fast Neural Style Transfer**: nếu muốn xử lý real‑time, cần huấn luyện một network chuyển đổi feed‑forward cho mỗi phong cách (tham khảo Johnson et al., Ulyanov, etc.).
- **Mixed precision**: dùng AMP nếu có GPU hiện đại để tăng tốc và giảm VRAM.
- **Crop & tile**: để xử lý ảnh rất lớn, chia ảnh thành các tile, áp phong cách riêng cho từng tile rồi ghép nối (cần blend mịn).

---

## Demo Web (gợi ý thiết kế UI)
- Trang upload gồm hai thẻ: Content image, Style image (drag & drop).
- Các control: sliders cho content_weight, style_weight, tv_weight, nút chọn optimizer, số bước.
- Preview panel: show ảnh nội dung, phong cách và ảnh kết quả (preview nhỏ) + nút export full resolution.
- Progress bar + logs (step, loss content/style/tv).
- Option để lưu và tải về kết quả.

---

## Nghiên cứu & tham khảo (ví dụ links)
- TensorFlow tutorial: https://www.tensorflow.org/tutorials/generative/style_transfer
- PyTorch tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
- Bài gốc: "A Neural Algorithm of Artistic Style" — Gatys et al.

---

## Các cải tiến mở rộng (ideas)
- Thêm chế độ **multi-style blending** (mix nhiều ảnh phong cách với tỉ lệ khác nhau).
- Huấn luyện fast‑style network cho nhiều phong cách (tổ hợp conditional instance normalization để chứa nhiều phong cách trong 1 model).
- Bổ sung **color preservation** hoặc **palette transfer** để giữ màu sắc gốc.
- UI nâng cao: lịch sử biến thể, undo/redo, brush mask để áp phong cách chọn lọc lên vùng cụ thể của ảnh.

---

## Vấn đề pháp lý & bản quyền
- Việc sử dụng ảnh phong cách (ví dụ tranh nghệ sĩ) để sinh tác phẩm mới có thể liên quan đến bản quyền/triển lãm thương mại tùy luật địa phương. Hãy kiểm tra luật bản quyền trước khi sử dụng cho mục đích thương mại.

---

## License
Chọn license phù hợp cho repository (MIT/Apache‑2.0 nếu muốn mở). Nếu bạn muốn restrictive hơn, ghi rõ.

---

## Ghi chú khi đóng gói để nộp bài / báo cáo
- Mô tả rõ cách chạy (step-by-step).
- Đưa kèm 2–3 cặp ảnh input (content/style) và ảnh output mẫu.
- Ghi thời gian chạy (CPU/GPU) và các thông số để người chấm có thể tái tạo.

---

## Muốn mình hỗ trợ thêm gì không?
- Mình có thể tạo sẵn file `run_style_transfer.py` (PyTorch) mẫu.
- Hoặc tạo demo Streamlit hoàn chỉnh để bạn deploy nhanh.


