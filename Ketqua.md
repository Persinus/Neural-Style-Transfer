# 🧠 Neural Style Transfer (NST) – Báo cáo Kết quả

## 📝 Tóm tắt đề bài
**Mục tiêu:**  
Xây dựng một ứng dụng **tạo ảnh nghệ thuật theo phong cách** (Neural Style Transfer) dựa trên mô hình **VGG19** được huấn luyện sẵn trên ImageNet.  

Người dùng có thể:
- Tải lên **ảnh nội dung (Content Image)** – ảnh gốc muốn giữ lại bố cục, vật thể.  
- Tải lên **ảnh phong cách (Style Image)** – ảnh mang màu sắc, họa tiết, phong cách nghệ thuật (ví dụ: tranh Van Gogh).  

Hệ thống sẽ kết hợp **nội dung** của ảnh thứ nhất với **phong cách** của ảnh thứ hai để tạo ra một tác phẩm mới.

---

## ⚙️ Nguyên lý mô hình
- **Mạng CNN dùng:** `VGG19 (pretrained on ImageNet)`
- **Không huấn luyện lại mô hình**, chỉ tối ưu hóa một ảnh “đích” (`target image`) để giảm tổng `loss`.  
- **Loss Function gồm 2 thành phần:**
  - 🧩 **Content Loss**: đo sự khác biệt đặc trưng nội dung giữa ảnh đích và ảnh gốc.  
  - 🎨 **Style Loss**: đo sự khác biệt phong cách qua **Gram Matrix** của các tầng CNN.  
- Ảnh đầu vào ban đầu là sự kết hợp giữa ảnh nội dung và nhiễu trắng (noise).  
- Quá trình tối ưu sử dụng **LBFGS optimizer** để cập nhật ảnh đích.

---

## 📊 Kết quả huấn luyện (trích log)
| Bước (Step) | Tổng Loss | Content Loss | Style Loss |
|--------------|------------|---------------|-------------|
| 0 | 622.9022 | 9.1775 | 0.0006 |
| 100 | 26.4273 | 9.7015 | 0.0000 |
| 200 | 20.0833 | 9.1539 | 0.0000 |
| 300 | 17.5357 | 8.8496 | 0.0000 |
| 400 | 16.3159 | 8.6629 | 0.0000 |
| 500 | 15.6263 | 8.5432 | 0.0000 |

---

## ✅ Những gì đoạn code đã thực hiện được

### 🎯 1. Mục tiêu chức năng
- ✅ Cho phép tải ảnh **Content** và **Style** từ URL (có thể dễ dàng thay bằng upload).  
- ✅ Chuẩn hóa và đưa ảnh về đúng kích thước, định dạng tensor cho VGG19.  
- ✅ Trích xuất đặc trưng từ các **layer nội dung và phong cách** của VGG19.  
- ✅ Tính **Gram Matrix** cho các đặc trưng phong cách.  
- ✅ Tối ưu ảnh đầu vào để giảm tổng loss (Content + Style).  
- ✅ Hiển thị ảnh kết quả ở từng mốc (Step 0, 100, 200, 300, 400, 500).  
- ✅ Cho kết quả hội tụ dần (Total Loss giảm đều → ảnh ổn định hơn).

---

## ⚠️ Những hạn chế / Chưa đạt yêu cầu hoàn chỉnh

### 🎨 1. **Phong cách chưa được thể hiện rõ**
- Style loss = 0 gần như ngay từ đầu ⇒ ảnh đầu ra gần như chỉ giữ nội dung, **không truyền phong cách**.  
- Có thể do:
  - **Tỷ lệ loss** chưa cân bằng (β = 1e6 quá lớn hoặc scale ảnh chưa chuẩn).  
  - Sai sót trong **chuẩn hóa ảnh hoặc tính Gram Matrix**.  
  - **Shape mismatch** khiến một vài tầng style không kích hoạt.

### 🧮 2. **Thiếu giao diện upload ảnh**
- Code hiện tại chỉ load ảnh qua URL, chưa có **UI hoặc input file** để người dùng chọn ảnh từ máy tính.

### ⚡ 3. **Tốc độ xử lý**
- NST theo phương pháp Gatys (2015) rất chậm (500 bước tối ưu), khó áp dụng thời gian thực.  
- Chưa áp dụng các kỹ thuật tăng tốc như:
  - Fast Style Transfer (Johnson et al. 2016)  
  - Adaptive Instance Normalization (AdaIN)

### 🖼️ 4. **Hiển thị kết quả hạn chế**
- Mỗi step chỉ hiển thị ảnh tạm, chưa lưu ảnh đầu ra cuối cùng vào file.  
- Chưa có so sánh trực quan giữa ảnh content/style/output.

---

## 💡 Đề xuất cải thiện

| Hướng nâng cấp | Mô tả |
|----------------|-------|
| ⚙️ Tinh chỉnh loss weights | Giảm `alpha`, tăng `beta` để ảnh có phong cách rõ hơn. |
| 🧰 Chuẩn hóa ảnh | Đảm bảo Normalize/Unnormalize đúng chuẩn của VGG19. |
| 🪄 Giao diện người dùng | Dùng Streamlit / Gradio để người dùng upload ảnh. |
| 🚀 Tăng tốc | Dùng Fast Neural Style Transfer (huấn luyện riêng từng phong cách). |
| 💾 Lưu ảnh kết quả | Ghi `final_img.save("output.jpg")` sau bước cuối. |

---

## 🧾 Tổng kết
Đoạn code đã **triển khai đúng quy trình Neural Style Transfer cổ điển**, sử dụng **VGG19 pretrained**, có cơ chế trích xuất đặc trưng và tối ưu ảnh đầu vào.  
Tuy nhiên, **phong cách nghệ thuật chưa được thể hiện rõ**, do Style Loss không ảnh hưởng đáng kể trong quá trình huấn luyện.  
Khi cân chỉnh lại tỷ lệ loss và chuẩn hóa dữ liệu, mô hình có thể cho kết quả tốt hơn.

---

**Từ khóa:** `Neural Style Transfer`, `VGG19`, `Gram Matrix`, `Content Loss`, `Style Loss`, `PyTorch`, `Gatys et al. 2015`

Cân bằng giữa Content và Style: Việc điều chỉnh trọng số giữa Content Loss và Style Loss sẽ ảnh hưởng lớn đến kết quả cuối cùng.
Chất lượng ảnh đầu ra: Kích thước và độ phân giải của ảnh có thể ảnh hưởng đến chi tiết nghệ thuật.
Link code mẫu tham khảo:
TensorFlow Tutorial: https://www.tensorflow.org/tutorials/generative/style_transfer
PyTorch Tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
