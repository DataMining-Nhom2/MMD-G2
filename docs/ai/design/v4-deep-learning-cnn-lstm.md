---
phase: design
title: V4 Deep Learning Architecture (CNN-BiLSTM)
description: Thiết kế kiến trúc mô hình Neural Network dự đoán ELO từ PGN và Thời gian (dựa trên paper arxiv 2409.11506). Bỏ qua hoàn toàn Stockfish engine.
---

# 🧠 Kiến trúc Mô hình Deep Learning: ELO Prediction (CNN-BiLSTM)

**Tài liệu tham khảo cốt lõi:** *"Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM"* (arXiv:2409.11506)

---

## 1. Tư duy Chiến lược (Hủy bỏ Stockfish)
Thay vì dùng XGBoost và tốn tài nguyên chạy Stockfish để trích xuất *Centipawn Loss* hay *Blunders*, chúng ta sẽ bắt Mạng Nơ-ron tự "nhìn" bàn cờ và "cảm nhận" thời gian trôi qua, từ đó định giá trình độ kỳ thủ.
Mô hình cũ XGBoost (MAE: 247.8) sẽ được dùng làm Baseline để nộp báo cáo.

---

## 2. Đầu vào Dữ liệu (Inputs)
Chúng ta sẽ rút trích trực tiếp từ file PGN nguyên thủy (`.pgn`). Với mỗi nước đi $t$ trong ván cờ, chúng ta cần 2 Vector:
1.  **Đặc trưng Không gian (Spatial - Bàn cờ):** 
    - Bàn cờ 8x8 sẽ được mã hóa thành các **Bitboards** (One-hot encoding của 12 loại quân cờ: 6 Trắng, 6 Đen).
    - Kích thước ma trận tại một thời điểm $t$: `(12, 8, 8)` hoặc `(14, 8, 8)` nếu tính cả đặc trưng lượt đi, nhập thành.
2.  **Đặc trưng Thời gian (Clock - Temporal):**
    - Lượng thời gian suy nghĩ cho nước đi $t$ (ính bằng giây hoặc normalized).
    - Thời gian còn lại trên đồng hồ.

=> Toàn bộ ván cờ (chiều dài chuỗi $L$) sẽ thành khối Tensor (Tensor Block) kích thước `(L, Khung_hình, Thời_gian)`.

---

## 3. Kiến trúc Mô hình (Model Architecture - PyTorch)
Mô hình được chia làm 2 tầng (Two-stage Network):

### Tầng 1: Mạng Tích Chập (CNN Feature Extractor)
- Nhận đầu vào là Ma trận Bàn cờ `[Batch, 14, 8, 8]`.
- Qua các lớp Convolutional 2D (kèm ReLU, MaxPool) để rút trích sự 복_tạp (complexity), các thế ghim, thế chiếu, sự suy yếu cấu trúc tốt.
- Đầu ra tầng 1: Vector không gian (Flattened Embeddings) kích thước `N`.

### Tầng 2: Mạng Dài-Ngắn Hai Chiều (Bi-LSTM)
- Vector không gian `N` ghép nối (Concatenate) với `Clock_time` tạo thành Feature Vector tổng hợp `V_t`.
- Bi-LSTM đọc chuỗi `V_1 -> V_2 -> ... -> V_L`.
- Đầu ra của LSTM đi qua một tầng Fully Connected (Linear) để dự đoán số ELO liên tục sau mỗi nước đi, hoặc gộp lại (Pooling) để dự đoán ELO cuối trận.

---

## 4. Kế hoạch Tác chiến (To-do List)
- [ ] **B1: Viết Data Loader & Parser:** Dùng thư viện `python-chess` dịch PGN sang Bitboard Tensors. Không cần Parser siêu việt 600 dòng như bản MMD nữa.
- [ ] **B2: Build Class CNN:** Dựng mạng CNN bằng PyTorch.
- [ ] **B3: Build Mạch LSTM:** Ghép CNN với BiLSTM.
- [ ] **B4: Đào tạo (Training):** Phân chia mini-batch, dùng Loss Function Hàm L1 (MAE Loss) hoặc MSE.
- [ ] **B5: Đối sánh:** Trực tiếp dẫm lên xác của báo cáo XGBoost để tôn vinh sự ưu việt của Deep Learning.
