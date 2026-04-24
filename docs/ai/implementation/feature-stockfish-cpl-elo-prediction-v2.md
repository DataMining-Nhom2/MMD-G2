---
phase: implementation
title: Báo cáo Triển khai & Đánh giá V2 — Nâng cấp Stockfish Feature Engineering
description: Tổng kết quá trình triển khai 11 features Stockfish, kết quả chạy pipeline 30k ván, phân tích đánh giá XGBoost baseline và đề xuất chuyển hướng
---

# Báo cáo Triển khai V2: Stockfish CPL & WDL

## 1. Tóm tắt kết quả (Executive Summary)
- Chuyển đổi thành công bộ tính năng từ phiên bản sơ khai (6 features) sang V2 mở rộng (11 features tập trung vào Engine CPL, WDL loss và Phase separation).
- Script Pipeline chạy song song đa tiến trình (18 workers) đã xử lý xong tập `sample_30k` trong ~36 phút.
- Đã đào tạo mô hình XGBoost Stratified 5-Fold, đạt tổng Accuracy **47.59%**, tăng **+3.35%** so với phiên bản V1. Đáng chú ý, cấu hình V2 đã loại bỏ hoàn toàn các cột có thể gây Leakage về cấp độ người chơi như `BaseTime` hay `Increment`.

## 2. Các Features được triển khai (Feature Sets)
Kiến trúc V2 đã bổ sung 3 nhóm chính:
- **Nhóm A (Tỷ lệ rủi ro):** `avg_cpl`, `cpl_std`, `blunder_rate`, `mistake_rate`, `inaccuracy_rate`. (Đã chuyển từ dạng Count sang Rate để chuẩn hóa theo độ dài ván cờ).
- **Nhóm B (Phase CPL):** `opening_cpl`, `midgame_cpl`, `endgame_cpl`. Cung cấp góc nhìn về hiệu năng trong từng giai đoạn (ví dụ Beginner thường trượt chân từ Opening).
- **Nhóm C (WDL & Best-move Match):** `avg_wdl_loss`, `max_wdl_loss`, `best_move_match_rate`. Đánh giá khả năng kỳ thủ đánh các nước giống hệt AI và dao động xác suất chiến thắng.

## 3. Kết quả đánh giá (Ablation Study)

Đánh giá tác động phân nhánh của các nhóm Feature bằng thuật toán XGBoost đa lớp (5 trình độ):

```text
  Config                             Accuracy     Macro F1
  ──────────────────────────────────────────────────────
  A: Tabular Only                34.40% ±0.43% 32.16% ±0.43%
  B: Tabular + Nhóm A            44.39% ±0.10% 43.16% ±0.18%
  C: Tabular + A + B             46.47% ±0.40% 45.56% ±0.48%
  D: Full V2 (A+B+C)             47.59% ±0.53% 46.79% ±0.63%
```

**Nhận định phân tích:**
- **Nhóm A** là xương sống của mọi suy luận chiến thuật (boost 10% accuracy). Mức độ CPL trung bình càng cao, trình độ càng thấp.
- **Nhóm B** chia nhỏ giai đoạn giúp ML học được sự sụt giảm tập trung ở tàn cuộc (Endgame), vớt vát thêm 2%.
- Mặc dù vậy, **Nhóm C** chỉ kéo lên được +1.1%. Accuracy cao nhất dừng ở 47.6%, chưa đạt mục tiêu tham vọng >50%.

## 4. Hạn chế phát hiện (Limitations)
1. **Rào cản Phần Cứng (Hardware Bottleneck):** Tính toán cờ vua tại `depth=10` tốn lượng tài nguyên ngổng lồ. Tốc độ mất 36 phút cho 30.000 ván đồng nghĩa rằng dự án phải mất hơn **100 ngày** CPU-time nếu muốn ép toàn bộ 186 triệu trận thi đấu vào Baseline này. Quá rủi ro và không thể Scale-up.
2. **Ngôn ngữ chuỗi (Sequential Language):** Phương pháp trích xuất dữ liệu CPL đập dẹt (flatten), tức là làm mất toàn bộ tính chuỗi (Sequence Hành động - Thời gian) của ván cờ.

## 5. Kế hoạch chuyển hướng chiến lược (Strategic Pivot)
Dừng việc đào sâu vào XGBoost + Stockfish ở mức Big Data. Baseline phân loại 47.59% được giữ làm tiêu chuẩn tham chiếu (Benchmark). Hành động tiếp theo:
- Giới thiệu và chuẩn bị triển khai phương pháp theo **Deep Learning Sequential Model (LSTM / Transformer)**.
- Kết hợp kỹ thuật Rating Net chắt lọc ma trận hóa đặc trưng của ván cờ (Board state) và thời gian quản trị của kỳ thủ (`Time_spent`).
- Cải thiện tốc độ bằng I/O parsing và GPU Acceleration, hứa hẹn vượt ngưỡng 50-60% một cách đột phá với chi phí thời gian giảm hàng ngàn lần.
