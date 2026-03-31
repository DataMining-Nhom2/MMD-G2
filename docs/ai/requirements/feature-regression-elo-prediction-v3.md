---
phase: requirements
title: Requirements V3 — Chuyển từ Classification sang Regression dự đoán ELO liên tục
description: Chuyển đổi mô hình từ phân loại 5 lớp sang hồi quy dự đoán điểm ELO số, tái sử dụng 11 features Stockfish V2
---

# Requirements V3 — Chuyển từ Classification sang Regression

## Problem Statement
**Bài toán chúng ta đang giải quyết là gì?**

- **Vấn đề cốt lõi**: V2 Classification (5 lớp) đạt **47.59% Accuracy** — không phá được mốc 50%. Nguyên nhân sâu xa không nằm ở features (đã mạnh với 11 features Stockfish) mà ở **bản chất bài toán Classification**:
  - **Ranh giới giả tạo**: ELO 1399 vs 1401 cách nhau 2 điểm nhưng rơi vào 2 class khác nhau → Model bị phạt 100% cho sai lệch nhỏ.
  - **Phong độ dao động**: Kỳ thủ ELO 1500 có thể đánh ván như 1300 hoặc 1700. Classification coi mọi lần lệch class là "Sai hoàn toàn".
  - **"Cái bụng" nát**: 3 class giữa (Intermediate/Advanced/Expert) chồng chéo cực độ về features. Model không thể tạo hàng rào phân cắt rõ ràng.

- **Giải pháp V3**: Chuyển sang **Regression** — dự đoán trực tiếp điểm ELO liên tục (ví dụ: 1542) thay vì dán nhãn class. Sai số được cân đo hợp lý hơn (dự đoán 1542 cho kỳ thủ 1600 = lệch 58 ELO, vẫn chấp nhận được).

- **Triết lý**: Giữ nguyên toàn bộ Features V2 (Lò Bát Quái), chỉ thay đổi "Phép Đo" (Loss Function & Metric).

## Goals & Objectives

### Mục tiêu chính (Primary Goals)
1. **Chuyển target** từ `ModelBand` (Int 0-4) sang `EloAvg` (Float liên tục, range 400 → 3700+)
2. **Thay model** từ `XGBClassifier` sang `XGBRegressor` với `objective: reg:squarederror`
3. **Đo bằng metrics regression**: MAE, RMSE, R² thay vì Accuracy/F1
4. **So sánh ngược**: Chuyển ELO dự đoán → Class để đối chiếu trực tiếp với V2 (47.59%)
5. **Trực quan hóa** kết quả bằng 4 biểu đồ cốt lõi

### Non-goals (Ngoài phạm vi)
- **KHÔNG** thay đổi features: Dùng nguyên `sample_30k_features_v2.parquet`
- **KHÔNG** thay đổi pipeline: `src/feature_engineering.py` giữ nguyên
- **KHÔNG** chạy lại Stockfish — tái sử dụng 100% kết quả analysis V2
- **KHÔNG** thêm features mới (TF-IDF, Sequence) — đó là V4
- **KHÔNG** tuning hyperparameter phức tạp (Optuna) — đó là V4

## User Stories & Use Cases

### Use Case chính
- **Dự đoán ELO từ ván cờ**: Cho 1 ván cờ (PGN), hệ thống phân tích bằng Stockfish → 11 features → XGBRegressor → Trả ra con số ELO dự đoán (ví dụ: "Ván này đánh ở trình độ ~1542 ELO").

### Các kịch bản
1. **Kỳ thủ Beginner (600 ELO)**: Nhiều blunder, CPL cao → Model dự đoán ~650 ELO (lệch 50) ✅
2. **Kỳ thủ Master (2500 ELO)**: best_move_match_rate cao, WDL loss thấp → Model dự đoán ~2450 ELO ✅
3. **Kỳ thủ Advanced (1800 ELO)**: Features trung bình → Model dự đoán ~1700-1900 → MAE ~100-150 ✅
4. **Ván "off-day"**: Kỳ thủ 2000 ELO đánh dở ván này → Model dự đoán 1600 (đúng với chất lượng ván chơi)

## Success Criteria

| Metric | Mục tiêu | Ghi chú |
|--------|----------|---------|
| **MAE** (Mean Absolute Error) | **≤ 220 ELO** | Optimistic ≈ 180, pessimistic ≈ 250 |
| **RMSE** (Root Mean Squared Error) | ≤ 280 ELO | Phạt nặng hơn ván dự đoán lệch cực lớn |
| **R² Score** | ≥ 0.55 | Tỷ lệ phương sai được giải thích (> 0.5 = mô hình có ý nghĩa) |
| **Tương đương Classification** | Accuracy ≥ 55% | Đổi ngược ELO dự đoán → Class rồi so sánh |

> **Ngưỡng Thành Công:** MAE ≤ 220 = THẮNG. MAE ≤ 180 = ĐẠI THẮNG.

## Constraints & Assumptions

### Technical Constraints
- **Data**: Tái sử dụng `data/features/sample_30k_features_v2.parquet` (30k ván, ~117 features)
- **Target**: Cần lấy `EloAvg` từ `data/processed/sample_30k.parquet` gốc (file features V2 chỉ lưu `ModelBand`)
- **Model**: XGBRegressor (XGBoost đã cài sẵn trong environment)
- **CPU**: i5-14600KF — training nhanh (không cần GPU cho XGBoost)

### Assumptions
1. Phân phối EloAvg: `mean ≈ 1650`, `std ≈ 400` (6000 ván/band × 5 bands)
2. KFold (không Stratified) phù hợp cho target regression liên tục
3. XGBoost xử lý tốt NaN trong `midgame_cpl`/`endgame_cpl` ở chế độ regression
4. Features correlation với ELO đủ mạnh để R² > 0.5 (đã chứng minh qua Classification V2)

## Questions & Open Items

1. ~~Nguồn target EloAvg~~ → Lấy từ `sample_30k.parquet` gốc, join theo index
2. **KFold vs RepeatedKFold**: Có nên dùng RepeatedKFold (3×5) cho ước lượng ổn định hơn?
3. **Huber Loss vs MSE**: Nên dùng `reg:squarederror` (MSE) hay `reg:pseudohubererror` (robust với outliers)?
4. **ELO normalization**: Có nên normalize EloAvg (StandardScaler) trước khi train regression? XGBoost thường không cần, nhưng có thể cải thiện convergence.
