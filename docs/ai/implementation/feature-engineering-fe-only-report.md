---
phase: implementation
title: Báo cáo FE-only — Feature Engineering ELO realtime
description: Báo cáo trạng thái triển khai phần Feature Engineering độc lập với train/eval
updated_at: 2026-03-11
---

# Báo cáo FE-only — Feature Engineering ELO realtime

## 1) Mục tiêu báo cáo

Báo cáo này tổng hợp trạng thái **Feature Engineering (FE-only)** cho bài toán dự đoán `ModelBand` từ dữ liệu cờ vua, không bao gồm huấn luyện mô hình và không dùng metric model làm tiêu chí pass/fail cho phase FE.

## 2) Phạm vi FE-only đã chốt

Bao gồm:

- Thiết kế và triển khai pipeline trích xuất feature từ Parquet nguồn.
- Tabular features: ECO, EcoCategory, GameFormat, BaseTime/Increment/NumMoves.
- Move sequence features: tokenize SAN, TF-IDF + SVD, entropy, board-state features (`python-chess`).
- Feature store: temporal split, batch processing, data quality gate, metadata artifacts.
- Kiểm thử tự động cho transformer và pipeline FE.

Không bao gồm (deferred sang Model Training phase):

- Re-baseline XGBoost, ablation, feature importance phục vụ đánh giá model.
- Mục tiêu accuracy `>= 60%`.

## 3) Thành phần kỹ thuật FE đã hoàn tất

- `src/feature_engineering.py`
  - `BatchLoader`: đọc dữ liệu theo batch để tránh OOM.
  - `TabularTransformer`: chuẩn hóa và encode tabular features.
  - `MoveTransformer`: tokenize SAN, TF-IDF/SVD, entropy, đặc trưng board-state.
  - `FeaturePipeline`: chạy end-to-end theo split, ghi parquet, kiểm tra schema.
- `src/feature_config.py`
  - Cấu hình nhất quán cho n-gram, SVD, quality gate, seed, đường dẫn artifact.

## 4) Artifacts FE đầu ra

Artifacts chính:

- `data/features/train_features.parquet`
- `data/features/val_features.parquet`
- `data/features/feature_columns.json`
- `data/features/tfidf_vocabulary.pkl`
- `data/features/svd_components.pkl`

Artifacts smoke lịch sử (không dùng để nghiệm thu FE-only) vẫn được giữ để tham khảo:

- `data/features/smoke/rebaseline_summary.json`
- `data/features/smoke/feature_importance_top20.png`

## 5) Chất lượng và kiểm thử

Kết quả test FE mới nhất:

- `16 passed, 1 warning` (sklearn `TruncatedSVD` runtime warning trên sample nhỏ).
- Bộ test đã chạy:
  - `tests/test_tabular_transformer.py`
  - `tests/test_move_transformer.py`
  - `tests/test_feature_pipeline_phase4.py`

Đánh giá:

- Pipeline FE đang ổn định ở mức unit/integration test hiện có.
- Cần thêm full-scale benchmark để xác nhận SLA `< 4 giờ` trên full 187M rows.

## 6) Data quality gate FE

- Chính sách: loại bản ghi `<5 ply` trong quá trình tạo feature.
- Ngưỡng:
  - Pass: `<= 1.5%` mỗi split.
  - Warning: `(1.5%, 3%]`.
  - Fail: `> 3%`.

## 7) Rủi ro còn lại trong FE-only

- Chưa chạy benchmark full-scale nên chưa chốt được SLA thực tế.
- Cần theo dõi drift schema giữa train/val khi chạy trên dữ liệu lớn.
- Warning của SVD trên sample nhỏ cần theo dõi thêm khi chạy batch thực tế lớn.

## 8) Kết luận

Phase FE-only đã có đầy đủ pipeline, artifacts, và kiểm thử nền tảng để bàn giao cho giai đoạn train/eval. Từ thời điểm này, các tác vụ model metric (accuracy/macro-F1/ablation) được coi là phạm vi của **Model Training phase**.
