---
phase: implementation
title: Implementation — Feature Engineering ELO realtime (FE-only)
description: Kết quả triển khai phần Feature Engineering độc lập với huấn luyện/evaluation
---

# Implementation — Feature Engineering ELO realtime (FE-only)

## Phạm vi đã triển khai

- Phase 1: Pipeline skeleton + config + projection + target encoding.
- Phase 2: Tabular transformer (ECO, EcoCategory, GameFormat, numeric, EloDiff offline).
- Phase 3: Move transformer (tokenizer, bigram, entropy, python-chess board-state, TF-IDF + SVD).
- Phase 4: Feature store pipeline (temporal split, batch processing, quality gate, metadata).
- Phase 5 (train/eval): **Deferred** sang Model Training phase.

## Artifact đầu ra FE

- `data/features/train_features.parquet`
- `data/features/val_features.parquet`
- `data/features/feature_columns.json`
- `data/features/tfidf_vocabulary.pkl`
- `data/features/svd_components.pkl`

## Artifact smoke train/eval (tham khảo lịch sử)

- `data/features/smoke/feature_importance_top20.png`
- `data/features/smoke/rebaseline_summary.json`
- `data/features/smoke/train_features.parquet`
- `data/features/smoke/val_features.parquet`

## Kết quả smoke train/eval trước đó (không dùng làm tiêu chí FE-only)

- Accuracy: `0.3723`
- Macro-F1: `0.2546`
- So với EDA baseline `0.4418`: **chưa đạt**, giảm `-0.0695` điểm accuracy.

## Iteration tối ưu #2 (move-first, lịch sử)

Thay đổi chính:

- `n_ply`: 10 -> 15
- TF-IDF: bigram-only -> 1-2gram, `max_features=500`, `sublinear_tf=True`
- Thêm move meta features: `unique_move_ratio`, `capture_ratio`, `check_symbol_ratio`
- Rebaseline: thêm class-balanced sample weights

Kết quả smoke mới:

- Accuracy: `0.3673` (giảm nhẹ so với smoke trước)
- Macro-F1: `0.3068` (tăng rõ so với smoke trước)

Nhận xét:

- Kết quả này chỉ dùng tham khảo cho giai đoạn huấn luyện.
- FE-only hiện tập trung vào chất lượng feature/artifact/pipeline thay vì metric model.

## Kết quả ablation (smoke, lịch sử)

- `tabular_only`: accuracy ~0.3687
- `tabular_plus_sequence`: accuracy ~0.3723
- `all_features`: accuracy ~0.3723

Nhận xét nhanh:

- Sequence features có cải thiện nhẹ so với tabular-only trên sample nhỏ.
- Mức cải thiện chưa đủ để tiệm cận mục tiêu `>= 0.60`.

## Hướng tối ưu FE-only hiện tại

1. Ổn định chất lượng transform:

- chuẩn hóa tokenizer SAN, giảm lỗi parsing edge-case.
- đảm bảo schema train/val không drift khi batch lớn.

2. Củng cố data quality gate:

- theo dõi drop rate `<5 ply` theo split.
- nếu warning/fail, phải xuất thống kê theo `GameFormat` và `ModelBand`.

3. Sẵn sàng bàn giao cho Model Training phase:

- chốt metadata artifacts (`feature_columns.json`, TF-IDF vocab, SVD components).
- lock seed và cấu hình để tái lập.

## Hướng tối ưu theo thứ tự đã chốt cho phase train/eval (Deferred - Task 5.6)

1. Move sequence:

- tăng chất lượng tokenization SAN (normalize ký hiệu check/mate, promotion, annotation).
- tăng chất lượng vectorizer (n-gram mix 1-2 hoặc 1-3, min_df/max_df hợp lý).
- tăng độ phủ signal từ python-chess (thêm feature tactical đơn giản từ board state đầu game).

2. ECO encoding:

- thử tăng `ECO_TOP_N` (100 -> 150/200) và gộp ECO hiếm vào nhóm fallback có kiểm soát.
- kiểm tra leakage gián tiếp từ ECO hiếm/mất cân bằng.

3. Game metadata:

- hiệu chỉnh clipping/normalization theo quantile động từ train set lớn hơn.
- kiểm tra tương tác giữa GameFormat và Increment/BaseTime.

## Rule thực thi thử nghiệm

- Áp dụng **Notebook-first** cho mọi test/trial exploratory.
- Chỉ chạy terminal cho automation/test batch hoặc full run đã chốt.

Notebook chính:

- `notebooks/feature-engineering/rebaseline/rebaseline_smoke.ipynb`

## Kiểm thử FE

- `tests/test_tabular_transformer.py`
- `tests/test_move_transformer.py`
- `tests/test_feature_pipeline_phase4.py`

Tổng: 16 tests pass, 1 warning sklearn SVD (runtime warning trên sample nhỏ), cập nhật gần nhất.
