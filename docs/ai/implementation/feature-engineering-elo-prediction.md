---
phase: implementation
title: Implementation — Feature Engineering & Re-baseline ELO realtime
description: Kết quả triển khai các phase và kết quả smoke-run cho re-baseline
---

# Implementation — Feature Engineering & Re-baseline ELO realtime

## Phạm vi đã triển khai

- Phase 1: Pipeline skeleton + config + projection + target encoding.
- Phase 2: Tabular transformer (ECO, EcoCategory, GameFormat, numeric, EloDiff offline).
- Phase 3: Move transformer (tokenizer, bigram, entropy, python-chess board-state, TF-IDF + SVD).
- Phase 4: Feature store pipeline (temporal split, batch processing, quality gate, metadata).
- Phase 5 (smoke): Re-baseline XGBoost + ablation + feature importance artifact.

## Artifact đầu ra (smoke)

- `data/features/smoke/feature_importance_top20.png`
- `data/features/smoke/rebaseline_summary.json`
- `data/features/smoke/train_features.parquet`
- `data/features/smoke/val_features.parquet`

## Kết quả smoke run (3k train + 3k val)

- Accuracy: `0.3723`
- Macro-F1: `0.2546`
- So với EDA baseline `0.4418`: **chưa đạt**, giảm `-0.0695` điểm accuracy.

## Kết quả ablation (smoke)

- `tabular_only`: accuracy ~0.3687
- `tabular_plus_sequence`: accuracy ~0.3723
- `all_features`: accuracy ~0.3723

Nhận xét nhanh:

- Sequence features có cải thiện nhẹ so với tabular-only trên sample nhỏ.
- Mức cải thiện chưa đủ để tiệm cận mục tiêu `>= 0.60`.

## Hướng tối ưu theo thứ tự đã chốt (Task 5.6)

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

## Kiểm thử mã

- `tests/test_tabular_transformer.py`
- `tests/test_move_transformer.py`
- `tests/test_feature_pipeline_phase4.py`

Tổng: 14 tests pass ở thời điểm cập nhật tài liệu.
