---
phase: requirements
title: Requirements — Feature Engineering cho dự đoán ELO realtime
description: Xây dựng feature set từ dữ liệu cờ vua thô để phục vụ mô hình ML dự đoán ELO
---

# Requirements — Feature Engineering cho Dự đoán ELO Realtime

## Problem Statement

**Pha EDA đã xác nhận**: first-N-moves có đủ signal để dự đoán ELO (5-class accuracy ~44% với chỉ tabular features). Pha Feature Engineering cần **trích xuất và encode features hiệu quả** từ dữ liệu thô thành input vector cho mô hình ML.

EDA baseline (XGBoost tabular, 18 features): **44.18% acc** — cần tăng lên **≥ 60%** với feature engineering tốt.

**Nhóm người dùng bị ảnh hưởng trực tiếp**:

- ML Engineer: cần feature vector ổn định để huấn luyện và tối ưu mô hình phân lớp ELO.
- Data Engineer: cần pipeline batch xử lý được full dữ liệu 187M rows mà không OOM.
- Data Scientist: cần feature set có thể giải thích và đánh giá được hiệu quả theo nhóm ELO.
- MLOps Engineer: cần artifact feature store nhất quán để tái lập train/val và tích hợp training workflow.

## Goals & Objectives

### Mục tiêu chính

1. **Tạo move sequence features**: Encode chuỗi nước đi (first 10-15 ply) thành vector số học
2. **Tạo opening complexity features**: Entropy, diversity, n-gram frequency per ELO band
3. **Tạo ECO embedding**: One-hot hoặc learned embedding cho top ECO codes
4. **Pipeline scalable**: Feature extraction chạy được trên full 187M rows (batch processing)
5. **Train/val split**: Tạo dataset split theo thời gian (Dec 2025 train, Jan 2026 val)

### Mục tiêu phụ

6. **Feature store**: Lưu features đã trích xuất ra Parquet để không phải tính lại
7. **Feature importance baseline**: Re-evaluate với feature set mới trên XGBoost GPU
8. **Missing value strategy**: Xử lý games có ít moves (< 5 ply)

### Non-goals

- Không xây model cuối (chỉ feature engineering + evaluation)
- Không dùng Stockfish engine eval (quá chậm cho 187M games)
- Không build serving pipeline (giai đoạn 4)

## Feature Candidates (từ EDA)

### Tier 1 — Must Have (EDA confirmed high importance)

| Feature                   | Source       | Encoding           | Notes                                    |
| ------------------------- | ------------ | ------------------ | ---------------------------------------- |
| ECO one-hot (top 100)     | `ECO` column | Binary vector 100d | Top 100 covers >95% games                |
| EcoCategory one-hot (A-E) | `ECO[:1]`    | Binary vector 5d   | Quick proxy                              |
| First-10-ply sequence     | `Moves`      | See below          | Core feature                             |
| GameFormat one-hot        | `GameFormat` | 5d                 | Bullet/Blitz/Rapid/Classical/UltraBullet |
| BaseTime log              | `BaseTime`   | log1p(x)           | Skewed distribution                      |
| Increment                 | `Increment`  | log1p(x)           |                                          |

### Tier 2 — Important

| Feature                         | Source         | Encoding    |
| ------------------------------- | -------------- | ----------- |
| Move bigram TF-IDF (top 200)    | `Moves`        | Sparse 200d |
| Move trigram TF-IDF (top 100)   | `Moves`        | Sparse 100d |
| Opening diversity entropy       | First-10 moves | Scalar      |
| NumMoves (capped at 100)        | `NumMoves`     | Scalar      |
| First move (e4/d4/c4/Nf3/other) | `Moves`        | 5d one-hot  |

### Tier 3 — Nice to Have

| Feature                                | Source                | Encoding     |
| -------------------------------------- | --------------------- | ------------ |
| Move length sequence (per ply)         | `Moves` token lengths | Sequence 10d |
| Check frequency in first 15 ply        | `Moves`               | Scalar       |
| Castling in first 20 ply               | `Moves`               | Binary       |
| Pawn structure proxy (e/d/c4 openings) | `Moves`               | Categorical  |

## Move Sequence Encoding Options

### Option A: Hash-based N-gram (recommended for baseline)

- Chuẩn hóa SAN → lowercase, remove move numbers
- Tạo unigram/bigram/trigram counts → TF-IDF → LSA (SVD 50d)
- Pros: simple, fast, no training needed
- Cons: mất position thông tin

### Option B: Position one-hot (first-5-move combinations)

- Map mỗi combination (first 5 moves trắng) → unique ID → embedding
- Pros: capture exact opening lines
- Cons: vocabulary explosion (>1M unique sequences)

### Option C: Sequential model input (for deep learning phase)

- Token = individual SAN move
- Vocabulary size: ~3000 unique moves
- Input: sequence của 10-15 token IDs
- Dùng trong LSTM/Transformer phase (Giai đoạn 3b)

**Chọn Option A** cho Feature Engineering phase. Option C cho Model Training phase (LSTM).

## User Stories & Use Cases

### User Stories

- As a ML Engineer, tôi muốn sinh ra feature matrix từ first-N-moves để tăng chất lượng dự đoán `ModelBand` so với baseline tabular.
- As a Data Engineer, tôi muốn chạy feature extraction theo batch trên full 187M rows để tránh OOM và hoàn thành trong SLA.
- As a Data Scientist, tôi muốn có train/val temporal split cố định để so sánh experiment công bằng và reproducible.
- As a MLOps Engineer, tôi muốn lưu feature store chuẩn hóa để pipeline training không phải tính lại features mỗi lần chạy.
- As a ML Engineer, tôi muốn có quy tắc xử lý game thiếu moves (<5 ply) rõ ràng để không làm sai lệch mô hình.

### Critical Flows

1. Full batch feature extraction:
   Load Parquet theo batch -> transform tabular + move sequence -> concat features -> ghi Parquet feature store.
2. Reproducible temporal split:
   Gán file tháng 2025-12 vào train và 2026-01 vào validation -> khóa quy tắc split trong pipeline.
3. Re-baseline và kiểm leakage:
   Train XGBoost trên feature mới -> báo cáo accuracy -> xác nhận không dùng trường ELO làm input feature.
4. Missing-moves handling:
   Phát hiện game có <5 ply -> loại bỏ bản ghi theo policy -> log tỉ lệ affected rows.

## Success Criteria

1. ✅ Feature pipeline chạy trên full 187M rows trong < 4 giờ (batch 10M, Polars)
2. ✅ Feature matrix saved ra Parquet (`data/features/train_features.parquet`, `data/features/val_features.parquet`)
3. ✅ XGBoost re-baseline với new features: **acc ≥ 60%**
4. ✅ Feature dimensionality: 50-300 features total
5. ✅ No data leakage: features chỉ dùng first N moves, không dùng WhiteElo/BlackElo làm input
6. ✅ Reproducibility: chạy lại pipeline với cùng input cho cùng schema output và cùng quy tắc split thời gian
7. ✅ Evaluation reporting: bắt buộc báo cáo `accuracy` và `macro_f1` cho tập validation; pass/fail dựa trên accuracy

## Constraints & Assumptions

### Constraints

- **Không dùng EloAvg làm input feature** (data leakage — đây là target)
- **Không dùng WhiteRatingDiff/BlackRatingDiff** (chỉ biết sau khi có ELO → leakage)
- **Memory**: Feature matrix cho 3M rows × 300 cols ≈ 3.6 GB numpy float32 — ok
- **Target**: `ModelBand` (5 classes: Beginner/Intermediate/Advanced/Expert/Master)
- **SLA cứng**: full run trên 187M rows phải đạt < 4 giờ; không dùng sample run để thay thế tiêu chí nghiệm thu
- **Data quality gate**: tỉ lệ bản ghi bị loại do `<5 ply` phải <= 1.5% trên từng split (train/val)
- **Data quality warning**: nếu tỉ lệ loại nằm trong (1.5%, 3%], vẫn cho phép chạy nhưng phải báo cáo phân bố theo `GameFormat` và `ModelBand`
- **Data quality fail**: nếu tỉ lệ loại > 3%, run không đạt tiêu chí nghiệm thu

### Assumptions (đã xác nhận)

- Bài toán mục tiêu là **phân lớp trực tiếp ELO band** (`ModelBand`) từ first-N-moves, không phải dự đoán tăng/giảm ELO.
- SLA pipeline full dữ liệu được chốt là **< 4 giờ**.
- Tên output feature store chuẩn là `data/features/train_features.parquet` và `data/features/val_features.parquet`.
- Các nhóm người dùng chính gồm ML Engineer, Data Engineer, Data Scientist, và MLOps Engineer.

## Questions & Open Items

- Nếu chưa đạt `accuracy >= 60%`, lộ trình tối ưu bắt buộc theo thứ tự: **move n-gram/sequence -> ECO encoding -> game metadata**.
- Sau mỗi vòng tối ưu, có cần lock seed + lưu artifact (feature columns, vocab, SVD) để so sánh công bằng giữa các vòng không?
