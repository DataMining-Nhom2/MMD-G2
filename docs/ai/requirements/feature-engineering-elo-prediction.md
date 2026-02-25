---
phase: requirements
title: Requirements — Feature Engineering cho dự đoán ELO realtime
description: Xây dựng feature set từ dữ liệu cờ vua thô để phục vụ mô hình ML dự đoán ELO
---

# Requirements — Feature Engineering cho Dự đoán ELO Realtime

## Problem Statement
**Pha EDA đã xác nhận**: first-N-moves có đủ signal để dự đoán ELO (5-class accuracy ~44% với chỉ tabular features). Pha Feature Engineering cần **trích xuất và encode features hiệu quả** từ dữ liệu thô thành input vector cho mô hình ML.

EDA baseline (XGBoost tabular, 18 features): **44.18% acc** — cần tăng lên **≥ 60%** với feature engineering tốt.

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
| Feature | Source | Encoding | Notes |
|---------|--------|----------|-------|
| ECO one-hot (top 100) | `ECO` column | Binary vector 100d | Top 100 covers >95% games |
| EcoCategory one-hot (A-E) | `ECO[:1]` | Binary vector 5d | Quick proxy |
| First-10-ply sequence | `Moves` | See below | Core feature |
| GameFormat one-hot | `GameFormat` | 5d | Bullet/Blitz/Rapid/Classical/UltraBullet |
| BaseTime log | `BaseTime` | log1p(x) | Skewed distribution |
| Increment | `Increment` | log1p(x) | |

### Tier 2 — Important
| Feature | Source | Encoding |
|---------|--------|----------|
| Move bigram TF-IDF (top 200) | `Moves` | Sparse 200d |
| Move trigram TF-IDF (top 100) | `Moves` | Sparse 100d |
| Opening diversity entropy | First-10 moves | Scalar |
| NumMoves (capped at 100) | `NumMoves` | Scalar |
| First move (e4/d4/c4/Nf3/other) | `Moves` | 5d one-hot |

### Tier 3 — Nice to Have
| Feature | Source | Encoding |
|---------|--------|----------|
| Move length sequence (per ply) | `Moves` token lengths | Sequence 10d |
| Check frequency in first 15 ply | `Moves` | Scalar |
| Castling in first 20 ply | `Moves` | Binary |
| Pawn structure proxy (e/d/c4 openings) | `Moves` | Categorical |

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

## Success Criteria
1. ✅ Feature pipeline chạy trên full 187M rows trong < 4 giờ (batch 10M, Polars)
2. ✅ Feature matrix saved ra Parquet (`data/features/train.parquet`, `val.parquet`)
3. ✅ XGBoost re-baseline với new features: **acc ≥ 55%** (target: 60%)
4. ✅ Feature dimensionality: 50-300 features total
5. ✅ No data leakage: features chỉ dùng first N moves, không dùng WhiteElo/BlackElo làm input

## Constraints
- **Không dùng EloAvg làm input feature** (data leakage — đây là target)
- **Không dùng WhiteRatingDiff/BlackRatingDiff** (chỉ biết sau khi có ELO → leakage)
- **Memory**: Feature matrix cho 3M rows × 300 cols ≈ 3.6 GB numpy float32 — ok
- **Target**: `ModelBand` (5 classes: Beginner/Intermediate/Advanced/Expert/Master)
