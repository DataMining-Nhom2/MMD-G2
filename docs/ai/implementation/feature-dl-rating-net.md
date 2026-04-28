---
phase: implementation
title: "DL Rating Net — Implementation Notes"
description: >
  Ghi chú triển khai chi tiết cho pipeline DL Rating Net.
date: 2026-04-27
---

# DL Rating Net — Implementation Notes

## Development Setup

### Prerequisites
- Conda environment `MMDS` đã activated
- PyTorch + CUDA (kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`)
- Stockfish v16+ (kiểm tra: `stockfish --help`)
- python-chess: `pip install python-chess`

### Cấu trúc code mới

```
src/
├── config.py                    # Đường dẫn trung tâm
├── data/
│   ├── create_30k_sample.py     # V1 (cũ, giữ reference)
│   ├── create_30k_sample_v2.py  # [MỚI] Sampling không filter
│   ├── preprocessing.py         # V1 (cũ, strip %clk)
│   ├── board_encoder.py         # [MỚI] Board → 12×8×8 tensor
│   └── chess_dataset.py         # [MỚI] PyTorch Dataset
├── features/
│   ├── extract_sequences.py     # CPL extraction (tái sử dụng)
│   └── feature_engineering.py   # V2 aggregate (giữ reference)
├── models/
│   ├── rating_net.py            # [MỚI] CNN + Bi-LSTM model
│   └── train.py                 # [MỚI] Training loop
├── baselines/
│   ├── paperbaseline.py         # Reference code từ paper
│   └── eval_xgboost_v3.py      # XGBoost baseline (reference)
```

## Implementation Notes

### Board Encoding (12 planes)
Theo paper, mỗi vị trí bàn cờ được mã hóa thành 12 binary planes:
- Plane 0-5: Quân trắng (P, N, B, R, Q, K)
- Plane 6-11: Quân đen (p, n, b, r, q, k)
- Giá trị: 1.0 tại ô có quân, 0.0 tại ô trống

```python
# Ví dụ encoding
PIECE_TO_PLANE = {
    chess.PAWN:   0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK:   3, chess.QUEEN:  4, chess.KING:   5,
}
# Quân trắng: plane_idx = PIECE_TO_PLANE[piece_type]
# Quân đen:   plane_idx = PIECE_TO_PLANE[piece_type] + 6
```

### CPL + Blunder Integration
Tại mỗi time-step t, LSTM nhận vector concat:
```
input_t = [cnn_output_t (128-dim), clock_t (1), cpl_t (1), blunder_t (1)]
# Total: 131 dimensions
```

### Normalization
- **ELO:** `(elo - mean) / std` — Paper dùng mean=1514, std=366
- **Clock:** `(seconds - mean) / std` — Paper dùng mean=273, std=380
- **CPL:** Cần tính mean/std từ dataset của chúng ta
- **Blunder:** Binary (0/1), không cần normalize
