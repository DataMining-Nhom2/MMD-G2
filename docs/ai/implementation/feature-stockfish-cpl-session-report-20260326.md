# Báo cáo Phiên làm việc — 26/03/2026

## Mục tiêu
Thay thế hoàn toàn phương pháp NLP (TF-IDF/SVD) bằng Stockfish Engine CPL để trích xuất features dự đoán ELO cờ vua theo 5 nhóm (ModelBand).

---

## Công việc đã hoàn thành

### Phase 1: Tạo mẫu 30.000 ván cờ
- Viết script `src/create_30k_sample.py` dùng **Reservoir Sampling** quét tuần tự 45GB Parquet gốc.
- Lọc bỏ ván chết đồng hồ (`Time forfeit`) và cờ chớp (`Bullet/Blitz`).
- Kết quả: **30.000 ván** (6.000/band), chỉ gồm Rapid (96%) + Classical (3.2%).
- Thời gian: **8.7 giây**.

### Phase 2: Stockfish Setup
- Build **Stockfish 18** từ source code (avx2) → `.tmp/stockfish_binary`.
- Cấu hình: `depth=10`, `threads=1`, `hash=128MB`.
- Test thành công trên ván Scholar's Mate → `avg_cpl=37.2`, `inaccuracy_count=2`.

### Phase 3: StockfishTransformer
- Viết lại hoàn toàn `src/feature_engineering.py`:
  - **Xóa sạch**: `MoveTransformer`, `TfidfVectorizer`, `TruncatedSVD` và mọi import liên quan.
  - **Thêm mới**: `StockfishTransformer` trích xuất 6 features/ván:

    | Feature | Ý nghĩa |
    |---------|---------|
    | `avg_cpl` | CPL trung bình toàn ván |
    | `blunder_count` | Số nước mất >300cp |
    | `mistake_count` | Số nước mất 100-300cp |
    | `inaccuracy_count` | Số nước mất 50-100cp |
    | `max_cpl` | Nước đi tệ nhất |
    | `cpl_std` | Độ dao động CPL |

  - **Tinh gọn** `TabularTransformer`: Bỏ `GameFormat/BaseTime/Increment` khỏi features (vẫn giữ trong data).

### Phase 4: Tích hợp Pipeline
- Viết lại `src/feature_config.py`: Xóa TF-IDF config, thêm Stockfish config.
- **Tối ưu đa luồng**: Chuyển từ single-thread (1.5 giờ) sang `ProcessPoolExecutor` 18 luồng → **20 phút** xử lý 30.000 ván.
- Output: `data/features/sample_30k_features.parquet` — **30.000 dòng × 113 cột**.

### Phase 5: Baseline XGBoost

> [!IMPORTANT]
> Kết quả Stratified 5-Fold CV:

| Mô hình | Accuracy | Macro F1 |
|---------|----------|----------|
| Tabular Only (ECO + NumMoves) | 34.38% ±0.35% | 31.84% |
| **Tabular + Stockfish CPL** | **44.24% ±0.24%** | **42.95%** |
| Baseline cũ (TF-IDF/SVD) | 44.18% | N/A |

- CPL features đóng góp **+10 điểm phần trăm** so với Tabular-only.
- Tương đương baseline NLP cũ nhưng chỉ với **6 cột engine** thay vì hàng nghìn cột TF-IDF.

---

## Phát hiện quan trọng

### Nghẽn cổ chai ở mốc 44%
Phân tích bài báo [arXiv:2409.11506](https://arxiv.org/abs/2409.11506) *(Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM, T9/2024)* đã giải mã nguyên nhân:

> [!WARNING]
> Việc loại bỏ hoàn toàn biến thời gian (BaseTime, Increment) khỏi features khiến model mất đi 1 mảnh ghép cốt lõi.

- Bài báo chứng minh: **Clock Times giảm 24% sai số** dự đoán ELO (34% cho Bullet).
- Cách quản lý thời gian phản ánh trình độ không kém gì cách đi cờ.
- Mục tiêu >55% Accuracy không đạt được là hệ quả trực tiếp của Design Decision 4 (Zero Time Context).

---

## Files đã tạo/sửa

| File | Trạng thái | Mô tả |
|------|-----------|-------|
| [create_30k_sample.py](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/src/create_30k_sample.py) | MỚI | Reservoir sampling + lọc |
| [feature_engineering.py](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/src/feature_engineering.py) | VIẾT LẠI | StockfishTransformer thay MoveTransformer |
| [feature_config.py](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/src/feature_config.py) | VIẾT LẠI | Stockfish config thay TF-IDF config |
| [run_pipeline.py](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/src/run_pipeline.py) | MỚI | Pipeline runner (đa luồng) |
| [eval_xgboost.py](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/src/eval_xgboost.py) | MỚI | XGBoost evaluation + Notebook generator |
| [stockfish-baseline.ipynb](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/notebooks/stockfish-baseline.ipynb) | MỚI | Notebook đánh giá (Confusion Matrix, Feature Importance) |
| [report-stockfish-cpl.md](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/docs/ai/implementation/report-stockfish-cpl.md) | MỚI | Implementation Report theo chuẩn AI DevKit |
| [Planning doc](file:///home/sakana/Code/PTIT/MMDs/MMD-G2/docs/ai/planning/feature-stockfish-cpl-elo-prediction.md) | CẬP NHẬT | Đánh dấu hoàn thành 21/21 tasks |

---

## Hướng nghiên cứu tiếp theo

Hai lựa chọn chiến lược:

1. **Quick Win (XGBoost + Time Features):** Trả lại `BaseTime`, `Increment` vào TabularTransformer. Dự kiến phá mốc 55% ngay lập tức.
2. **Deep Learning (CNN-LSTM trên Colab Pro):** Theo hướng Paper arXiv:2409.11506 — extract Clock Time từng nước đi `[%clk]`, dùng CNN 8×8×12 + Bi-LSTM. Cần GPU A100 và viết lại PGN Parser.

---

## Thời gian thực thi

| Bước | Thời gian |
|------|-----------|
| Sampling 30k ván | 8.7 giây |
| Stockfish CPL (đa luồng 18 cores) | 20 phút |
| XGBoost 5-Fold CV | ~10 giây |
| **Tổng phiên làm việc** | **~1.5 giờ** |
