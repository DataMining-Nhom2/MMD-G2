---
phase: implementation
title: Implementation Report - Stockfish CPL Feature Engineering
description: Technical implementation notes, patterns, and performance reports for the Stockfish CPL engine pipeline
---

# Implementation Guide

## Development Setup
**How do we get started?**

- **Prerequisites:** Python 3.11+, `conda` environment (MMDS).
- **External Binaries:** Stockfish 18 đã được build từ Source code (hỗ trợ tập lệnh `avx2`) và đặt tại `.tmp/stockfish_binary`. Cần cấp quyền thực thi `chmod +x` trước khi gọi qua `chess.engine`.
- **Dependencies:** `polars` (để Lazy Scan 45GB parquet), `python-chess` (Parse chuỗi SAN), `xgboost` và `scikit-learn` (Huấn luyện Baseline).

## Code Structure
**How is the code organized?**

- **`src/create_30k_sample.py`**: Script thực hiện Reservoir Sampling để trích xuất 30.000 ván cờ cân bằng (6.000 ván/band), đồng thời lọc sạch các ván hỏng (Time forfeit) và cờ chớp tĩnh (Bullet/Blitz).
- **`src/feature_config.py`**: Chứa toàn bộ hằng số cấu hình Stockfish (Depth=10, Threads=1, Hash=128MB) và danh sách Input/Output columns.
- **`src/feature_engineering.py`**: Module cốt lõi. Chứa `TabularTransformer` (Xử lý mã ECO) và `StockfishTransformer` (Tính CPL/Blunders/Mistakes bằng Engine).
- **`src/run_pipeline.py`**: Script kết nối đầu cuối, thực thi đa luồng và xuất dữ liệu `sample_30k_features.parquet`.
- **`src/eval_xgboost.py`**: Script huấn luyện XGBoost và tự động Generate ra Jupyter Notebook cho Task đánh giá số liệu.

## Implementation Notes
**Key technical details to remember:**

### Core Features
- **Stockfish CPL Extraction**: Thiết kế class stateless `StockfishTransformer`. Cứ mỗi ván cờ, nó duyệt mảng SAN, cập nhật Board State ảo và gọi Engine tính `score` Centipawns trước/sau nước đi. Xuất ra 6 cột: `avg_cpl, blunder_count, mistake_count, inaccuracy_count, max_cpl, cpl_std`.
- **Zero Time Context**: Chủ đích bỏ cột `BaseTime` và `Increment` ở bước `TabularTransformer` để gượng ép Mô hình học phân loại ELO 100% bằng CPL Chiến thuật. (Hành động này sau đó được chứng minh là đi ngược lại ranh giới thời gian tự nhiên, tạo ra nghẽn cổ chai Accuracy ở mốc 44.24%).

### Patterns & Best Practices
- **Design Patterns**: Sử dụng mô hình `Transformer` (fit/transform) chuẩn của Scikit-learn để tổ chức luồng Data.
- **Multiprocessing**: Áp dụng Python `ProcessPoolExecutor` chia nhỏ 30.000 ván thành các Chunk (100 ván/chunk) thông qua hàm Map ngoài Module. Việc mở 18 tiến trình Engine đã tận dụng kịch trần toàn bộ lõi CPU đa luồng.

## Integration Points
**How do pieces connect?**

- Cấu trúc file Parquet trung gian biến thành trạm Tích hợp:
  1. Parquet 45GB Gốc -> (Lọc Sampling) -> `sample_30k.parquet`
  2. `sample_30k.parquet` -> (Stockfish Engine Multiprocessing) -> `sample_30k_features.parquet`
  3. `sample_30k_features.parquet` -> (XGBoost) -> Baseline Accuracy.

## Error Handling
**How do we handle failures?**

- **SAN Parsing Error**: Ván cờ có nước đi lỗi hoặc không tuân chuẩn `python-chess` sẽ bị hàm `try/except` bắt tín hiệu. Ván hỏng sẽ trả về CPL = 0.0 mặc định thay vì làm sụp (crash) toàn bộ Multiprocessing Chunk.
- **Engine Crash**: Mọi kết nối Pop Uci Engine đều được bao bọc trong block `finally: engine.quit()` để đảm bảo không bị rò rỉ (leak) RAM của Zombie Processes trên Hệ điều hành.

## Performance Considerations
**How do we keep it fast?**

- **Optimization strategies**: Áp dụng Reservoir Sampling bằng thư viện `pyarrow` Iter Batches (1 Triệu Rows/Lần). Nhờ đó không cần nạp 45GB file gốc vào RAM (Chỉ tốn vài trăm MB RAM trong 8 giây quét).
- **Engine Tiers**: Giảm `Depth` của Stockfish xuống mốc an toàn là `10`. Vừa đủ để bắt đứt các lỗi Blunders phổ thông mà chỉ mất vỏn vẹn `5 mili-giây / lệnh phân tích`.
- **Parallel Computing**: Rút ngắn thời gian Extract Feature 30.000 ván từ 1.5 Tiếng xuống còn đúng **20 phút** nhờ kỹ năng bóp xung 18 luồng CPU.

## Security Notes
**What security measures are in place?**

- Binary Stockfish tải từ Github Release chính thức và Build tại Local. Không khởi chạy lệnh Shell ẩn, tránh nguy cơ ACE (Arbitrary Code Execution).
- Đường dẫn tới Engine/Data được quản lý bằng `pathlib` bảo mật Absolute Path, không lộ mã nguồn File System.
