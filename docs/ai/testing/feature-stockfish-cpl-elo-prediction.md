---
phase: testing
title: Testing — Stockfish CPL Feature Engineering cho dự đoán ELO
description: Chiến lược kiểm thử cho pipeline trích xuất features từ Stockfish engine
---

# Testing — Stockfish CPL Feature Engineering

## Test Coverage Goals

- Unit test cho `StockfishTransformer`: analyze_game(), transform()
- Unit test cho `create_30k_sample.py`: phân phối mẫu đúng
- Integration test: full pipeline từ sample → features parquet
- Tất cả test phải chạy pass trước khi chạy batch 30k thật

## Unit Tests

### StockfishTransformer
- [ ] Test CPL tính đúng cho ván Scholar's Mate (4 nước, có blunder rõ ràng)
- [ ] Test ván draw ngắn (5 nước, CPL thấp cho cả hai bên)
- [ ] Test ván có castling + promotion (edge case SAN)
- [ ] Test ván rỗng / null Moves → trả về giá trị mặc định (0.0)
- [ ] Test ván bị timeout (ít nước, kết quả do hết giờ) → CPL chỉ tính trên nước đã đánh

### Sampling Script
- [ ] Test output có đúng 30.000 rows
- [ ] Test phân phối: mỗi ModelBand có đúng 6.000 rows
- [ ] Test schema output khớp schema input (đủ cột)
- [ ] Test seed reproducibility: chạy 2 lần ra kết quả giống nhau

### TabularTransformer (regression — giữ nguyên)
- [ ] Test ECO one-hot columns ổn định
- [ ] Test GameFormat one-hot columns ổn định
- [ ] Test numeric features (basetime_log, num_moves_norm) range hợp lý

## Integration Tests

- [ ] Full pipeline: Load sample_30k → TabularTransformer + StockfishTransformer → Concat → Save parquet
- [ ] Schema consistency: Kiểm tra cột output khớp với feature_columns.json
- [ ] Không còn bất kỳ tham chiếu nào đến TF-IDF / SVD / MoveTransformer trong codebase

## Test Data

- **Fixture nhỏ**: 5-10 ván cờ cố định dưới dạng SAN string (hardcoded trong test file)
- **Scholar's Mate**: `"1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#"` — biết trước nước 4...Nf6 là blunder
- **Sample parquet nhỏ**: Tạo file 50-100 rows cho integration test (không cần Stockfish thật — mock engine)

## Manual Testing

- [ ] Chạy `python src/create_30k_sample.py` và kiểm tra file output tồn tại
- [ ] Chạy phân tích Stockfish trên 10 ván đầu tiên, kiểm tra CPL output có ý nghĩa
- [ ] So sánh CPL trung bình của band Beginner vs Master — kỳ vọng Beginner CPL >> Master CPL
- [ ] Xem Feature Importance plot — kỳ vọng `avg_cpl` nằm top-3

## Performance Testing

- [ ] Benchmark Stockfish: Bao nhiêu giây/ván ở depth=12? Nhân ước tính lên 30k.
- [ ] Monitor RAM usage khi chạy batch 30k (kỳ vọng < 4GB)
- [ ] Kiểm tra disk space output: sample_30k_features.parquet kỳ vọng < 50MB
