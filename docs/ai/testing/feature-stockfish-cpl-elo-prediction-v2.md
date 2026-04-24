---
phase: testing
title: Testing V2 — Nâng cấp Stockfish Feature Engineering (11 Features)
description: Chiến lược kiểm thử cho StockfishTransformer V2 (WDL, Phase CPL, PV Match)
---

# Testing V2 — Nâng cấp Stockfish Feature Engineering

## Test Coverage Goals

- Unit test cho `StockfishTransformer V2`: tất cả 11 features output
- Unit test riêng cho từng nhóm A, B, C
- Integration test: full pipeline V2 từ sample → features_v2 parquet
- Regression test: đảm bảo avg_cpl và cpl_std V2 = V1 (không bị thay đổi logic)

## Unit Tests

### Nhóm A: Tỷ lệ hóa
- [ ] Test `blunder_rate` trên ván có 2 blunders / 40 nước = 0.05
- [ ] Test `mistake_rate` trên ván có 5 mistakes / 20 nước = 0.25
- [ ] Test `inaccuracy_rate` trên ván rỗng (0 nước) → trả về 0.0
- [ ] Test `avg_cpl` và `cpl_std` khớp kết quả V1 (regression test)

### Nhóm B: Phân giai đoạn CPL
- [ ] Test Scholar's Mate (4 nước): `opening_cpl` có giá trị, `midgame_cpl = NaN`, `endgame_cpl = NaN`
- [ ] Test ván 25 nước: `opening_cpl` (nước 1-10), `midgame_cpl` (nước 11-25), `endgame_cpl = NaN`
- [ ] Test ván dài 50 nước: cả 3 giai đoạn đều có giá trị
- [ ] Test ván rỗng: tất cả 3 cột = NaN hoặc 0.0

### Nhóm C: WDL & PV Match
- [ ] Test `avg_wdl_loss` ≥ 0 (không âm — mất xác suất thắng luôn dương hoặc 0)
- [ ] Test `max_wdl_loss` ≥ `avg_wdl_loss` (max luôn ≥ trung bình)
- [ ] Test `best_move_match_rate` nằm trong [0, 1]
- [ ] Test ván Scholar's Mate: kỳ vọng `best_move_match_rate` < 1.0 (vì Nf6 không phải best move)
- [ ] Test WDL output được normalize về [0, 1] (không còn scale 0-1000)

### Edge Cases
- [ ] Test ván 0 nước (Moves rỗng) → tất cả features = 0.0 hoặc NaN
- [ ] Test ván chỉ có 1 nước → opening_cpl = CPL đó, midgame = NaN, endgame = NaN
- [ ] Test SAN lỗi (ký tự bất hợp pháp) → try/except bắt, trả giá trị mặc định

## Integration Tests

- [ ] Full pipeline V2: Load sample_30k → TabularTransformer + StockfishTransformer V2 → Concat → Save parquet
- [ ] Schema V2: Kiểm tra output có đủ 117 cột (106 tabular + 11 engine)
- [ ] Regression: `avg_cpl` column trong V2 xấp xỉ bằng V1 (cho phép sai nhỏ do floating point)
- [ ] NaN distribution: Kiểm tra `midgame_cpl` NaN khoảng 5-10% (ván ngắn), `endgame_cpl` NaN khoảng 20-30%

## Test Data

- **Scholar's Mate**: `"1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#"` — 4 nước, chỉ có opening
- **Ván trung bình 25 nước**: Hardcode 1 ván SAN cố định có opening + midgame
- **Ván dài 50 nước**: Hardcode 1 ván SAN cố định có cả 3 giai đoạn
- **Ván hỏng**: `""` (empty string) và `"invalid_move xyz"` (SAN lỗi)

## Performance Testing

- [ ] Benchmark V2 vs V1: Đo thời gian/ván. Kỳ vọng V2 chậm hơn V1 không quá 25%
- [ ] Pipeline V2 trên 30k ván: kỳ vọng hoàn thành trong <30 phút (18 luồng)
- [ ] RAM usage: kỳ vọng <4GB (thêm WDL data không đáng kể)

## Manual Testing

- [ ] So sánh `opening_cpl` Beginner vs Master — kỳ vọng Beginner >> Master
- [ ] So sánh `best_move_match_rate` Beginner vs Master — kỳ vọng Master >> Beginner
- [ ] Xem Feature Importance plot V2 — kỳ vọng `avg_wdl_loss` hoặc `best_move_match_rate` nằm top-5
