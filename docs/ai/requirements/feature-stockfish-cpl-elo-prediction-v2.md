---
phase: requirements
title: Requirements V2 — Nâng cấp Stockfish Feature Engineering (11 Features chuyên sâu)
description: Mở rộng StockfishTransformer từ 6 features thô sang 11 features phân tích đa chiều (Giai đoạn, WDL, PV Match)
---

# Requirements V2 — Nâng cấp Stockfish Feature Engineering

## Problem Statement
**Bài toán chúng ta đang giải quyết là gì?**

- **Vấn đề cốt lõi**: Phiên bản V1 chỉ trích xuất 6 features CPL thô (avg_cpl, blunder_count, mistake_count, inaccuracy_count, max_cpl, cpl_std). Kết quả XGBoost Baseline dừng ở **44.24% Accuracy** (5-class). Nguyên nhân: CPL thô không phân biệt được "Học vẹt khai cuộc nhưng mù tàn cuộc" vs "Yếu toàn diện", cũng không đo được tác động thực sự của sai lầm (mất 3 centipawns vs mất 30% cơ hội thắng).
- **Giải pháp V2**: Mở rộng StockfishTransformer sang **11 features chuyên sâu** chia thành 3 nhóm: (A) Tỷ lệ hóa toàn ván, (B) Phân tích theo giai đoạn ván cờ (Opening/Midgame/Endgame), (C) Xác suất WDL & độ khớp máy.
- **Dọn dẹp Leakage**: Loại bỏ triệt để mọi cột có nguy cơ rò rỉ thông tin ELO (WhiteElo, BlackElo, Result, RatingDiff, BaseTime, Increment, GameFormat, Termination).

## Goals & Objectives

### Mục tiêu chính (Primary Goals)
1. **Nâng cấp `StockfishTransformer`** phiên bản V2: từ 6 features → **11 features** phân tích đa chiều
2. **Tỷ lệ hóa** blunder/mistake/inaccuracy thành rate (% trên tổng số nước) thay vì đếm thô
3. **Phân mảnh CPL theo giai đoạn**: Opening (nước 1-10), Midgame (nước 11-30), Endgame (nước 31+)
4. **Tích hợp WDL** (Win/Draw/Loss probability): Đo lường mức ném đi xác suất thắng
5. **Tính Best Move Match Rate**: Đo tỷ lệ đi trùng nước tối ưu của Stockfish
6. **Phá mốc >50% Accuracy** trên bài toán phân loại 5 nhóm ELO

### Non-goals (Ngoài phạm vi)
- **KHÔNG** thay đổi mẫu dữ liệu 30k (giữ nguyên `sample_30k.parquet` đã tạo)
- **KHÔNG** thay đổi TabularTransformer (ECO + NumMoves giữ nguyên)
- **KHÔNG** thêm lại các cột thời gian (BaseTime, Increment, GameFormat) — giữ triết lý Zero Time Context
- **KHÔNG** triển khai Deep Learning trong phase này

## Success Criteria

1. ✅ StockfishTransformer V2 xuất đủ **11 features** cho mỗi ván cờ
2. ✅ Các features blunder/mistake/inaccuracy đã chuyển sang **tỷ lệ %** (rate)
3. ✅ Có 3 cột CPL phân giai đoạn: `opening_cpl`, `midgame_cpl`, `endgame_cpl`
4. ✅ Có cột WDL: `avg_wdl_loss`, `max_wdl_loss`
5. ✅ Có cột PV Match: `best_move_match_rate`
6. ✅ Pipeline V2 chạy hoàn tất trên 30k ván trong **<30 phút** (đa luồng)
7. ✅ XGBoost Accuracy **>50%** (tăng >6% so với V1 baseline 44.24%)

## Constraints & Assumptions

### Technical Constraints
- **Stockfish 18** đã build sẵn tại `.tmp/stockfish_binary` (hỗ trợ NNUE + WDL output)
- **WDL Output**: Cần Stockfish 16+ với NNUE enabled để phun `wdl()` probability. Stockfish 18 đã hỗ trợ sẵn.
- **CPU**: i5-14600KF (20 threads) — đã chứng minh chạy 18 luồng song song ổn định ở V1
- **RAM**: 31GB — đủ cho 30k sample in-memory
- **Depth**: Giữ nguyên `depth=10` để tốc độ ổn định

### Assumptions
1. Stockfish 18 NNUE phun ra WDL probability đủ chính xác ở depth=10
2. Phân chia giai đoạn cứng (10/30 nước) là xấp xỉ hợp lý cho đa số ván cờ
3. NaN cho ván ngắn (<11 nước không có midgame, <31 nước không có endgame) — XGBoost xử lý tốt NaN natively

## Questions & Open Items

1. ~~Stockfish binary~~ → Đã giải quyết (Build sẵn Stockfish 18 từ source)
2. ~~Multi-processing~~ → Đã giải quyết (ProcessPoolExecutor 18 luồng, 20 phút/30k ván)
3. **WDL normalization**: WDL trả về (win, draw, loss) tổng = 1000. Nên dùng raw hay normalize về [0, 1]?
4. **Best Move Match**: So sánh move thực tế vs PV[0] của engine — có nên dùng `multipv=1` hay `multipv=2` để lấy top-2?
