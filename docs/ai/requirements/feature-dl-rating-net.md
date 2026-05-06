---
phase: requirements
title: "DL Rating Net — Dự đoán ELO bằng CNN + Bi-LSTM"
description: >
  Xây dựng mô hình Deep Learning dự đoán ELO kỳ thủ dựa trên paper arXiv:2409.11506.
  Mở rộng bằng cách bổ sung CPL & Blunder sequence từ Stockfish.
date: 2026-04-27
---

# DL Rating Net — Yêu cầu

## Problem Statement

**Bài toán:** Dự đoán chính xác rating ELO của kỳ thủ dựa trên diễn biến ván cờ.

- **Hạn chế V1-V3:** Aggregate features (avg_cpl, blunder_rate...) đã vứt bỏ tính tuần tự, MAE tốt nhất chỉ đạt 247.8 ELO.
- **Hạn chế V4 cũ:** Bỏ qua CNN (Board State), chỉ dùng CPL sequence + XGBoost — sai hướng so với paper.
- **Paper arXiv:2409.11506** đạt MAE = 182 ELO bằng CNN-LSTM + clock times.

## Goals & Objectives

### Primary Goals
1. Triển khai mô hình RatingNet (CNN + Bi-LSTM) theo đúng kiến trúc paper
2. Bổ sung thêm CPL sequence + Blunder flag vào đầu vào model (cải tiến so với paper)
3. Đạt MAE ≤ 220 ELO trên sample 30k ván

### Secondary Goals
4. Xây dựng pipeline dữ liệu có thể scale lên 300k-1M ván
5. So sánh kết quả có/không CPL để đánh giá giá trị bổ sung

### Non-goals
- Chưa cần deploy model lên production
- Chưa cần real-time inference
- Chưa cần train trên toàn bộ dataset (1M+)

## User Stories

1. **Nhóm trưởng** muốn xem kết quả dự đoán ELO theo phương pháp DL chuẩn paper để đánh giá tiềm năng của hướng tiếp cận.
2. **Nhóm phát triển** cần pipeline data rõ ràng: PGN → Board State + Clock + CPL → Tensor → Model.
3. **Nghiên cứu** cần so sánh trực tiếp: Model gốc (CNN + Clock) vs Model cải tiến (CNN + Clock + CPL).

## Success Criteria

| Tiêu chí | Mục tiêu |
|---|---|
| MAE (Mean Absolute Error) | ≤ 220 ELO |
| Pipeline chạy thành công | 30k ván, đầy đủ Board + Clock + CPL |
| Training hoàn tất | Convergence trên 30k sample |
| So sánh baseline | Bảng so sánh V1/V2/V3 vs DL RatingNet |

## Constraints & Assumptions

### Constraints
- **Dữ liệu gốc (PGN.zst):** Đã bị xóa khỏi máy. Cần download lại (~30GB) để giữ `%clk`. File parquet hiện tại (2×23GB, ~187M ván) đã bị strip toàn bộ clock bởi `preprocessing.py` dòng 224 (`variation_san()`).
- **GPU:** Cần GPU để train LSTM hiệu quả (CPU quá chậm cho 30k ván × 100 moves).
- **Stockfish:** Cần engine Stockfish v16+ để tính CPL per-move.

### Assumptions
- 2 file parquet lớn (`lichess_2025-12_ml.parquet`, `lichess_2026-01_ml.parquet`) có đủ data cho 30k sample.
- Lichess PGN gốc (`pgn.zst`) vẫn available để re-extract clock times.
- Conda env `MMDS` đã có PyTorch + CUDA.

## Questions & Open Items

1. ~~Có bỏ filter thể thức không?~~ → **Đã quyết định: BỎ filter, giữ tất cả thể thức**
2. ~~Có bỏ ván thua do chết đồng hồ?~~ → **Đã quyết định: GIỮ, như paper gốc**
3. ~~Clock data từ đâu?~~ → **PGN.zst đã bị xóa.** Chiến lược: Train trước không clock (Track A), download PGN.zst để bổ sung sau (Track B)
