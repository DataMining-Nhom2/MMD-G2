---
phase: requirements
title: Requirements — Dự đoán ELO bằng Stockfish CPL (Full-game Analysis)
description: Phân tích chất lượng nước đi toàn ván bằng Stockfish engine để dự đoán nhóm trình độ ELO
---

# Requirements — Dự đoán ELO bằng Stockfish CPL

## Problem Statement
**Bài toán chúng ta đang giải quyết là gì?**

- **Vấn đề cốt lõi**: Xây dựng mô hình ML phân loại ELO rating người chơi cờ vua dựa trên **chất lượng nước đi trong toàn bộ ván cờ**, sử dụng engine Stockfish để đo lường sai số (Centipawn Loss — CPL).
- **Thay đổi so với hướng cũ**: Phiên bản trước cố gắng dự đoán ELO từ **vài nước đi đầu tiên** sử dụng TF-IDF/SVD trên chuỗi SAN. Hướng mới loại bỏ hoàn toàn NLP approach, thay bằng **đánh giá engine trực tiếp** — phương pháp được cộng đồng cờ vua công nhận là thước đo chính xác nhất phản ánh kỹ năng thực sự.
- **Ai sử dụng kết quả**: Nền tảng cờ vua (Lichess, Chess.com) dùng để phát hiện smurf, cải thiện matchmaking; giảng viên cờ dùng để đánh giá trình độ học viên.
- **Tại sao CPL?**: Centipawn Loss đo khoảng cách giữa nước đi thực tế và nước đi tối ưu theo engine. Người chơi ELO thấp mắc nhiều lỗi nặng (Blunders), người chơi ELO cao đánh gần tối ưu hơn → CPL trung bình trực tiếp tương quan với trình độ.

## Goals & Objectives

### Mục tiêu chính (Primary Goals)
1. **Trích xuất mẫu cân bằng 30.000 ván** từ dataset 187M — phân bổ đều 6.000 ván/nhóm ELO (5 nhóm)
2. **Tính toán Stockfish CPL** cho toàn bộ mỗi ván trong mẫu — sử dụng `python-chess` + engine binary
3. **Xây dựng bộ features mới** kết hợp: (a) Engine features (CPL trung bình, số Blunders/Mistakes/Inaccuracies) + (b) Tabular features đã có (ECO, GameFormat, BaseTime, NumMoves)
4. **Train XGBoost baseline** trên bộ features mới, đánh giá Accuracy/F1 trên Stratified K-Fold

### Mục tiêu phụ (Secondary Goals)
5. So sánh hiệu quả Engine features vs Tabular-only để đo mức cải thiện
6. Phân tích Feature Importance — xác nhận CPL có phải feature #1 không

### Non-goals (Ngoài phạm vi)
- **KHÔNG** chạy Stockfish trên toàn bộ 187M ván (chỉ chạy trên sample 30k)
- **KHÔNG** deploy model hay xây API trong phase này
- **KHÔNG** dùng TF-IDF, SVD, Bag-of-Words, hay bất kỳ NLP approach nào trên chuỗi nước đi
- **KHÔNG** tối ưu hyperparameter — chỉ cần baseline đủ tốt để chứng minh tính khả thi

## User Stories & Use Cases

1. **Là một Data Scientist**, tôi muốn **có bộ features dựa trên engine evaluation** để **phân loại ELO chính xác hơn hẳn so với NLP approach**, vì CPL phản ánh trực tiếp kỹ năng chiến thuật
2. **Là một ML Engineer**, tôi muốn **chạy pipeline trên sample 30k ván** để **kiểm chứng nhanh ý tưởng** trước khi đầu tư thời gian chạy full dataset
3. **Là một stakeholder**, tôi muốn thấy **Accuracy tăng đáng kể** (mục tiêu >60%, từ baseline 44%) để **justify việc đầu tư thêm tài nguyên tính toán** cho Stockfish analysis

### Edge Cases
- Ván cờ kết thúc bằng timeout/forfeit (Stockfish không có lỗi chiến thuật → CPL thấp giả tạo)
- Ván cờ với computer-assisted play (CPL cực thấp bất thường)
- Ván < 10 nước đi (quá ít dữ liệu để CPL có ý nghĩa thống kê)

## Success Criteria

1. ✅ Script tạo sample 30k chạy đúng: 6.000 ván/band, tổng 30.000 ván
2. ✅ Stockfish analysis hoàn tất trong **< 6 giờ** trên i5-14600KF
3. ✅ Mỗi ván có ≥4 features mới: `avg_cpl`, `blunder_count`, `mistake_count`, `inaccuracy_count`
4. ✅ XGBoost Accuracy trên 5-class classification **> 55%** (tăng >10% so với baseline 44%)
5. ✅ Feature importance cho thấy `avg_cpl` nằm trong top-3 features

## Constraints & Assumptions

### Technical Constraints
- **CPU**: Intel i5-14600KF (20 threads) — chạy Stockfish depth 10-12 để cân đối tốc độ/chất lượng
- **GPU**: RTX 3060 12GB — dùng cho XGBoost GPU training
- **RAM**: 31GB — đủ cho 30k sample in-memory
- **Stockfish binary**: Cần cài đặt sẵn hoặc tải binary Linux x86_64
- **Thời gian phân tích ước tính**: 30.000 ván × ~60 nước/ván × 0.05s/nước ≈ 90.000 giây ≈ **~25 giờ single-thread**. Cần multi-thread hoặc giảm depth/time limit
- **Tối ưu multi-thread**: Chia 30k ván thành N chunks, chạy N process Stockfish song song

### Assumptions
1. Stockfish binary có sẵn hoặc có thể cài qua `apt install stockfish` hoặc tải binary
2. CPL trung bình tương quan mạnh với ELO (giả thuyết dựa trên nghiên cứu cộng đồng cờ vua)
3. Sample 30k đủ đại diện cho 187M ván (stratified random sampling đảm bảo coverage)
4. Depth 10-12 của Stockfish đủ chính xác để phân biệt các mức ELO

## Questions & Open Items

1. **Stockfish binary**: Đã có cài trên máy chưa? Nếu chưa, dùng `apt` hay tải source build?
2. **Depth vs Time limit**: Nên dùng `depth=12` (cố định) hay `time=0.05s` (cố định thời gian)?
3. **Lọc ván forfeit/timeout**: Loại bỏ hay giữ lại nhưng đánh dấu flag?
4. **Multi-processing**: Bao nhiêu process song song là an toàn trên hệ thống hiện tại?
