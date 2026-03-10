---
phase: requirements
title: Requirements & Problem Understanding — EDA cho dự đoán ELO realtime
description: Phân tích khai phá dữ liệu (EDA) chuyên sâu phục vụ xây dựng mô hình dự đoán ELO người chơi từ vài nước đi đầu tiên
---

# Requirements & Problem Understanding — EDA cho Dự đoán ELO Realtime

## Problem Statement
**Bài toán chúng ta đang giải quyết là gì?**

- **Vấn đề cốt lõi**: Xây dựng một mô hình ML có khả năng **ước lượng ELO rating** của một người chơi cờ vua chỉ dựa trên **vài nước đi đầu tiên** (5-15 nước đi) trong một ván cờ realtime — trước khi ván cờ kết thúc.
- **Pha EDA này giải quyết gì?**: Trước khi xây mô hình, cần hiểu sâu dữ liệu để trả lời: *Liệu thông tin trong vài nước đi đầu có đủ tín hiệu (signal) để dự đoán ELO không?* Những features nào mang tính dự đoán cao nhất?
- **Ai bị ảnh hưởng?**: Các nền tảng cờ vua online (Lichess, Chess.com) muốn phát hiện smurf accounts, matchmaking tốt hơn; người chơi muốn hiểu trình độ thực sự; giảng viên cờ muốn đánh giá học viên nhanh.
- **Tình trạng hiện tại**: Dữ liệu thô (~187 triệu ván cờ từ Lichess, tháng 12/2025 & 01/2026) đã được parse thành Parquet. Chưa có EDA hay feature engineering nào.

## Goals & Objectives
**Chúng ta muốn đạt được gì?**

### Mục tiêu chính (Primary Goals)
1. **Hiểu phân phối ELO** trong dataset: trung bình, median, skewness, các ELO band phổ biến
2. **Phân tích mối quan hệ Opening (ECO) ↔ ELO**: Khai cuộc nào đặc trưng cho từng mức ELO? Đây là tín hiệu mạnh nhất từ vài nước đi đầu
3. **Đánh giá tính khả thi (feasibility)** của bài toán: Liệu first-N-moves có đủ discriminative power để phân biệt ELO bands?
4. **Phát hiện class imbalance**: Phân bố ELO có bị lệch nghiêm trọng không? Cần stratified sampling hay oversampling?
5. **Xác định feature candidates** cho mô hình: những biến nào từ dữ liệu hiện có (và có thể trích xuất thêm) sẽ có giá trị dự đoán cao

### Mục tiêu phụ (Secondary Goals)
6. **Phân tích ảnh hưởng Time Control**: ELO prediction accuracy có khác nhau giữa Bullet/Blitz/Rapid/Classical?
7. **Phân tích move sequence patterns**: N-gram nước đi, opening tree depth theo ELO band
8. **Phát hiện outliers và data noise**: Ván cờ bất thường, ELO bất thường (sandbaggers, new accounts)
9. **Đánh giá chất lượng cột Moves**: Tỷ lệ missing, format consistency, average length

### Non-goals (Ngoài phạm vi)
- **KHÔNG** xây mô hình ML trong pha này (chỉ EDA + insight)
- **KHÔNG** tích hợp Stockfish engine evaluation (sẽ xem xét ở giai đoạn feature engineering)
- **KHÔNG** xử lý deployment hay API (giai đoạn 4+)
- **KHÔNG** phân tích endgame patterns (focus vào opening/early middlegame)
- **NGOẠI LỆ chấp nhận**: Quick baseline model (DecisionTree/RandomForest/XGBoost-GPU trên sample nhỏ) để đo feature importance được phép, nhưng KHÔNG tuning hay deploy

## User Stories & Use Cases
**Người dùng sẽ tương tác với kết quả EDA như thế nào?**

1. **Là một Data Engineer**, tôi muốn **hiểu phân phối ELO** để **biết cần binning như thế nào** cho balanced classes khi training, giảm riệu ro class imbalance
2. **Là một ML Engineer**, tôi muốn **biết features nào có correlation cao với ELO** để **ưu tiên feature engineering** đúng thứ tự, tiết kiệm thời gian phát triển
3. **Là một Data Scientist**, tôi muốn **thấy opening preferences theo ELO band** để **validate giả thuyết** "opening choice ≈ proxy for skill level" trước khi invest vào feature phức tạp
4. **Là một stakeholder**, tôi muốn **xem báo cáo trực quan** với biểu đồ rõ ràng để **quyết định có đầu tư** vào mô hình ELO prediction không
5. **Là một researcher**, tôi muốn **biết time control ảnh hưởng thế nào** để **quyết định** nên train một model chung hay tách theo format

### Edge Cases cần xét
- Ván cờ có ELO cực thấp (<600) hoặc cực cao (>2800) — outliers?
- Ván bị forfeit/timeout ngay nước đầu (NumMoves rất thấp)
- Người chơi mới chưa có ELO ổn định (provisional rating)
- ECO code giống nhau nhưng line khác nhau (A00 rất rộng)

## Success Criteria
**Làm sao biết EDA hoàn thành tốt?**

1. ✅ **Notebook chạy mượt**: EDA_Notebook.ipynb chạy end-to-end không lỗi, load data nhanh (<30s cho sampling)
2. ✅ **Tối thiểu 8-10 biểu đồ** chất lượng cao (title, xlabel, ylabel, legend, annotation) — vượt mức 3-5 trong TODO.md gốc
3. ✅ **Mỗi biểu đồ kèm Markdown insight**: Giải thích ý nghĩa nghiệp vụ cờ vua + hàm ý cho mô hình ML
4. ✅ **Trả lời được 3 câu hỏi chiến lược**:
   - *Q1*: First-N-moves features có đủ signal cho ELO prediction không? → Feasibility assessment
   - *Q2*: Nên bin ELO thành bao nhiêu classes, thế nào? → Class design
   - *Q3*: Features nào ưu tiên engineer đầu tiên? → Feature roadmap
5. ✅ **Phát hiện ≥3 insights actionable** dẫn đến quyết định cụ thể cho giai đoạn Feature Engineering
6. ✅ **Document class imbalance** với con số cụ thể và đề xuất giải pháp (sampling strategy)

## Constraints & Assumptions
**Các ràng buộc và giả định**

### Technical Constraints
- **CPU**: Intel Core i5-14600KF — 20 logical threads. Tận dụng multi-thread cho Polars
- **GPU**: NVIDIA GeForce RTX 3060 — 12 GB VRAM. Dùng cho GPU-accelerated feature importance (XGBoost/LightGBM GPU), và có thể thử cuDF (RAPIDS) cho data wrangling nhanh hơn
- **RAM**: 31 GB tổng, ~24 GB available → đủ load sample 3-5M rows + visualization. Không đủ load toàn bộ 187M rows. Cần sampling hoặc lazy evaluation (Polars LazyFrame)
- **Storage**: Parquet ~45 GB (2 files). Đọc bằng Polars scan_parquet (lazy, zero-copy)
- **Compute**: Ưu tiên Polars trên CPU cho data wrangling (20 threads), GPU cho model-based feature importance assessment
- **Visualization**: Matplotlib + Seaborn. Không dùng interactive dashboard (Plotly/Dash) để giữ notebook nhẹCác constraints này sẽ ảnh hưởng đến testing strategy, implementation details, và timeline. Cần cân nhắc trade-offs giữa thoroughness của EDA và performance. 

### Business Constraints
- EDA phải hoàn thành trước khi bắt đầu Feature Engineering (Giai đoạn 3)
- Kết quả EDA phải đủ rõ ràng để justify whether to proceed with the ELO prediction model

### Time Constraints
- Effort ước tính: 5.5–8.5 giờ (xem chi tiết tại planning doc)
- Không có hard deadline, nhưng EDA là blocker cho Giai đoạn 3 → càng sớm càng tốt

### Assumptions
1. **ELO trong dataset là đáng tin cậy** (Lichess rated games, đã filter bỏ unrated)
2. **Opening choice correlates với skill**: Giả thuyết cốt lõi — cần validate bằng EDA
3. **First 10-15 moves đủ thông tin**: Dựa trên nghiên cứu Maia Chess (Microsoft Research) cho thấy move patterns strongly ELO-dependent
4. **Cả 2 tháng data có phân phối tương tự**: Không có seasonal shift lớn giữa Dec 2025 và Jan 2026
5. **Sample 1-5M ván đủ đại diện** cho EDA (không cần dùng toàn bộ 187M)

## Dữ liệu Hiện có (Data Inventory)

### Schema Parquet (14 cột)
| Cột              | Kiểu        | Mô tả                                      | Vai trò cho ELO Prediction |
|------------------|-------------|---------------------------------------------|---------------------------|
| Result           | Categorical | "1-0", "0-1", "1/2-1/2"                    | Target phụ / context       |
| WhiteElo         | Int16       | ELO người chơi quân trắng                   | **TARGET chính**           |
| BlackElo         | Int16       | ELO người chơi quân đen                     | **TARGET chính**           |
| EloAvg           | Int16       | (WhiteElo + BlackElo) / 2                   | Proxy for game level       |
| NumMoves         | Int16       | Tổng số nước đi trong ván                   | Feature / filter           |
| WhiteRatingDiff  | Int16       | Thay đổi rating White sau ván               | Context                    |
| BlackRatingDiff  | Int16       | Thay đổi rating Black sau ván               | Context                    |
| ECO              | Categorical | Mã khai cuộc (A00-E99)                      | **Feature rất quan trọng** |
| Termination      | Categorical | "Normal", "Time forfeit", ...               | Feature / filter           |
| Moves            | String      | Chuỗi nước đi SAN đầy đủ                   | **Nguồn feature chính**   |
| ResultNumeric    | Float32     | 1.0 / 0.5 / 0.0                            | Target phụ                 |
| BaseTime         | Int16       | Thời gian cơ bản (giây)                     | Feature context            |
| Increment        | Int16       | Thời gian cộng thêm mỗi nước (giây)        | Feature context            |
| GameFormat       | Categorical | "Bullet"/"Blitz"/"Rapid"/"Classical"/...    | Stratification variable    |

### Volume
- **lichess_2025-12_ml.parquet**: 93,891,902 rows (22.51 GB)
- **lichess_2026-01_ml.parquet**: 93,428,457 rows (22.42 GB)
- **Tổng**: ~187.3 triệu ván cờ

## Questions & Open Items
**Những câu hỏi cần làm rõ qua EDA**

1. **Phân phối ELO có bimodal không?** (Ví dụ: peak ở ~1500 cho casual và ~2000 cho serious players)
2. **ECO codes nào có discriminative power cao nhất** giữa các ELO bands?
3. **Bao nhiêu nước đi đầu là đủ** để phân biệt ELO? (5? 10? 15?)
4. **Time control có nên là một stratification variable** hay một input feature?
5. **Tỷ lệ games bị timeout/forfeit** ở mỗi ELO band — có cần filter riêng?
6. **Có cần tách mô hình** cho White vs Black hay train chung?
7. **Move sequence encoding**: One-hot per move? N-gram? Embedding? → quyết định sau EDA
