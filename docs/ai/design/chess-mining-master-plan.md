# ĐẠI PHÁP TRÌNH: CHIẾN LƯỢC MINING DỮ LIỆU CỜ VUA (MASSIVE DATA)

## I. MỤC TIÊU CỐT LÕI VÀ TRIẾT LÝ DỰ ÁN
*   **Bản chất dự án:** Không xây dựng lại hệ thống đánh giá ELO dựa trên kết quả Thắng/Thua/Hoà (như Chess.com hay Lichess đang dùng điểm số tích luỹ qua ngàn ván).
*   **Mục tiêu:** Xây dựng một cỗ máy "Performance Rating" (Năng lực tức thời). Dự đoán ELO của kỳ thủ **chỉ thông qua 1 ván đấu duy nhất** bằng cách phân tích chất lượng của từng nước đi.
*   **Sản phẩm cuối cùng:** Một ứng dụng cho phép 2 người đánh cờ (hoặc nhập file PGN), hệ thống sẽ ngầm phân tích và xuất ra: ELO ước tính của từng người trong ván đó, kèm theo lời bình luận giải thích bằng LLM (nhấn mạnh điểm mạnh/yếu).

---

## II. GIAI ĐOẠN 1: XỬ LÝ KHỐI LƯỢNG DỮ LIỆU KHỔNG LỒ (INGESTION & FORMATTING)
*   **Vấn đề:** Dữ liệu gốc cực kỳ lớn (186 triệu ván cờ, dạng PGN/JSONL hàng trăm GB). Đọc trực tiếp định dạng văn bản để train model là bất khả thi, gây nghẽn cổ chai (I/O Bottleneck).
*   **Giải pháp (Hút Chân Không):** Chuyển đổi toàn bộ dữ liệu sang định dạng **Parquet** (Columnar Storage).
    *   Sử dụng cơ chế Streaming chia lô (Batching) để xử lý mà không làm tràn RAM.
    *   Lợi ích: Nén nhị phân giảm dung lượng gấp ~10 lần, tăng tốc độ truy xuất dữ liệu theo cột dọc cực nhanh.

---

## III. GIAI ĐOẠN 2: EDA & TÌM KIẾM TẬP MẪU TINH HOA (SAMPLING)
*   **Nguyên tắc:** Không dùng toàn bộ 186 triệu ván để train model (tốn kém tài nguyên tính toán không cần thiết và dễ bị nhiễu).
*   **Quy trình Lọc:**
    1.  **Lọc loại bỏ (Garbage Collection):** Bỏ các trường siêu dữ liệu vô ích (Site, Event, Round, Date). Bỏ các ván Buller (siêu chớp), ván dưới 10 nước, ván chênh lệch ELO lớn.
    2.  **Giữ lại (Gold Features):** WhiteElo, BlackElo, Result, Moves, ECO, TimeControl, Termination.
*   **Kết quả:** Chọn lọc ra một tập mẫu (Sample) khoảng **1 đến 5 triệu ván cờ** chất lượng cao, có phân bố đại diện đồng đều cho các mức trình độ (từ ELO thấp đến đại kiện tướng).

---

## IV. GIAI ĐOẠN 3: LÒ BÁT QUÁI - FEATURE ENGINEERING
Máy học (Machine Learning) không hiểu chuỗi ký tự nước đi, bắt buộc phải số hoá đại diện cho trình độ dựa vào kiến thức miền (Domain Knowledge).
Công cụ chủ lực: **Stockfish (Chạy Local, Đa luồng)**.

**Các Nhóm Đặc Trưng (Features) Cần Rèn Ra:**
1.  **Nhóm Dữ Liệu Thô (Metadata):** ELO (Target Label), Chuyển Result thành số (1/0/0.5), Target Encoding cho mã Khai cuộc (ECO).
2.  **Nhóm Heuristic (Thuần logic nước đi):** Đếm số lượng nước đi (Độ dai dẳng), đếm tần suất ăn quân (Độ khát máu), đếm số lần chiếu Tướng.
3.  **Nhóm Engine (Tinh Hoa Stockfish):** Đẩy ván cờ vào engine để tính sai số:
    *   **CPL (Centipawn Loss):** Đo lường độ rớt điểm so với nước đi tối ưu của máy tính.
    *   **ACPL (Average Centipawn Loss):** Trung bình độ rớt điểm cả ván (Thước đo cốt lõi của đẳng cấp).
    *   **Tỉ lệ Sai Lầm (Blunder_Rate / Mistake_Rate):** Phân loại các nhóm CPL rớt nặng (>300cp) để tính tần suất mắc sai lầm ngớ ngẩn.
    *   **Endgame ACPL:** Tách riêng độ rớt điểm ở 15-20 nước cuối game để xem bản lĩnh tàn cuộc.

*   **Chiến lược Tối ưu tính toán:** Giới hạn chiều sâu của Stockfish (Depth = 10-12) để tiết kiệm 80% thời gian nhưng vẫn đủ bắt lỗi sai. Chạy Multi-worker (10-16 luồng song song) chia batch để rút ngắn thời gian xử lý xuống còn 1-2 ngày.

---

## V. GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH (MODELING & EVALUATION)
*   **Lựa chọn Thuật toán:** Sử dụng **LightGBM** (hoặc XGBoost) vì đây là "bá chủ" của dữ liệu Bảng (Tabular Data). LightGBM ưu việt hơn vì tốc độ đào tạo cực cao, tiêu tốn ít RAM và xử lý rất tốt tập dữ liệu hàng triệu ván cờ. Tránh dùng Deep Learning vì cồng kềnh, khó giải thích.
*   **Đánh giá:** Dựa vào độ lệch ELO (MAE). Sử dụng biểu đồ `Feature Importance` để chứng minh và giải thích luận điểm (giúp nhận biết ở ELO nào thì Blunder hay ACPL quan trọng hơn).

---

## VI. GIAI ĐOẠN 5: CHUẨN HOÁ SẢN PHẨM & TÍCH HỢP LLM (APPLICATION)
*   **Tích hợp UI:** Đóng gói Model đã huấn luyện vào một API. Khi có 1 ván cờ PGN mới đẩy lên, backend sẽ lấy bộ Features qua Stockfish (rất nhanh vì chỉ 1 ván), sau đó đưa vào Model LightGBM để lấy kết quả số ELO dự đoán.
*   **Thêm Hồn LLM:** Đưa thông số Features (ACPL, Blunder Rate...) cùng ELO dự kiến vào mỏm Large Language Model (như Gemini/ChatGPT) với một Prompt có bối cảnh cờ vua. LLM sẽ trả ra một đoạn nhận xét chuyên sâu bằng ngôn ngữ tự nhiên, biến đồ án từ một con số khô khan thành một "Quân sư đánh giá trình độ".

---
*Tài liệu này là thiết kế chiến lược kiến trúc tổng thể, vui lòng bám sát tư duy "Dữ liệu lớn và Lập mô hình giải thích được" trong quá trình lập trình.*
