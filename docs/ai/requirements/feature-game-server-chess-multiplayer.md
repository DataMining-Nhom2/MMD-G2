---
phase: requirements
title: Game Server Chess Multiplayer
description: Xây dựng module Game Server cho web chess multiplayer với FastAPI + Streamlit, tích hợp AI ELO prediction và XAI Engine.
---

# Requirements: Game Server Chess Multiplayer

## Problem Statement
**Vấn đề cần giải quyết?**

Dự án MMD-G2 đã có AI model dự đoán ELO từ PGN + Clock Times (XGBoost V3 với MAE ~200 ELO). Cần xây dựng giao diện web để:
1. Cho phép 2 người chơi đánh cờ vua online theo thời gian thực
2. Ghi nhận toàn bộ nước đi và thời gian suy nghĩ
3. Tự động gọi AI Model để dự đoán ELO sau mỗi ván
4. Hiển thị kết quả kèm lời giải thích từ XAI Engine

**Ai chịu ảnh hưởng?**
- Thành viên 1 (Game Server): Phát triển module này
- Thành viên 2 (AI Model): Cung cấp hàm `predict_elo()` được gọi từ Game Server
- Thành viên 3 (XAI Engine): Cung cấp hàm `get_explanation()` được gọi từ Game Server
- Người dùng cuối: Người chơi cờ muốn biết trình độ ELO của mình

**Tình trạng hiện tại?**
- AI Model đã được implement trong `src/` (feature engineering V2, eval XGBoost V3)
- Chưa có giao diện web để demo
- Chưa có XAI Engine

---

## Goals & Objectives
**Mục tiêu chính:**
1. Xây dựng hệ thống phòng chơi (Room) cho 2 người tham gia
2. Hiển thị bàn cờ 8x8 tương tác, kiểm tra luật FIDE
3. Đồng bộ real-time qua WebSocket
4. Đồng hồ thi đấu (Clock) chính xác
5. Xuất PGN + Clock Times khi ván kết thúc
6. Gọi AI Model + XAI Engine (mock) để hiển thị kết quả

**Mục tiêu thứ cấp:**
- Hỗ trợ nhiều thể thức thời gian (Blitz, Rapid, Classical)
- Lịch sử các ván đã chơi (in-memory)

**Non-goals (ngoài phạm vi):**
- Authentication/User accounts
- Database persistence lâu dài
- AI Model thực sự (chỉ mock interface)
- XAI Engine thực sự (chỉ mock interface)
- Multiplayer > 2 người
- Hỗ trợ mobile
- Xử lý disconnect/reconnect

---

## User Stories & Use Cases

### User Story 1: Tạo phòng chơi mới
> **Như một** người chơi muốn thử nghiệm,
> **Tôi muốn** tạo một phòng chơi mới và nhận mã phòng,
> **để** chia sẻ mã đó cho đối thủ.

**Use Cases:**
- UC1.1: Người dùng click "Tạo phòng mới" → Hệ thống sinh mã phòng 6 ký tự → Hiển thị mã để chia sẻ
- UC1.2: Người dùng copy mã phòng và gửi cho bạn
- UC1.3: Hệ thống hiển thị trạng thái "Chờ đối thủ..."

### User Story 2: Tham gia phòng chơi
> **Như một** người chơi được mời,
> **Tôi muốn** nhập mã phòng để tham gia,
> **để** bắt đầu thi đấu.

**Use Cases:**
- UC2.1: Người dùng nhập mã phòng → Hệ thống kiểm tra phòng còn trống → Tham gia thành công
- UC2.2: Phòng đã đầy → Hiển thị thông báo lỗi
- UC2.3: Phòng không tồn tại → Hiển thị thông báo lỗi

### User Story 3: Chơi cờ với đối thủ
> **Như một** người chơi,
> **Tôi muốn** di chuyển quân cờ hợp lệ,
> **để** chơi ván cờ với đối thủ theo luật FIDE.

**Use Cases:**
- UC3.1: Người dùng kéo thả quân cờ → Kiểm tra luật → Cập nhật bàn cờ → Gửi nước đi qua WebSocket
- UC3.2: Nhận nước đi từ đối thủ qua WebSocket → Cập nhật bàn cờ
- UC3.3: Kiểm tra chiếu/chiếu bí/hòa → Kết thúc ván nếu cần

### User Story 4: Đồng hồ thi đấu
> **Như một** người chơi,
> **Tôi muốn** thấy đồng hồ đếm ngược,
> **để** biết còn bao nhiêu thời gian.

**Use Cases:**
- UC4.1: Đồng hồ bắt đầu khi ván bắt đầu
- UC4.2: Đồng hồ ngừng khi tôi đi nước, bắt đầu cho đối thủ
- UC4.3: Ghi nhận thời gian suy nghĩ cho mỗi nước đi
- UC4.4: Khi hết giờ → Tự động thua

### User Story 5: Xem kết quả ELO
> **Như một** người chơi,
> **Tôi muốn** xem điểm ELO dự đoán và giải thích,
> **để** hiểu tại sao mô hình đưa ra con số đó.

**Use Cases:**
- UC5.1: Ván kết thúc → Hệ thống gọi `predict_elo(pgn, clock_times)` → Nhận kết quả
- UC5.2: Hệ thống gọi `get_explanation(prediction_result)` → Nhận lời giải thích
- UC5.3: Hiển thị ELO cho Trắng/Đen, CPL, Blunders, lời giải thích

---

## Success Criteria
**Kết quả đo lường được:**

| STT | Tiêu chí | Điều kiện đạt |
|-----|----------|----------------|
| 1 | Tạo phòng | Tạo được phòng, nhận mã phòng 6 ký tự |
| 2 | Tham gia phòng | Người thứ 2 tham gia thành công qua mã phòng |
| 3 | Di chuyển quân | Mọi nước đi hợp lệ được chấp nhận, không hợp lệ bị từ chối |
| 4 | Luật FIDE | Castling, En passant, Promotion hoạt động đúng |
| 5 | WebSocket sync | Cả 2 bên nhìn thấy bàn cờ giống nhau |
| 6 | Clock chính xác | Thời gian suy nghĩ được ghi nhận chính xác đến 0.1s |
| 7 | Xuất PGN | PGN hợp lệ, có thể import vào chess.com |
| 8 | AI Integration | Mock `predict_elo()` được gọi đúng interface |
| 9 | XAI Integration | Mock `get_explanation()` được gọi đúng interface |
| 10 | Hiển thị kết quả | ELO, stats, explanation hiển thị đầy đủ |

---

## Constraints & Assumptions

**Ràng buộc kỹ thuật:**
- Python 3.11+
- FastAPI cho backend
- Socket.IO cho real-time (FastAPI + python-socketio)
- Streamlit cho frontend
- python-chess cho xử lý luật cờ
- Chạy local, không dùng Docker

**Ràng buộc thời gian:**
- Full implementation (không MVP)

**Giả định:**
- Người dùng có trình duyệt hiện đại (Chrome, Firefox, Safari)
- Không cần authentication (local demo)
- Không cần persistence (dữ liệu mất khi tắt server)
- Không cần xử lý disconnect/reconnect
- Không cần hỗ trợ mobile
- AI Model và XAI Engine sẽ được mock với interface đã định nghĩa

---

## Questions & Open Items

**Đã xác nhận:**
1. Frontend: **Streamlit**
2. Real-time: **Socket.IO** (FastAPI + python-socketio)
3. Mobile: **Không hỗ trợ**
4. Thể thức thời gian mặc định: **15 phút (15+0)**
5. Disconnect handling: **Không cần**

**Cần xác nhận thêm:**
1. Socket.IO compatibility với FastAPI? → Cần kiểm tra, có thể dùng `python-socketio` làm bridge

**Research cần thiết:**
1. Tích hợp Socket.IO với FastAPI
2. Tích hợp chessboard.js với Streamlit
3. Cách tốt nhất để hiển thị đồng hồ cờ trên Streamlit

---

# USER COMMENTS
- về phần tạo và tham gia thì kiếm soát bằng socket là được, không cần phải lằng nhằng như thế. tạo mã phòng và cho player 2 nhập là được
- không cần mvp
- dùng socker io có kết hợp được với fastapi không
- không cần hỗ trợ mobile
- thời gian mặc đinh 15p
- không cần xử lý disconnect
