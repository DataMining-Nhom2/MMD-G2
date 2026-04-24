---
phase: design
title: PoC Web Application Architecture (Chess AI)
description: Tài liệu thiết kế hệ thống PoC cho Web Game Cờ Vua tích hợp AI dự đoán ELO, quy hoạch kiến trúc Codebase theo mô hình Modular cho nhóm 3 thành viên.
---

# Thiết Kế Hệ Thống: Web Game Chess AI — Proof of Concept

> **⚠️ Đây là bản mẫu thiết kế (Design Template).**
> Tài liệu cung cấp ý tưởng tổng quan, quy ước codebase và các yêu cầu kỹ thuật cốt lõi. Người thực hiện hoàn toàn có thể tự tìm hiểu thêm về công nghệ và điều chỉnh chi tiết triển khai cho phù hợp với năng lực cá nhân, miễn sao tuân thủ đúng các nguyên tắc giao tiếp giữa các module được mô tả bên dưới.

> **🔴 YÊU CẦU ƯU TIÊN CAO NHẤT — ĐỌC TRƯỚC KHI CODE:**
> Hệ thống được chia thành 3 module độc lập do 3 thành viên phát triển song song. Để đảm bảo khả năng tích hợp, thành viên phụ trách Game Server **BẮT BUỘC** phải:
> 1. Tạo sẵn các **hàm Placeholder** (xem mục 2.2) với đúng cấu trúc Input/Output đã quy ước.
> 2. Xây dựng luồng giao diện hoàn chỉnh sử dụng các hàm Placeholder này (trả kết quả giả lập).
> 3. Khi module AI Model và XAI Engine hoàn thiện, team chỉ cần **thay thế nội dung hàm Placeholder bằng lời gọi thực tế** — toàn bộ giao diện và luồng dữ liệu không cần chỉnh sửa thêm.

Tài liệu này mô tả kiến trúc tổng thể của ứng dụng PoC, bao gồm ý tưởng sản phẩm (Concept) và các nguyên tắc tổ chức mã nguồn (Codebase Convention) nhằm đảm bảo khả năng phát triển song song giữa 3 thành viên trong nhóm.

---

## 1. Concept — Ý Tưởng Sản Phẩm

Hệ thống là một ứng dụng Web cho phép hai người chơi đánh cờ trực tuyến (Multiplayer), sau đó sử dụng mô hình Deep Learning để ước lượng trình độ ELO của mỗi người chơi dựa trên chất lượng nước đi và quản lý thời gian suy nghĩ trong ván đấu.

### 1.1. Luồng hoạt động chính (End-to-End Workflow)

Hệ thống vận hành qua 3 giai đoạn tuần tự:

1. **Thi đấu (Game Server):**
   - Người dùng A tạo phòng chơi, hệ thống sinh ra một đường dẫn (URL) duy nhất.
   - Người dùng B truy cập URL đó từ máy tính khác để tham gia ván đấu.
   - Hai người chơi thực hiện các nước đi theo thời gian thực với đồng hồ thi đấu (time control).
   - Hệ thống ghi nhận toàn bộ nước đi (PGN) và thời gian suy nghĩ từng nước (Clock Time).

2. **Ước lượng ELO (AI Model Pipeline):**
   - Khi ván cờ kết thúc, dữ liệu PGN và Clock Time được chuyển đến module xử lý phía backend.
   - Dữ liệu lần lượt đi qua: Stockfish Engine (trích xuất CPL, Blunder theo từng nước đi) → Mô hình CNN-LSTM (dự đoán ELO liên tục).
   - Kết quả trả về bao gồm: điểm ELO dự đoán cho Trắng và Đen, kèm các chỉ số phân tích (CPL trung bình, số lượng Blunders, nước đi bị Attention đánh dấu).

3. **Giải thích kết quả (Explainable AI - XAI):**
   - Các chỉ số phân tích từ bước 2 được đóng gói thành một Prompt có cấu trúc.
   - Prompt này được gửi đến một LLM (GPT/Gemini) thông qua API.
   - LLM trả về đoạn văn bản giải thích bằng ngôn ngữ tự nhiên lý do mô hình đưa ra mức ELO đó.
   - Kết quả ELO và lời giải thích được hiển thị trên giao diện cho cả hai người chơi.

---

## 2. Nguyên Tắc Tổ Chức Codebase

Dự án được phát triển bởi 3 thành viên trên **chung một Repository**. Để đảm bảo khả năng phát triển song song và giảm thiểu xung đột mã nguồn (merge conflict), toàn bộ codebase phải tuân thủ **Kiến trúc Mô-đun (Modular Architecture)**.

### 2.1. Phân vùng thư mục theo trách nhiệm

Mỗi thành viên làm việc trong thư mục riêng biệt của mình. Không chỉnh sửa chéo sang thư mục của người khác trừ khi có sự đồng thuận.

```
src/
├── game_server/      # Thành viên 1: Web Server, WebSocket, quản lý phòng chơi
├── ai_model/         # Thành viên 2: Stockfish Pipeline, CNN-LSTM, Data Loader
└── xai_engine/       # Thành viên 3: Prompt Engineering, tích hợp LLM API
```

### 2.2. Giao tiếp giữa các module qua Interface (API Contract)

Các module không gọi trực tiếp vào logic nội bộ của nhau, mà giao tiếp thông qua các **hàm giao diện (Interface Functions)** được định nghĩa rõ ràng về Input/Output.

Thành viên làm Game Server cần tạo sẵn các **hàm giả lập (Placeholder/Mock Functions)** thay thế cho các module chưa hoàn thiện. Điều này cho phép phát triển và kiểm thử toàn bộ luồng giao diện mà không phụ thuộc vào tiến độ của 2 module còn lại.

Ví dụ minh họa cho file `src/game_server/integration.py`:

```python
# ── Placeholder cho AI Model (Thành viên 2 sẽ thay thế khi hoàn thiện) ──
def predict_elo(pgn_string: str, clock_times: list[float]) -> dict:
    """Dự đoán ELO từ ván cờ. Trả về dict chứa kết quả."""
    return {
        "white_elo": 1500,
        "black_elo": 1200,
        "stats": {
            "white_avg_cpl": 45.0,
            "black_avg_cpl": 78.0,
            "white_blunders": 1,
            "black_blunders": 3,
        }
    }

# ── Placeholder cho XAI Engine (Thành viên 3 sẽ thay thế khi hoàn thiện) ──
def get_explanation(prediction_result: dict) -> str:
    """Sinh lời giải thích từ kết quả dự đoán ELO."""
    return "Phân tích chi tiết sẽ được cập nhật khi module XAI hoàn thiện."
```

### 2.3. Quy trình tích hợp cuối cùng (Integration)

Khi các module AI Model và XAI hoàn thiện, việc tích hợp chỉ cần thực hiện theo 2 bước:
1. Thay thế nội dung hàm Placeholder bằng lời gọi đến hàm thực tế trong `ai_model/` và `xai_engine/`.
2. Kiểm thử toàn luồng (End-to-End Testing) để xác nhận dữ liệu truyền qua các module hoạt động đúng.

> **Lưu ý:** Cấu trúc Input/Output của các hàm Placeholder chính là bản cam kết (Contract) giữa các thành viên. Mọi thay đổi về cấu trúc dữ liệu phải được thông báo và đồng thuận trước khi chỉnh sửa.

---

## 3. Chi Tiết Module: Game Server (Thành Viên 1)

Thành viên phụ trách module này chịu trách nhiệm xây dựng toàn bộ phần giao diện người dùng (Frontend) và máy chủ trò chơi (Backend Server), đảm bảo hai người chơi có thể thi đấu cờ vua trực tuyến theo thời gian thực.

### 3.1. Phạm vi công việc

| STT | Hạng mục | Mô tả |
|:---:|:---------|:------|
| 1 | **Giao diện bàn cờ** | Hiển thị bàn cờ 8x8 tương tác, cho phép người chơi kéo thả quân cờ. Kiểm tra tính hợp lệ của nước đi (luật cờ vua chuẩn FIDE). |
| 2 | **Hệ thống phòng chơi (Room)** | Cho phép tạo phòng mới (sinh URL duy nhất), người chơi thứ hai tham gia qua URL. Quản lý trạng thái phòng: chờ đối thủ / đang chơi / kết thúc. |
| 3 | **Đồng hồ thi đấu (Clock)** | Hiển thị đồng hồ đếm ngược cho mỗi bên. Hỗ trợ cấu hình thể thức thời gian (ví dụ: 5+3 Blitz, 10+0 Rapid). Ghi nhận chính xác thời gian suy nghĩ (time spent) tại từng nước đi — đây là dữ liệu đầu vào bắt buộc cho module AI. |
| 4 | **Đồng bộ trạng thái (Real-time Sync)** | Truyền tải nước đi giữa 2 máy tính theo thời gian thực thông qua WebSocket. Đảm bảo cả hai phía luôn nhìn thấy trạng thái bàn cờ giống nhau. |
| 5 | **Xuất dữ liệu ván cờ** | Khi ván cờ kết thúc, tổng hợp toàn bộ nước đi thành chuỗi PGN chuẩn và mảng Clock Time, sau đó gọi đến các hàm Placeholder (`predict_elo`, `get_explanation`) để nhận kết quả từ module AI và XAI. |
| 6 | **Hiển thị kết quả** | Sau khi nhận phản hồi từ AI/XAI, hiển thị trên giao diện: điểm ELO dự đoán của mỗi bên, các chỉ số thống kê (CPL, Blunders), và đoạn văn giải thích từ LLM. |

### 3.2. Dữ liệu đầu ra cần chuẩn bị cho module AI

Khi ván cờ kết thúc, Game Server phải chuẩn bị và truyền đi 2 đối tượng dữ liệu sau:

**a) Chuỗi PGN (Portable Game Notation):**
Chuỗi ký tự mô tả toàn bộ nước đi của ván cờ theo định dạng chuẩn quốc tế.
```
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 ...
```

**b) Mảng Clock Time (Thời gian suy nghĩ từng nước):**
Một danh sách số thực (đơn vị: giây) ghi nhận thời gian từng người chơi đã sử dụng cho mỗi nước đi. Thứ tự xen kẽ: Trắng → Đen → Trắng → Đen.
```json
[5.2, 3.1, 12.0, 8.5, 2.1, 45.3, ...]
```

### 3.3. Công nghệ tham khảo

#### Backend (Máy chủ)

| Công nghệ | Vai trò | Ghi chú |
|:-----------|:--------|:--------|
| **FastAPI** (Python) | Framework Web chính | Hỗ trợ WebSocket tích hợp sẵn, tương thích trực tiếp với code Python của module AI (cùng ngôn ngữ, không cần bridge). |
| **Flask + Flask-SocketIO** | Phương án thay thế | Đơn giản hơn FastAPI, phù hợp nếu thành viên đã quen Flask. |
| **Ngrok** hoặc **Cloudflare Tunnel** | Expose localhost ra Internet | Cho phép biến máy tính cá nhân thành server công khai mà không cần thuê hosting. Người chơi khác truy cập qua URL do Ngrok cấp (ví dụ: `https://abc123.ngrok-free.app`). |

#### Frontend (Giao diện người dùng)

| Công nghệ | Vai trò | Ghi chú |
|:-----------|:--------|:--------|
| **chessboard.js** | Thư viện hiển thị bàn cờ | Cung cấp sẵn bàn cờ đồ họa đẹp, hỗ trợ kéo thả quân cờ, tự xoay bàn theo phe. Không cần tự vẽ lại UI bàn cờ. |
| **chess.js** | Thư viện xử lý luật cờ | Kiểm tra nước đi hợp lệ, phát hiện chiếu/chiếu bí/hòa, sinh chuỗi PGN tự động. |
| **Socket.IO Client** | Kết nối WebSocket | Giao tiếp real-time giữa trình duyệt và server. |
| **HTML/CSS/JS thuần** hoặc **React** | Xây dựng giao diện | Tùy theo kinh nghiệm của thành viên. Nếu dùng HTML thuần thì đơn giản hơn, React thì dễ mở rộng về sau. |

#### Tài liệu & Ví dụ mẫu nên tham khảo

- [chessboard.js — Official Examples](https://chessboardjs.com/examples): Các ví dụ từ cơ bản đến nâng cao về hiển thị bàn cờ và xử lý sự kiện.
- [chess.js — GitHub](https://github.com/jhlywa/chess.js): Tài liệu API đầy đủ về kiểm tra luật cờ, sinh PGN.
- [FastAPI WebSocket Tutorial](https://fastapi.tiangolo.com/advanced/websockets/): Hướng dẫn chính thức cách tạo kết nối WebSocket với FastAPI.
- [Ngrok Quickstart](https://ngrok.com/docs/getting-started/): Hướng dẫn cài đặt và expose localhost trong 5 phút.
- Tìm kiếm thêm: `"multiplayer chess websocket python"` trên GitHub để tham khảo các dự án mã nguồn mở tương tự.

