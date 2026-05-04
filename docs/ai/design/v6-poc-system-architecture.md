---
phase: design
title: PoC System Architecture v6 — Chess AI Web Demo
description: Tài liệu thiết kế hệ thống PoC cho Web Game Cờ Vua tích hợp AI dự đoán ELO. Kiến trúc Microservices phân tán, chia rõ trách nhiệm giữa Web Game Server và AI Engine Server.
---

# Thiết Kế Hệ Thống v6: Web Chess AI Demo — Proof of Concept

> **Tài liệu này là bản thiết kế chính thức (Living Document).**  
> Mọi thành viên trong nhóm phải đọc kỹ và tuân thủ trước khi viết code. Mọi thay đổi kiến trúc phải được đồng thuận và cập nhật vào đây.

---

## Phần 1: Ý Tưởng, Thực Trạng và Luồng Hoạt Động Lý Tưởng

---

### 1.1. Ý Tưởng Sản Phẩm (Product Concept)

Xây dựng một ứng dụng Web Demo (PoC) cho phép:

1. **Hai người chơi ở hai máy tính khác nhau** tham gia đánh một ván cờ vua trực tuyến theo thời gian thực (Online Multiplayer).
2. Khi ván cờ kết thúc, hệ thống **tự động thu thập** toàn bộ lịch sử nước đi (PGN) và thời gian suy nghĩ từng nước (Clock Time).
3. Dữ liệu ván cờ được gửi đến một Mô hình Deep Learning (CNN-BiLSTM) để **ước lượng trình độ ELO** của cả hai người chơi dựa trên chất lượng nước đi và quản lý thời gian suy nghĩ.
4. Kết quả ELO kèm theo các chỉ số phân tích (CPL, Blunders, Mã khai cuộc ECO) được đóng gói thành Prompt gửi đến LLM (GPT/Gemini) để **sinh lời giải thích bằng ngôn ngữ tự nhiên** — giống như một bình luận viên chuyên nghiệp phân tích ván đấu.

> **Nguyên tắc thiết kế:**  
> - Đây là bản Demo, **KHÔNG** cần Đăng nhập, Đăng ký, Database lưu trữ người dùng.  
> - Tập trung 100% vào trải nghiệm cốt lõi: **Đánh cờ → Xem điểm AI → Đọc bình luận.**  
> - Kiến trúc phải chuẩn chỉ, tách bạch rõ ràng để dễ mở rộng và phân chia công việc trong nhóm.

---

### 1.2. Thực Trạng — Những Gì Đang Có và Vấn Đề (Current State & Problems)

#### a) Repo `Chess-Predict-Demo` (Web — Node.js/React)

Một thành viên đã xây dựng một ứng dụng Web khá hoàn chỉnh với React + Vite (Frontend) và Express.js (Backend). Tuy nhiên, bản này **đi quá xa so với yêu cầu PoC**:

| Vấn đề | Chi tiết |
|:-------|:---------|
| Thừa tính năng | Đăng ký, Đăng nhập, Avatar, Database Supabase, Upload ảnh — tất cả đều không cần thiết cho Demo. |
| Phụ thuộc ngoài | Yêu cầu file `.env` chứa khóa bảo mật Supabase mới chạy được. Không có file này thì Backend crash ngay lập tức. |
| Thiếu tính năng cốt lõi | Chưa có hệ thống Phòng chơi (Room), chưa có WebSocket Multiplayer, chưa có đồng hồ thi đấu, chưa có kết nối sang AI. |

**Kết luận:** Bản này cần được **viết lại từ đầu** theo đúng đặc tả mới bên dưới. Có thể tái sử dụng một số component React (bàn cờ, giao diện) nhưng phải gỡ bỏ toàn bộ hệ thống Auth/Database.

#### b) Repo `MMD-G2` (AI & Data — Python)

Đây là repo chính chứa toàn bộ pipeline xử lý dữ liệu và mô hình AI. Bên trong có một thư mục `src/game_server/` đã được code sẵn bằng Python (FastAPI) với đầy đủ:

| Thành phần | Trạng thái |
|:-----------|:-----------|
| `main.py` | Server FastAPI + WebSocket + Frontend HTML nhúng trực tiếp trong code Python. |
| `integration.py` | Hàm Placeholder `predict_elo()` và `get_explanation()` — đang trả dữ liệu giả (Mock). |
| `rooms.py` | Quản lý phòng chơi, trạng thái ván đấu. |
| `chess_engine.py` | Bọc thư viện `python-chess` để kiểm tra luật đi cờ. |
| `clock.py` | Quản lý đồng hồ thi đấu. |

**Vấn đề:** Code Frontend (HTML/CSS/JS) bị nhét thẳng vào biến String bên trong file Python `main.py`. Đây là cách làm **thiếu chuyên nghiệp**, rất khó bảo trì và không thể phát triển giao diện phức tạp hơn. Kiến trúc Monolithic (gom tất cả vào 1 cục) khiến code AI và code Web bị dí dính vào nhau.

**Kết luận:** Tách phần Web UI ra khỏi repo này. Repo `MMD-G2` chỉ giữ lại vai trò **AI Engine Server** — một API Python thuần túy chạy ngầm chờ nhận dữ liệu và trả kết quả.

---

### 1.3. Kiến Trúc Mục Tiêu — Microservices (Target Architecture)

Hệ thống được chia thành **2 máy chủ độc lập**, mỗi máy chủ là một repo riêng, chạy trên cổng riêng, viết bằng ngôn ngữ riêng. Chúng giao tiếp với nhau qua **duy nhất 1 đường API HTTP**.

```
┌─────────────────────────────────────────────────────────────────┐
│                        NGƯỜI CHƠI A                             │
│                     (Máy tính / Trình duyệt)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  WebSocket (real-time)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MÁY CHỦ 1: WEB GAME SERVER                        │
│              (Node.js + React — Repo mới)                       │
│                                                                 │
│  Nhiệm vụ:                                                     │
│  • Hiển thị bàn cờ, quản lý phòng chơi (Room)                  │
│  • WebSocket: đồng bộ nước đi giữa 2 máy tính                  │
│  • Đồng hồ thi đấu (Clock / Time Control)                      │
│  • Thu thập PGN + Clock Times khi ván kết thúc                  │
│  • Gọi API sang Máy chủ 2 để lấy kết quả AI                    │
│  • Hiển thị ELO + Lời giải thích LLM lên giao diện             │
│                                                                 │
│  KHÔNG chứa: Database, Auth, Stockfish, Model AI, LLM           │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP POST (1 request duy nhất)
                           │  Gửi: { pgn, clock_times }
                           │  Nhận: { elo, stats, explanation }
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MÁY CHỦ 2: AI ENGINE SERVER                        │
│              (Python / FastAPI — Repo MMD-G2)                   │
│                                                                 │
│  Nhiệm vụ:                                                     │
│  • Nhận PGN + Clock Times từ Web Server                         │
│  • Phân loại mã Khai cuộc ECO từ chuỗi PGN                     │
│  • Chạy Stockfish: trích xuất CPL, Blunders theo từng nước      │
│  • Chạy Model CNN-BiLSTM (hoặc XGBoost): dự đoán ELO           │
│  • Tạo Prompt gửi LLM API (GPT/Gemini): sinh lời giải thích    │
│  • Đóng gói kết quả JSON trả về cho Web Server                  │
│                                                                 │
│  KHÔNG chứa: Giao diện Web, Bàn cờ, WebSocket, Room             │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │  WebSocket (real-time)
┌──────────────────────────┴──────────────────────────────────────┐
│                        NGƯỜI CHƠI B                             │
│                     (Máy tính / Trình duyệt)                   │
└─────────────────────────────────────────────────────────────────┘
```

> **Lưu ý quan trọng:**  
> Người chơi A và Người chơi B đều kết nối vào cùng một Máy chủ 1 (Web Game Server) qua WebSocket. Máy chủ 2 (AI Engine) hoàn toàn vô hình với người chơi — nó chỉ được Máy chủ 1 gọi đến khi cần phân tích.

---

### 1.4. Luồng Hoạt Động Lý Tưởng End-to-End (Ideal User Flow)

#### Bước 1: Tạo Phòng (Lobby)
- Người chơi A mở trình duyệt, truy cập địa chỉ Web Game Server (ví dụ: `http://localhost:3000`).
- Màn hình hiển thị một giao diện tối giản: **Nút `[ Tạo Phòng Mới ]`**.
- Không cần đăng nhập, không cần gõ tên. Bấm là chơi.

#### Bước 2: Mời Đối Thủ
- Sau khi bấm Tạo Phòng, hệ thống sinh ra một đường link duy nhất chứa mã phòng (ví dụ: `http://localhost:3000/room/a1b2c3`).
- Hiện popup to rõ ràng: *"Gửi link này cho đối thủ để bắt đầu!"*
- Người A giữ phe Trắng, ngồi chờ.

#### Bước 3: Đối Thủ Vào Phòng
- Người chơi B (ở một máy tính khác) mở link trong trình duyệt của mình.
- WebSocket kết nối thành công. Popup chờ tắt đi.
- Người B được gán phe Đen. **Trận đấu chính thức bắt đầu!**
- Đồng hồ thi đấu kích hoạt (ví dụ: 5 phút mỗi bên).

#### Bước 4: Thi Đấu Real-time
- Người A (Trắng) đi một nước → WebSocket truyền đến Người B trong < 0.1 giây → Bàn cờ trên máy B tự cập nhật.
- Đồng hồ bên Trắng dừng, đồng hồ bên Đen bắt đầu đếm ngược.
- Hai bên đánh qua đánh lại cho đến khi:
  - Một bên bị **Chiếu bí (Checkmate)**.
  - Một bên **Hết giờ (Timeout)**.
  - Một bên bấm nút **Xin Thua (Resign)**.
  - Hòa (hết nước đi / lặp thế cờ).

#### Bước 5: Phân Tích AI (Hậu trường — Người chơi không nhìn thấy)
- Màn hình cả hai bên mờ đi, hiện overlay: *"🔄 Đang phân tích trình độ ELO bằng AI..."*
- **Ngầm bên dưới:**
  1. Web Game Server gom toàn bộ PGN và mảng Clock Times.
  2. Gửi 1 request `POST` sang AI Engine Server (Repo MMD-G2).
  3. AI Engine xử lý tuần tự:
     - Phân loại mã ECO (Khai cuộc) từ các nước đi đầu tiên.
     - Chạy Stockfish để tính CPL và đếm Blunders cho mỗi bên.
     - Đưa đặc trưng vào Model Deep Learning → Ra ELO dự đoán.
     - Tạo Prompt kèm ECO + CPL + ELO → Gọi API LLM → Nhận lời giải thích.
  4. Trả JSON kết quả về cho Web Game Server.

#### Bước 6: Hiển Thị Kết Quả (Result Modal)
- Overlay loading tắt đi. Bung ra bảng kết quả (Modal) ngay giữa màn hình:

  ```
  ┌──────────────────────────────────────┐
  │         ♚  KẾT QUẢ VÁN ĐẤU  ♚       │
  │                                      │
  │       TRẮNG THẮNG (Chiếu bí)         │
  │                                      │
  │   ┌──────────┐    ┌──────────┐       │
  │   │  TRẮNG   │    │   ĐEN    │       │
  │   │   1540   │    │   1200   │       │
  │   │   ELO    │    │   ELO    │       │
  │   └──────────┘    └──────────┘       │
  │                                      │
  │  📊 Thống kê:                        │
  │  • Khai cuộc: Sicilian Defense (B50) │
  │  • CPL Trắng: 25.3 | CPL Đen: 78.1  │
  │  • Blunders: Trắng 0 | Đen 3        │
  │                                      │
  │  🤖 AI Nhận xét:                     │
  │  "Trắng khai cuộc Phòng thủ Sicilian│
  │   biến thể B50 cực kỳ bài bản. Đen  │
  │   mắc 3 sai lầm nghiêm trọng từ     │
  │   nước 15-22, đặc biệt nước Hậu d5  │
  │   ở nước 18 khiến mất kiểm soát     │
  │   hoàn toàn trung tâm..."           │
  │                                      │
  │        [ 🔄 Chơi Lại Ván Mới ]       │
  └──────────────────────────────────────┘
  ```

- Cả hai người chơi (Máy A và Máy B) đều nhìn thấy bảng này đồng thời.
- Nút **"Chơi Lại"** sẽ đưa cả hai về trạng thái bàn cờ mới, sẵn sàng cho ván kế tiếp.

---

## Phần 2: Đặc Tả Kỹ Thuật — Web Game Server (Dành cho Web Developer)

> **⚠️ LƯU Ý QUAN TRỌNG:**  
> Tài liệu Phần 2 này là bản **tham khảo kiến trúc**, không phải hướng dẫn từng dòng code.  
> Web Developer hoàn toàn có thể tự điều chỉnh chi tiết triển khai theo năng lực cá nhân, **miễn sao tuân thủ đúng các nguyên tắc giao tiếp (API Contract) và cấu trúc dữ liệu được mô tả bên dưới.**  
> Phần đánh dấu `<!-- AI-HIGHLIGHT -->` là những đoạn **BẮT BUỘC** phải tuân thủ chính xác vì phía AI Engine Server phụ thuộc vào cấu trúc này.

---

### 2.1. Công Nghệ Khuyến Nghị (Tech Stack)

| Tầng | Công nghệ | Vai trò | Ghi chú |
|:-----|:----------|:--------|:--------|
| **Frontend** | React + Vite | Xây dựng giao diện SPA | Dùng thư viện `react-chessboard` để render bàn cờ, `chess.js` để kiểm tra luật đi. |
| **Backend** | Node.js + Express | HTTP Server + REST API | Phục vụ file tĩnh (React build) và làm trung gian gọi AI Engine. |
| **Real-time** | Socket.IO | WebSocket 2 chiều | Đồng bộ nước đi, đồng hồ, trạng thái phòng giữa 2 trình duyệt. |
| **Proxy** | `axios` hoặc `node-fetch` | Gọi HTTP sang AI Engine | Dùng trong Backend Node.js để POST dữ liệu sang Python API. |

> Không cần Database. Không cần ORM. Không cần Auth library. Toàn bộ state quản lý trong bộ nhớ RAM (in-memory).

---

### 2.2. Cấu Trúc Thư Mục Tham Khảo

```
chess-web-game/
├── server/                     # Backend Node.js
│   ├── index.js                # Entry point: Express + Socket.IO setup
│   ├── roomManager.js          # Quản lý phòng chơi (tạo/xóa/tìm phòng)
│   ├── gameLogic.js            # Xử lý logic ván cờ phía server (validate move, detect game over)
│   └── aiClient.js             # Module gọi API sang AI Engine Server
│
├── client/                     # Frontend React + Vite
│   ├── src/
│   │   ├── App.jsx             # Router chính (Lobby vs Game)
│   │   ├── pages/
│   │   │   ├── Lobby.jsx       # Màn hình tạo phòng / nhập mã phòng
│   │   │   └── GameRoom.jsx    # Màn hình bàn cờ + đồng hồ + kết quả
│   │   ├── components/
│   │   │   ├── ChessBoard.jsx  # Component bàn cờ (react-chessboard)
│   │   │   ├── Clock.jsx       # Component đồng hồ đếm ngược
│   │   │   ├── MoveHistory.jsx # Danh sách nước đi
│   │   │   └── ResultModal.jsx # Modal hiển thị kết quả ELO + AI giải thích
│   │   └── index.css           # Styling
│   └── index.html
│
├── package.json
└── README.md
```

---

### 2.3. Quản Lý Phòng Chơi — Room Manager

Mỗi phòng chơi (Room) là một object lưu trong bộ nhớ RAM của server. Không cần Database.

**Cấu trúc dữ liệu phòng (tham khảo):**

```javascript
// roomManager.js
const rooms = new Map();  // Lưu tất cả phòng đang hoạt động

function createRoom() {
  const roomId = generateId();  // Sinh mã ngẫu nhiên 6-8 ký tự
  rooms.set(roomId, {
    id: roomId,
    status: 'waiting',        // 'waiting' | 'playing' | 'finished'
    players: {
      white: null,             // socket.id của người chơi Trắng
      black: null,             // socket.id của người chơi Đen
    },
    // ──────────────────────────────────────────────────
    // <!-- AI-HIGHLIGHT: Hai trường dữ liệu dưới đây 
    //      BẮT BUỘC phải thu thập chính xác vì AI Engine
    //      phụ thuộc vào chúng để phân tích -->
    moves: [],                 // Mảng các nước đi SAN: ["e4", "e5", "Nf3", ...]
    clockTimes: [],            // Mảng thời gian suy nghĩ (giây): [5.2, 3.1, 12.0, ...]
    // ──────────────────────────────────────────────────
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  // Trạng thái hiện tại
    timeControl: { initial: 300, increment: 0 },  // 5 phút, không có cộng giờ
    whiteTimeLeft: 300,        // Giây còn lại của Trắng
    blackTimeLeft: 300,        // Giây còn lại của Đen
    lastMoveTimestamp: null,   // Date.now() cập nhật mỗi nước đi
  });
  return roomId;
}
```

**Quy tắc quản lý phòng:**
- Người đầu tiên vào phòng → gán phe **Trắng** (`players.white = socket.id`).
- Người thứ hai vào phòng → gán phe **Đen**, chuyển `status` sang `'playing'`.
- Nếu người thứ ba cố vào → từ chối kết nối (phòng đầy).
- Khi ván kết thúc → chuyển `status` sang `'finished'`.
- (Optional) Tự dọn phòng sau 5 phút không hoạt động để giải phóng RAM.

---

### 2.4. Giao Thức WebSocket — Socket Events

Dưới đây là danh sách các sự kiện Socket.IO cần implement. Hướng mũi tên cho biết chiều truyền dữ liệu.

#### a) Kết nối & Phòng chơi

| Event | Hướng | Payload | Mô tả |
|:------|:------|:--------|:------|
| `create_room` | Client → Server | `{}` | Client yêu cầu tạo phòng mới. Server trả `room_created`. |
| `room_created` | Server → Client | `{ roomId: "abc123" }` | Trả mã phòng cho client hiển thị link mời. |
| `join_room` | Client → Server | `{ roomId: "abc123" }` | Client yêu cầu vào phòng. |
| `joined` | Server → Client | `{ color: "white"\|"black", roomId, fen }` | Xác nhận đã vào phòng, gán màu quân. |
| `opponent_joined` | Server → Client | `{}` | Thông báo đối thủ đã vào. Bắt đầu ván đấu! |
| `room_full` | Server → Client | `{ message: "Phòng đã đầy" }` | Từ chối nếu phòng đã có 2 người. |

#### b) Trong ván đấu

| Event | Hướng | Payload | Mô tả |
|:------|:------|:--------|:------|
| `make_move` | Client → Server | `{ move: "Nf3", fen: "...", timeSpent: 5.2 }` | Người chơi gửi nước đi. `timeSpent` tính bằng giây. |
| `move_made` | Server → Opponent | `{ move: "Nf3", fen: "...", whiteTime, blackTime }` | Broadcast nước đi sang đối thủ + cập nhật đồng hồ. |
| `clock_update` | Server → Both | `{ whiteTime: 245.3, blackTime: 280.1 }` | (Optional) Sync đồng hồ định kỳ mỗi 1 giây. |
| `resign` | Client → Server | `{}` | Người chơi xin thua. |

<!-- AI-HIGHLIGHT: BẮT BUỘC — Cấu trúc sự kiện `make_move` -->

> **Quan trọng:** Mỗi lần người chơi đi một nước, client **BẮT BUỘC** phải tính và gửi kèm trường `timeSpent` (số giây thực tế người chơi đã suy nghĩ cho nước đi đó). Server sẽ lưu giá trị này vào mảng `room.clockTimes[]`. Đây là dữ liệu đầu vào bắt buộc cho Model AI.
>
> **Cách tính `timeSpent` ở Frontend:**
> ```javascript
> // Lưu timestamp khi đến lượt mình
> let turnStartTime = Date.now();
>
> // Khi người chơi thực hiện nước đi
> function onMove(moveData) {
>   const timeSpent = (Date.now() - turnStartTime) / 1000;  // Đổi sang giây
>   socket.emit('make_move', {
>     move: moveData.san,
>     fen: game.fen(),
>     timeSpent: Math.round(timeSpent * 100) / 100  // Làm tròn 2 chữ số
>   });
>   turnStartTime = Date.now();  // Reset cho lượt tiếp theo
> }
> ```

#### c) Kết thúc ván

| Event | Hướng | Payload | Mô tả |
|:------|:------|:--------|:------|
| `game_over` | Server → Both | `{ result, reason }` | Thông báo ván kết thúc. Bắt đầu gọi AI. |
| `ai_loading` | Server → Both | `{}` | Báo client hiện overlay Loading "Đang phân tích..." |
| `ai_result` | Server → Both | `{ ...AI Response... }` | Trả kết quả từ AI Engine. Client hiện Result Modal. |
| `ai_error` | Server → Both | `{ message }` | Nếu AI Engine lỗi thì thông báo, vẫn hiện kết quả cơ bản (không có ELO). |

**Giá trị `result`:** `"1-0"` (Trắng thắng) | `"0-1"` (Đen thắng) | `"1/2-1/2"` (Hòa).  
**Giá trị `reason`:** `"checkmate"` | `"timeout"` | `"resign"` | `"stalemate"` | `"draw"`.

---

### 2.5. Xử Lý Đồng Hồ Thi Đấu (Clock)

- Đồng hồ mặc định: **5 phút mỗi bên** (300 giây), không cộng giờ (increment = 0).
- Đồng hồ chỉ bắt đầu đếm **sau khi cả 2 người chơi đã vào phòng**.
- Khi một bên đi xong nước cờ → đồng hồ bên đó **dừng**, đồng hồ bên kia **chạy**.
- Nếu đồng hồ của bên nào về **0** → bên đó thua do hết giờ (`timeout`).

**Lưu ý kỹ thuật:**
- Đồng hồ chính xác phải chạy ở **phía Server** (tránh hack bằng DevTools ở client).
- Client chỉ nhận giá trị `whiteTime` / `blackTime` từ Server để hiển thị, **không tự đếm ngược** làm nguồn sự thật.
- (Trong PoC Demo, nếu để đơn giản hóa, có thể chấp nhận đếm ở Client và chỉ sync định kỳ.)

---

### 2.6. Thu Thập Dữ Liệu Khi Ván Kết Thúc

<!-- AI-HIGHLIGHT: PHẦN NÀY LÀ CỐT LÕI — BẮT BUỘC TUÂN THỦ -->

Khi ván cờ kết thúc (bất kể lý do), Backend Node.js phải thực hiện tuần tự:

**Bước 1:** Tổng hợp chuỗi PGN từ mảng `room.moves[]`.

```javascript
// Ví dụ: room.moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
// Kết quả PGN: "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"

function buildPGN(moves) {
  let pgn = '';
  for (let i = 0; i < moves.length; i++) {
    if (i % 2 === 0) {
      pgn += `${Math.floor(i / 2) + 1}. `;
    }
    pgn += moves[i] + ' ';
  }
  return pgn.trim();
}
```

**Bước 2:** Lấy mảng `room.clockTimes[]` nguyên bản.

```javascript
// Ví dụ: [5.2, 3.1, 12.0, 8.5, 2.1, 45.3]
// Thứ tự xen kẽ: Trắng, Đen, Trắng, Đen...
const clockTimes = room.clockTimes;
```

**Bước 3:** Gửi HTTP POST sang AI Engine Server (xem mục 2.7).

---

### 2.7. API Contract — Giao Tiếp Với AI Engine Server

<!-- AI-HIGHLIGHT: ĐÂY LÀ BẢN CAM KẾT (CONTRACT) GIỮA 2 TEAM — KHÔNG ĐƯỢC TỰ Ý THAY ĐỔI -->

Khi ván cờ kết thúc, Backend Node.js gọi đúng **1 request duy nhất** sang AI Engine:

#### Request

```
POST http://<AI_ENGINE_HOST>:8000/api/predict-elo
Content-Type: application/json
```

**Body gửi đi:**

```json
{
  "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
  "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3, 3.0, 7.2],
  "result": "1-0",
  "time_control": "5+0"
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|:-------|:-----|:---------|:------|
| `pgn` | `string` | ✅ | Chuỗi PGN chuẩn quốc tế, chỉ chứa nước đi (không chứa header). |
| `clock_times` | `number[]` | ✅ | Mảng số thực (giây), xen kẽ Trắng → Đen. Độ dài = tổng số nước đi. |
| `result` | `string` | ✅ | Kết quả ván: `"1-0"` \| `"0-1"` \| `"1/2-1/2"`. |
| `time_control` | `string` | ❌ | Thể thức thời gian (ví dụ `"5+0"`, `"10+5"`). Mặc định `"5+0"`. |

#### Response (AI Engine trả về)

```json
{
  "success": true,
  "data": {
    "white_elo": 1540,
    "black_elo": 1200,
    "eco": {
      "code": "B50",
      "name": "Sicilian Defense"
    },
    "stats": {
      "white_avg_cpl": 25.3,
      "black_avg_cpl": 78.1,
      "white_blunders": 0,
      "black_blunders": 3,
      "total_moves": 42
    },
    "explanation": "Trắng khai cuộc Phòng thủ Sicilian biến thể B50 cực kỳ bài bản. Đen mắc 3 sai lầm nghiêm trọng từ nước 15-22, đặc biệt nước Hậu d5 ở nước 18 khiến mất kiểm soát hoàn toàn trung tâm..."
  }
}
```

| Trường | Kiểu | Mô tả |
|:-------|:-----|:------|
| `success` | `boolean` | `true` nếu xử lý thành công, `false` nếu lỗi. |
| `data.white_elo` | `number` | ELO dự đoán cho phe Trắng (range: 400–3000). |
| `data.black_elo` | `number` | ELO dự đoán cho phe Đen. |
| `data.eco.code` | `string` | Mã ECO khai cuộc (VD: `"B50"`, `"C42"`). |
| `data.eco.name` | `string` | Tên khai cuộc bằng tiếng Anh (VD: `"Sicilian Defense"`). |
| `data.stats` | `object` | Các chỉ số thống kê phân tích ván đấu. |
| `data.explanation` | `string` | Đoạn văn bản giải thích bằng ngôn ngữ tự nhiên (tiếng Việt), do LLM sinh ra. |

#### Response khi lỗi

```json
{
  "success": false,
  "error": "Stockfish engine not available"
}
```

> **Lưu ý cho Web Developer:** Nếu AI Engine trả về lỗi hoặc timeout (> 30 giây), Web Server vẫn phải hiển thị kết quả ván đấu (Thắng/Thua/Hòa), kèm dòng thông báo: *"Không thể kết nối AI Engine. Phân tích ELO tạm thời không khả dụng."* — Không được để trang web treo trắng.

---

### 2.8. Code Tham Khảo — Gọi AI Engine Từ Node.js

```javascript
// aiClient.js — Module gọi API sang AI Engine Server

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

async function requestELOPrediction(room) {
  const pgn = buildPGN(room.moves);
  const payload = {
    pgn: pgn,
    clock_times: room.clockTimes,
    result: room.result,          // "1-0", "0-1", "1/2-1/2"
    time_control: "5+0",
  };

  try {
    const response = await fetch(`${AI_ENGINE_URL}/api/predict-elo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000),  // Timeout 30 giây
    });

    const result = await response.json();

    if (result.success) {
      return result.data;
    } else {
      console.error('[AI] Error:', result.error);
      return null;
    }
  } catch (error) {
    console.error('[AI] Connection failed:', error.message);
    return null;  // Trả null để Web Server tự xử lý fallback
  }
}
```

---

### 2.9. Luồng Xử Lý Game Over Trong Backend Node.js

```javascript
// Trong index.js — Khi phát hiện ván kết thúc

async function handleGameOver(room, result, reason, io) {
  room.status = 'finished';
  room.result = result;

  // 1. Báo cho cả 2 client: ván kết thúc + hiện Loading
  io.to(room.id).emit('game_over', { result, reason });
  io.to(room.id).emit('ai_loading');

  // 2. Gọi AI Engine (bất đồng bộ)
  const aiResult = await requestELOPrediction(room);

  // 3. Trả kết quả cho client
  if (aiResult) {
    io.to(room.id).emit('ai_result', aiResult);
  } else {
    io.to(room.id).emit('ai_error', {
      message: 'Không thể kết nối AI Engine. Phân tích ELO tạm thời không khả dụng.'
    });
  }
}
```

---

### 2.10. Danh Sách Màn Hình UI Cần Xây Dựng (Frontend React)

#### Màn hình 1: Lobby (Sảnh chờ)
- Nền tối, thiết kế tối giản sang trọng.
- Nút lớn: **`[ Tạo Phòng Mới ]`**.
- Ô input nhỏ: *"Hoặc nhập mã phòng..."* + nút **`[ Vào Phòng ]`**.
- Không có form đăng nhập. Không có sidebar. Chỉ 2 thành phần trên.

#### Màn hình 2: Game Room (Phòng đấu)
- **Header:** Mã phòng + Màu quân của bạn.
- **Vùng chính (Grid 3 cột):**
  - Cột trái: Thông tin Đen + Đồng hồ Đen.
  - Cột giữa: Bàn cờ 480×480px (react-chessboard). Xoay hướng theo phe.
  - Cột phải: Thông tin Trắng + Đồng hồ Trắng.
- **Phía dưới:** Danh sách nước đi (Move History) + Nút "Xin Thua".
- **Trạng thái hiện:** Khi chỉ có 1 người → Hiện popup chờ đối thủ kèm link mời.

#### Màn hình 3: Result Modal (Bảng kết quả)
- Overlay mờ phủ toàn màn hình.
- Card trung tâm chứa:
  - Tiêu đề: Kết quả (Trắng thắng / Đen thắng / Hòa).
  - 2 ô ELO: Trắng và Đen (số to, font đậm).
  - Bảng thống kê nhỏ: Khai cuộc (ECO), CPL, Blunders.
  - Khung text: Lời nhận xét từ LLM (scrollable nếu dài).
  - Nút: **`[ Chơi Lại ]`**.
- **Trạng thái Loading:** Trước khi có kết quả AI, hiện spinner + dòng chữ *"Đang phân tích ELO..."*

---

### 2.11. Checklist Nghiệm Thu Cho Web Developer

Trước khi tuyên bố "xong", hãy đảm bảo tất cả các mục sau đều PASS:

- [ ] Tạo phòng mới sinh mã duy nhất, hiện link mời share được.
- [ ] Mở link trên trình duyệt khác (hoặc Incognito) → vào đúng phòng, được gán Đen.
- [ ] Đi nước cờ trên máy A → máy B cập nhật bàn cờ trong < 1 giây.
- [ ] Đồng hồ chạy đúng bên đang đi, dừng bên đã đi xong.
- [ ] Hết giờ → tự phát hiện thua (timeout) và trigger game_over.
- [ ] Chiếu bí → tự phát hiện và trigger game_over.
- [ ] Bấm "Xin Thua" → trigger game_over.
- [ ] Khi game_over → hiện Loading overlay → Gọi API AI Engine.
- [ ] Nhận response từ AI → hiện Result Modal với ELO + Explanation.
- [ ] AI Engine không phản hồi (timeout/lỗi) → vẫn hiện kết quả cơ bản, không crash.
- [ ] Bấm "Chơi Lại" → reset bàn cờ, sẵn sàng ván mới.
- [ ] **Console.log ghi đầy đủ:** Mỗi nước đi ghi kèm `timeSpent`, mỗi game_over ghi PGN + clockTimes để debug.

---

## Phần 3: Đặc Tả Kỹ Thuật — AI Engine Server (Repo MMD-G2)

> **Phần này do Team AI phụ trách.**  
> AI Engine Server là một ứng dụng Python chạy độc lập, đóng vai trò "bộ não" phân tích ván cờ.  
> Nó **KHÔNG** có giao diện, **KHÔNG** có bàn cờ, **KHÔNG** có WebSocket.  
> Chỉ mở đúng **1 endpoint API** duy nhất để chờ Web Game Server gọi đến.

---

### 3.1. Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Vai trò |
|:-----------|:----------|:--------|
| **API Framework** | FastAPI + Uvicorn | Mở cổng HTTP để nhận request từ Web Server. |
| **Chess Engine** | Stockfish (binary) + `python-chess` | Phân tích từng nước đi, tính CPL, đếm Blunders. |
| **Deep Learning** | PyTorch (CNN-BiLSTM) hoặc XGBoost | Model đã được train sẵn — chỉ cần load weights và inference. |
| **ECO Classifier** | File từ điển `eco.json` + `python-chess` | Map các nước đi mở đầu sang mã ECO + tên Khai cuộc. |
| **XAI / LLM** | Gemini API hoặc OpenAI API | Gửi Prompt chứa số liệu phân tích → Nhận lời giải thích bằng ngôn ngữ tự nhiên. |
| **Data Processing** | Polars / Pandas / NumPy | Xử lý và biến đổi dữ liệu đặc trưng trước khi đưa vào Model. |

---

### 3.2. Vị Trí Trong Repo MMD-G2

AI Engine Server sẽ được đặt trong thư mục `src/` của repo hiện tại, tổ chức lại như sau:

```
MMD-G2/
├── src/
│   ├── ai_engine/                  # ← THƯ MỤC MỚI: API Server
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app, định nghĩa endpoint /api/predict-elo
│   │   ├── pipeline.py             # Orchestrator: điều phối toàn bộ pipeline xử lý
│   │   ├── eco_classifier.py       # Phân loại mã ECO từ PGN
│   │   ├── stockfish_analyzer.py   # Gọi Stockfish tính CPL, Blunders
│   │   ├── model_predictor.py      # Load model + inference → ELO
│   │   ├── llm_explainer.py        # Tạo Prompt + gọi LLM API → Lời giải thích
│   │   └── eco_data/
│   │       └── eco.json            # Từ điển ECO (mã → tên khai cuộc)
│   │
│   ├── game_server/                # (Giữ lại để tham khảo, không dùng trong production)
│   ├── eval_rating_net.py          # Code train/eval Model CNN-BiLSTM (có sẵn)
│   ├── eval_xgboost_v3.py          # Code train/eval XGBoost (có sẵn)
│   ├── feature_engineering.py      # Trích xuất đặc trưng (có sẵn)
│   └── ...
│
├── models/                         # Thư mục chứa file weights Model đã train
│   ├── cnn_bilstm_best.pt          # (hoặc tên file weights thực tế)
│   └── xgboost_v3.json
│
├── data/
│   └── ...
└── ...
```

> **Lưu ý:** Thư mục `src/game_server/` cũ (code Monolithic nhúng HTML) sẽ được giữ lại để tham khảo nhưng **KHÔNG** dùng trong kiến trúc mới. Toàn bộ logic AI được chuyển sang `src/ai_engine/`.

---

### 3.3. FastAPI Endpoint

File `src/ai_engine/main.py` chỉ cần định nghĩa đúng **1 endpoint**:

```python
# src/ai_engine/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.ai_engine.pipeline import run_prediction_pipeline

app = FastAPI(title="MMD-G2 AI Engine", description="Chess ELO Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────

class PredictRequest(BaseModel):
    pgn: str                        # Chuỗi PGN nước đi
    clock_times: list[float]        # Mảng thời gian suy nghĩ (giây)
    result: str = "1-0"             # Kết quả ván: "1-0", "0-1", "1/2-1/2"
    time_control: str = "5+0"       # Thể thức thời gian

class PredictResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None

# ─── Endpoint duy nhất ────────────────────────────────────────

@app.post("/api/predict-elo", response_model=PredictResponse)
async def predict_elo_endpoint(request: PredictRequest):
    """
    Nhận PGN + Clock Times từ Web Game Server.
    Chạy pipeline phân tích và trả kết quả ELO + giải thích.
    """
    try:
        result = await run_prediction_pipeline(
            pgn=request.pgn,
            clock_times=request.clock_times,
            game_result=request.result,
            time_control=request.time_control,
        )
        return PredictResponse(success=True, data=result)
    except Exception as e:
        return PredictResponse(success=False, error=str(e))
```

**Khởi chạy server:**

```bash
# Từ thư mục gốc MMD-G2
uvicorn src.ai_engine.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 3.4. Pipeline Xử Lý (5 Bước Tuần Tự)

File `src/ai_engine/pipeline.py` là bộ điều phối (Orchestrator). Khi nhận được request, nó thực hiện 5 bước tuần tự:

```
PGN + Clock Times (Input)
    │
    ▼
┌─────────────────────────────────┐
│  Bước 1: Phân loại ECO          │  → eco_classifier.py
│  (Map nước đi đầu → mã B50)    │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  Bước 2: Stockfish Analysis     │  → stockfish_analyzer.py
│  (Tính CPL + đếm Blunders)     │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  Bước 3: Feature Engineering    │  → feature_engineering.py (có sẵn)
│  (Ghép CPL + Clock → Tensor)   │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  Bước 4: Model Inference        │  → model_predictor.py
│  (CNN-BiLSTM hoặc XGBoost)     │
│  → Output: white_elo, black_elo│
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  Bước 5: LLM Explanation        │  → llm_explainer.py
│  (Prompt = ECO + CPL + ELO)     │
│  → Output: explanation string   │
└─────────────┬───────────────────┘
              ▼
     JSON Response (Output)
```

**Code tham khảo `pipeline.py`:**

```python
# src/ai_engine/pipeline.py

from src.ai_engine.eco_classifier import classify_eco
from src.ai_engine.stockfish_analyzer import analyze_game
from src.ai_engine.model_predictor import predict_elo
from src.ai_engine.llm_explainer import generate_explanation


async def run_prediction_pipeline(
    pgn: str,
    clock_times: list[float],
    game_result: str,
    time_control: str,
) -> dict:
    """
    Chạy toàn bộ pipeline phân tích ván cờ.
    Trả về dict chứa ELO, ECO, stats, explanation.
    """

    # Bước 1: Phân loại Khai cuộc (ECO)
    eco = classify_eco(pgn)
    # → {"code": "B50", "name": "Sicilian Defense"}

    # Bước 2: Chạy Stockfish phân tích từng nước đi
    analysis = analyze_game(pgn)
    # → {"white_avg_cpl": 25.3, "black_avg_cpl": 78.1,
    #     "white_blunders": 0, "black_blunders": 3,
    #     "per_move_cpl": [...], "total_moves": 42}

    # Bước 3 + 4: Ghép features rồi cho vào Model inference
    elo_result = predict_elo(
        pgn=pgn,
        clock_times=clock_times,
        stockfish_features=analysis,
    )
    # → {"white_elo": 1540, "black_elo": 1200}

    # Bước 5: Tạo Prompt gửi cho LLM lấy lời giải thích
    explanation = await generate_explanation(
        eco=eco,
        stats=analysis,
        elo=elo_result,
        game_result=game_result,
        time_control=time_control,
    )
    # → "Trắng khai cuộc Phòng thủ Sicilian biến thể B50..."

    # Đóng gói kết quả trả về
    return {
        "white_elo": elo_result["white_elo"],
        "black_elo": elo_result["black_elo"],
        "eco": eco,
        "stats": {
            "white_avg_cpl": analysis["white_avg_cpl"],
            "black_avg_cpl": analysis["black_avg_cpl"],
            "white_blunders": analysis["white_blunders"],
            "black_blunders": analysis["black_blunders"],
            "total_moves": analysis["total_moves"],
        },
        "explanation": explanation,
    }
```

---

### 3.5. Chi Tiết Từng Module

#### a) `eco_classifier.py` — Phân loại Khai cuộc

- Load file `eco_data/eco.json` (bảng tra cứu ~500 khai cuộc chuẩn quốc tế).
- Parse PGN lấy **5–15 nước đi đầu tiên**.
- So khớp dãy nước đi với bảng ECO (greedy match — lấy dãy dài nhất khớp).
- Trả về `{"code": "B50", "name": "Sicilian Defense"}`.
- Nếu không khớp → trả về `{"code": "A00", "name": "Uncommon Opening"}`.

#### b) `stockfish_analyzer.py` — Phân tích bằng Stockfish

- Yêu cầu: Có sẵn **Stockfish binary** trên máy (path cấu hình qua biến môi trường `STOCKFISH_PATH`).
- Replay toàn bộ ván cờ từ PGN bằng `python-chess`.
- Tại mỗi nước đi, gọi Stockfish đánh giá thế cờ (evaluation) với depth tham khảo (depth=15–20 cho PoC).
- Tính **Centipawn Loss (CPL)** = `|eval_trước - eval_sau|` cho từng nước.
- Đếm **Blunders** = nước đi có CPL > 200 (ngưỡng tham khảo).
- Trả về dict tổng hợp: `white_avg_cpl`, `black_avg_cpl`, `white_blunders`, `black_blunders`, `per_move_cpl`, `total_moves`.

> **Lưu ý hiệu năng:** Phân tích Stockfish là bước **tốn thời gian nhất** trong pipeline (có thể mất 5–15 giây tùy số nước đi và depth). Cân nhắc giảm depth xuống 12–15 cho PoC Demo để giữ response time < 30 giây.

#### c) `model_predictor.py` — Suy luận Model

- Load weights từ file đã train (`models/cnn_bilstm_best.pt` hoặc `models/xgboost_v3.json`).
- Nhận đầu vào: PGN (để tạo board state cho CNN), Clock Times, và Stockfish features (CPL per move).
- Chạy inference → Output: `white_elo` (int), `black_elo` (int).
- **Quan trọng:** Model chỉ cần load **1 lần** khi server khởi động, sau đó cache trong RAM. Không được load lại mỗi request.

```python
# model_predictor.py — Ý tưởng khởi tạo

import torch

# Load 1 lần duy nhất khi import module
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = torch.load("models/cnn_bilstm_best.pt", map_location="cpu")
        MODEL.eval()
    return MODEL

def predict_elo(pgn, clock_times, stockfish_features):
    model = load_model()
    # ... feature engineering + inference ...
    return {"white_elo": 1540, "black_elo": 1200}
```

#### d) `llm_explainer.py` — Sinh lời giải thích bằng LLM

- Nhận tất cả số liệu phân tích (ECO, CPL, Blunders, ELO dự đoán).
- Soạn một **Prompt có cấu trúc** (Structured Prompt) chứa đầy đủ ngữ cảnh.
- Gọi API LLM (Gemini / OpenAI) để sinh lời giải thích bằng **tiếng Việt**.
- Trả về chuỗi `str` giải thích.

**Prompt tham khảo:**

```python
def build_prompt(eco, stats, elo, game_result, time_control):
    return f"""Bạn là một bình luận viên cờ vua chuyên nghiệp.
Hãy phân tích ngắn gọn (3-5 câu) ván cờ sau dựa trên các số liệu:

- Khai cuộc: {eco['name']} ({eco['code']})
- Thể thức: {time_control}
- Kết quả: {game_result}
- ELO dự đoán: Trắng {elo['white_elo']}, Đen {elo['black_elo']}
- CPL trung bình: Trắng {stats['white_avg_cpl']:.1f}, Đen {stats['black_avg_cpl']:.1f}
- Blunders: Trắng {stats['white_blunders']}, Đen {stats['black_blunders']}
- Tổng số nước: {stats['total_moves']}

Phân tích bằng tiếng Việt, tập trung vào:
1. Nhận xét chiến thuật khai cuộc
2. Đánh giá chất lượng nước đi (CPL) của mỗi bên
3. Chỉ ra điểm yếu dẫn đến thua/thắng
Giọng văn tự nhiên, dễ hiểu, không dùng thuật ngữ quá chuyên sâu."""
```

> **Lưu ý:** API Key cho LLM được đặt trong biến môi trường (`GEMINI_API_KEY` hoặc `OPENAI_API_KEY`). Nếu không có key hoặc LLM lỗi, trả về chuỗi fallback: *"Phân tích chi tiết tạm thời không khả dụng."* — Pipeline vẫn phải trả đủ ELO và stats.

---

### 3.6. Biến Môi Trường Cần Cấu Hình (`.env`)

```env
# AI Engine Server
PORT=8000
STOCKFISH_PATH=/path/to/stockfish          # Đường dẫn tới binary Stockfish
STOCKFISH_DEPTH=15                          # Depth phân tích (15-20 cho PoC)
MODEL_PATH=models/cnn_bilstm_best.pt       # Đường dẫn tới file weights

# LLM API (chọn 1 trong 2)
GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
```

---

### 3.7. Checklist Nghiệm Thu Cho AI Team

- [ ] Server khởi động thành công tại `http://localhost:8000`.
- [ ] Truy cập `http://localhost:8000/docs` → Hiển thị Swagger UI (tự động bởi FastAPI).
- [ ] Gửi POST `/api/predict-elo` với PGN hợp lệ → Nhận response JSON đúng format.
- [ ] ECO trả về đúng mã khai cuộc (test: `1. e4 e5` → C20 King's Pawn Game).
- [ ] Stockfish chạy được, CPL và Blunders có giá trị hợp lý (không âm, không vô cực).
- [ ] Model inference trả ELO trong khoảng hợp lý (400–3000).
- [ ] LLM trả lời bằng tiếng Việt, đề cập đúng khai cuộc và số liệu.
- [ ] Gửi PGN rỗng/sai format → Trả `success: false` kèm error message, không crash server.
- [ ] Response time toàn pipeline < 30 giây (PoC chấp nhận được).
- [ ] Model chỉ load 1 lần khi startup, không reload mỗi request.

---

*Hết tài liệu thiết kế. Đội ngũ triển khai dựa trên bản đặc tả này và báo cáo kết quả theo Checklist nghiệm thu tại Phần 2.11 và Phần 3.7.*
