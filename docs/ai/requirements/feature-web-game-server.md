---
phase: requirements
title: Web Game Server - Chess Multiplayer with AI ELO Prediction
description: Xây dựng Web Game Server (repo chess-realm) — multiplayer chess với React/Node.js/Socket.IO, tích hợp AI Engine Server (repo MMD-G2) để dự đoán ELO và sinh lời giải thích bằng LLM. Kiến trúc microservices theo v6-poc-system-architecture.md.
---

# Requirements: Web Game Server — Chess Multiplayer với AI ELO Prediction

## Problem Statement

**Vấn đề cần giải quyết?**

Kiến trúc v6 (`v6-poc-system-architecture.md`) đã định nghĩa rõ hệ thống cần chia thành 2 máy chủ độc lập:
- **Máy chủ 1**: Web Game Server (Node.js + React + Socket.IO) — repo `chess-realm`
- **Máy chủ 2**: AI Engine Server (Python + FastAPI) — repo `MMD-G2`

Repo `chess-realm` hiện tại đã có bộ khung cơ bản (React + MUI + react-chessboard + chess.js + Socket.IO server). Team Web cần hoàn thiện các phần còn thiếu:

1. **Đồng hồ thi đấu server-side** — chưa có, là dữ liệu bắt buộc cho AI Engine
2. **Thu thập dữ liệu cho AI** — PGN + Clock Times
3. **Gọi API AI Engine** — kết nối HTTP sang MMD-G2 để lấy ELO + Explanation
4. **Result Modal** — hiện chỉ hiển thị "Checkmate!", cần hiện ELO + Stats + LLM comment
5. **Disconnect/Reconnect handling** — giữ game state khi player tạm mất kết nối
6. **Xóa username dialog** — không login, không auth

**Ai chịu ảnh hưởng?**
- Team Web: phát triển repo `chess-realm` theo spec v6
- Team AI: cung cấp API `/api/predict-elo` tại `http://<AI_HOST>:8000`
- Người dùng cuối: người chơi cờ muốn đánh online và xem ELO dự đoán

**Tình trạng hiện tại?**
- `chess-realm` có: React (CRA) + MUI + react-chessboard (v4.5.0) + chess.js (v1.0.0-beta.8) + Socket.IO server cơ bản
- `chess-realm` thiếu: Server-side Clock, AI client, Result Modal, PGN collection, time tracking, Resign, Play Again, Disconnect/Reconnect

---

## Goals & Objectives

**Mục tiêu chính:**
1. Giữ nguyên codebase React (CRA) + MUI + Socket.IO hiện tại, chỉ mở rộng phần còn thiếu
2. Implement đồng hồ thi đấu **15+0** (server-side, chống hack), thu thập `clockTimes[]` chính xác
3. Thu thập PGN string khi ván kết thúc
4. Gọi đúng **1 request HTTP POST** sang AI Engine Server
5. Hiển thị Result Modal với ELO, ECO, CPL, Blunders, LLM Explanation
6. Handle graceful fallback khi AI Engine không phản hồi
7. Xử lý disconnect/reconnect — giữ room state, clock pause/resume, auto-rejoin khi socket reconnect

**Mục tiêu thứ cấp:**
- Xóa hoàn toàn username dialog (không đăng nhập)
- Nút "Chơi Lại" để reset ván mới
- Nút "Xin Thua" (Resign)
- Nút "Thoát Phòng" khi đang chờ hoặc đang chơi
- Danh sách nước đi (Move History) — format SAN
- Hiệu ứng loading overlay khi chờ AI phân tích
- Promotion auto-Queen khi lên hàng cuối (dùng `onPromotion` callback)

**Non-goals (ngoài phạm vi):**
- Authentication / User accounts / Database
- Hỗ trợ mobile (không tối ưu giao diện cho điện thoại)
- Nhiều hơn 2 người trong 1 phòng
- Thay đổi kiến trúc AI Engine (repo MMD-G2)
- Thay đổi API contract đã định nghĩa với AI Engine
- Sound effects
- Detect game over bằng 50-move rule hoặc threefold repetition (chỉ checkmate/stalemate)

---

## User Stories & Use Cases

### US1: Tạo phòng chơi mới
> **Như một** người chơi muốn thử nghiệm,  
> **Tôi muốn** bấm "Tạo Phòng Mới" và nhận link mời,  
> **để** chia sẻ với đối thủ bắt đầu ván đấu.

- UC1.1: Bấm "Tạo Phòng Mới" → Server sinh mã phòng 6–8 ký tự → Hiện link chia sẻ `http://host:3000/room/{roomId}` + popup "Gửi link cho đối thủ"
- UC1.2: Người tạo phòng tự động được gán **Trắng**, ngồi chờ
- UC1.3: Không cần nhập username — bước này bị loại bỏ hoàn toàn

### US2: Tham gia phòng chơi
> **Như một** người chơi được mời,  
> **Tôi muốn** mở link phòng trên trình duyệt khác,  
> **để** tự động vào phòng và được gán Đen.

- UC2.1: Mở link → WebSocket kết nối → Gán phe **Đen** → Popup chờ tắt → Ván bắt đầu
- UC2.2: Phòng đầy → Hiện thông báo lỗi
- UC2.3: Phòng không tồn tại → Hiện thông báo lỗi

### US3: Thi đấu real-time
> **Như một** người chơi,  
> **Tôi muốn** kéo thả quân cờ để đi nước,  
> **để** chơi ván cờ với đối thủ theo luật FIDE.

- UC3.1: Drag-drop quân cờ → `chess.js` kiểm tra luật → Cập nhật bàn cờ → Gửi WebSocket
- UC3.2: Nhận nước đi từ đối thủ → Cập nhật bàn cờ
- UC3.3: Luật FIDE: Castling, En passant hoạt động đúng
- UC3.4: **Promotion auto-Queen** — dùng `onPromotion` callback của react-chessboard: khi quân đến hàng cuối, tự động lên Hậu (trả về `'q'`)
- UC3.5: Phát hiện Check → Hiển thị indicator (highlight vua)
- UC3.6: Phát hiện Checkmate / Stalemate → Kết thúc ván

### US4: Đồng hồ thi đấu (Server-side)
> **Như một** người chơi,  
> **Tôi muốn** thấy đồng hồ đếm ngược,  
> **để** biết còn bao nhiêu thời gian và đo lường thời gian suy nghĩ.

- UC4.1: Đồng hồ bắt đầu khi cả 2 người đã vào phòng (**15 phút mỗi bên, 15+0**)
- UC4.2: Server-side clock là nguồn sự thật — tránh người chơi dùng DevTools hack
- UC4.3: Khi đi nước → đồng hồ bên mình **dừng**, bên đối thủ **chạy**
- UC4.4: Server tính `timeSpent` từ `lastMoveTimestamp` khi switch clock, lưu vào `clockTimes[]`
- UC4.5: Hết giờ → bên đó thua (`timeout`)
- UC4.6: Server sync đồng hồ mỗi 1 giây tới cả 2 client qua `clock_update` event

### US5: Kết thúc ván & Phân tích AI
> **Như một** người chơi,  
> **Tôi muốn** xem kết quả ELO và lời bình luận từ AI,  
> **để** hiểu trình độ và nhận xét về ván đấu.

- UC5.1: Ván kết thúc (Checkmate / Timeout / Resign / Stalemate) → Server tổng hợp PGN + clockTimes
- UC5.2: Hiện overlay loading: *"Đang phân tích ELO bằng AI..."*
- UC5.3: Gửi HTTP POST `/api/predict-elo` sang AI Engine
- UC5.4: Nhận response `{ white_elo, black_elo, eco, stats, explanation }` → Hiện Result Modal
- UC5.5: AI Engine lỗi/timeout → Vẫn hiện kết quả cơ bản, thông báo fallback

### US6: Chơi lại ván mới
> **Như một** người chơi,  
> **Tôi muốn** bấm "Chơi Lại" sau khi xem kết quả,  
> **để** bắt đầu ván mới ngay trong cùng phòng.

- UC6.1: Bấm "Chơi Lại" → Reset bàn cờ về vị trí ban đầu → Reset đồng hồ → Sẵn sàng ván mới
- UC6.2: Giữ nguyên phe Trắng/Đen đã gán, không cần tạo phòng mới

### US7: Xin thua
> **Như một** người chơi,  
> **Tôi muốn** bấm "Xin Thua" khi thấy thế cờ bất lợi,  
> **để** chủ động kết thúc ván và nhận kết quả ELO.

- UC7.1: Bấm "Xin Thua" → Server phát hiện → `game_over` với `reason: "resign"`

### US8: Disconnect & Reconnect
> **Như một** người chơi bị mất mạng tạm thời hoặc vô tình refresh trang,  
> **Tôi muốn** khi reconnect thì quay lại ván đấu đang dở,  
> **để** không mất tiến độ ván cờ.

- UC8.1: **Refresh = disconnect** → Socket mất kết nối → Reconnect flow chạy bình thường
- UC8.2: **1 player disconnect**:
  - Server KHÔNG xóa phòng
  - Opponent nhận `opponent_disconnected` → hiện indicator "Đối thủ đã disconnect"
  - Clock của bên disconnect → **tạm dừng (pause)**. Clock bên còn lại → **tiếp tục chạy**
  - Timeout vẫn xử lý bình thường nếu bên còn lại hết giờ
- UC8.3: **Player reconnect** (bấm F5 hoặc mất mạng rồi vào lại):
  - Client kiểm tra `localStorage` có `sessionToken` → emit `reconnect` kèm token
  - Server lookup token → restore full game state → gán lại `socketId` mới
  - Opponent nhận `opponent_reconnected` → tắt indicator
  - Clock: bên vừa reconnect → **tiếp tục từ lúc pause** (hoặc reset về lúc disconnect — tùy impl)
- UC8.4: **Cả 2 cùng disconnect** → server xóa phòng sau 30 giây không ai reconnect
- UC8.5: **Sau `game_over`**: cả 2 đóng tab → sau **5 phút** không ai vào lại → xóa room
- UC8.6: **Room timeout khi đang chờ** (chỉ 1 người trong phòng `waiting`):
  - Sau 5 phút không ai join → server tự xóa phòng
  - Người tạo phòng đóng tab → opponent đang chờ → opponent nhận thông báo → có thể thoát

### US9: Thoát phòng
> **Như một** người chơi đang chờ hoặc đang chơi,  
> **Tôi muốn** bấm "Thoát Phòng" để rời đi,  
> **để** không bị treo trong phòng vô ích.

- UC9.1: Khi đang ở trạng thái `waiting` (chờ opponent) → bấm "Thoát Phòng" → rời phòng, về Lobby
- UC9.2: Khi đang `playing` → bấm "Thoát Phòng" → tự động xử lý như "Xin Thua"
- UC9.3: Opponent nhận `opponent_left` → hiện thông báo "Đối thủ đã rời phòng"

---

## Success Criteria

**Kết quả đo lường được (phải PASS tất cả):**

| # | Tiêu chí | Điều kiện đạt |
|---|----------|----------------|
| 1 | Tạo phòng | Mã phòng duy nhất, hiện link mời shareable trên port 3000 |
| 2 | Vào phòng | Mở link trên tab khác → vào đúng phòng, được gán Đen |
| 3 | Realtime sync | Đi nước trên máy A → máy B cập nhật trong < 1 giây |
| 4 | Clock server-side | Đồng hồ chạy trên server, sync 1 giây/lần tới client |
| 5 | Clock tracking | Mỗi nước đi server tính `timeSpent`, lưu vào `clockTimes[]` |
| 6 | Default 15 phút | Time control mặc định là 15+0 |
| 7 | Clock pause on disconnect | Bên disconnect → clock tạm dừng. Bên còn → clock tiếp tục |
| 8 | Timeout detection | Hết giờ → server phát hiện `timeout`, trigger `game_over` |
| 9 | Checkmate detection | Chiếu bí → tự phát hiện, trigger `game_over` |
| 10 | Resign | Bấm "Xin Thua" → trigger `game_over` |
| 11 | Exit room | Bấm "Thoát Phòng" → rời phòng hoặc xử lý như resign |
| 12 | PGN build đúng | Khi `game_over` → build chuỗi PGN chuẩn SAN (VD: "1. e4 e5 2. Nf3 Nc6") |
| 13 | Gọi AI Engine | POST `/api/predict-elo` với đúng payload JSON |
| 14 | Hiển thị Result Modal | ELO, ECO, CPL, Blunders, Explanation hiển thị đầy đủ |
| 15 | AI fallback | AI Engine không phản hồi → vẫn hiện kết quả cơ bản, không crash |
| 16 | Chơi Lại | Reset bàn cờ + clock, giữ nguyên phòng |
| 17 | Không login | Không có username dialog, không auth |
| 18 | Reconnect restore | Refresh trang → reconnect → khôi phục full game state |
| 19 | Socket ID reset | Sau reconnect, socket ID thay đổi nhưng game state vẫn đúng |
| 20 | Auto-promote Queen | Lên hàng cuối → `onPromotion` callback → tự động thành Hậu |
| 21 | Move History | Hiển thị danh sách nước đi dạng SAN, 2 cột Trắng/Đen |
| 22 | Room cleanup | Cả 2 disconnect → xóa phòng. Sau 5 phút không ai vào → xóa phòng |
| 23 | Console debug | Mỗi nước đi log `timeSpent`, mỗi `game_over` log PGN + clockTimes |

---

## Constraints & Assumptions

**Ràng buộc kỹ thuật:**
- Frontend: **React + CRA** (giữ nguyên codebase hiện tại, không migrate Vite)
- UI: **MUI v5** (đã dùng, giữ nguyên)
- Chess Board: **react-chessboard v4.5.0** + **chess.js v1.0.0-beta.8** (đã dùng, giữ nguyên)
- Real-time: **Socket.IO v4.7.5** (đã dùng, giữ nguyên)
- Backend: **Node.js + Express** (đã có, cần mở rộng)
- Web Server port: **3000**
- AI Engine URL: env var `AI_ENGINE_URL`, default `http://localhost:8000`
- Không dùng: Database, ORM, Auth library, Docker, Sound

**Ràng buộc kiến trúc:**
- Chỉ giao tiếp với AI Engine qua **1 endpoint duy nhất**: `POST /api/predict-elo`
- Tất cả game state lưu **in-memory** (RAM), không persistence
- Đồng hồ **server-side** là nguồn sự thật (chống client-side manipulation)
- Clock **pause** khi player disconnect, **resume** khi reconnect
- Session token dùng `localStorage` để restore sau refresh/disconnect

**Giả định:**
- Người dùng có trình duyệt hiện đại (Chrome/Firefox mới nhất) trên desktop
- Không cần hỗ trợ mobile (UI không tối ưu cho điện thoại)
- AI Engine Server chạy trên cổng 8000 và cùng mạng LAN (hoặc localhost)
- **Refresh trang = disconnect** → reconnect flow chạy bình thường (đây là use case chính)
- **react-chessboard v4.5.0**: dùng `onPromotion` callback để auto-promote Queen (trả về `'q'`); `boardOrientation` để flip board; `position` nhận FEN string
- **chess.js v1.0.0-beta.8**: dùng `.move()` để validate và thực hiện nước đi, `.fen()` lấy FEN, `.isCheckmate()`, `.isStalemate()`, `.isDraw()`, `.inCheck()`
- **Socket.IO**: dùng `socket.io.on('reconnect', ...)` để handle auto-reconnect; socket ID có thể thay đổi sau reconnect
- Game over chỉ detect bằng **Checkmate** và **Stalemate** (không detect 50-move rule hoặc threefold repetition)

---

## Questions & Open Items

**Đã xác định rõ (from spec v6 + user confirmation):**
1. Tech stack: React + CRA + MUI + react-chessboard + chess.js + Socket.IO + Node.js/Express ✅
2. Không login, không auth ✅
3. Default time control: **15 phút + 0 giây (15+0)** ✅
4. Server port: **3000** ✅
5. Server-side clock là nguồn sự thật ✅
6. Promotion: **auto-Queen** qua `onPromotion` callback ✅
7. Move History: **format SAN** (e4, Nf3, O-O) ✅
8. Không sound effects ✅
9. Không mobile support ✅
10. **Disconnect/Reconnect**: clock pause/resume, session token, refresh = disconnect ✅
11. **Cả 2 disconnect**: xóa phòng sau 30 giây ✅
12. **Sau game_over**: 5 phút không ai vào → xóa phòng ✅
13. **Thoát phòng**: có thể thoát, đang chơi = resign ✅
14. Game over: chỉ Checkmate + Stalemate ✅
15. API endpoint: `POST http://localhost:8000/api/predict-elo` ✅
16. API payload: `{ pgn, clock_times, result, time_control }` ✅
17. API response: `{ success, data: { white_elo, black_elo, eco, stats, explanation } }` ✅

**Không còn câu hỏi mở** — tất cả đã được xác nhận.
