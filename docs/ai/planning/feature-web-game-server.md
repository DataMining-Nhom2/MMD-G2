---
phase: planning
title: Web Game Server - Project Planning & Task Breakdown (Repo chess-realm)
description: Kế hoạch triển khai chi tiết cho Web Game Server (repo chess-realm). Phase-based task breakdown với dependencies, estimates, và milestones. Tech: React + Socket.IO + Node.js/Express, time control 15+0, server-side clock (pause/resume), disconnect/reconnect, room lifecycle.
---

# Planning: Web Game Server — Chess Multiplayer với AI ELO Prediction

## Milestones

- [ ] **M1: Foundation** ✅ — Done: 1.1–1.5 all complete
- [ ] **M2: Multiplayer Core** ✅ — Done: Socket events đúng spec, board auto-flip/promote
- [ ] **M3: Game Logic** ✅ — Done: Clock 15+0 pause/resume, PGN collection, time tracking, game over detection, check indicator
- [ ] **M4: Disconnect/Reconnect** ✅ — Done: session token, clock pause/resume, exit room, cleanup timers
- [ ] **M5: AI Integration** ✅ — Done: AI client, Result Modal, fallback
- [ ] **M6: Polish & QA** ✅ — Done: Play Again, Result Modal, check indicator, clock component

---

## Task Breakdown

### Phase 1: Foundation — Cơ sở hạ tầng

> **Mục tiêu:** Clean codebase, xóa auth, fix bug, setup routing, session token

#### 1.1: Fix Bug Room Check trong Server
- [x] Fix `room.length` → `room.players` (rooms là Map, không phải array)
- [x] Verify tất cả room access đều dùng `rooms.get(roomId)`

#### 1.2: Xóa Username Dialog hoàn toàn
- [x] Xóa `CustomDialog` username prompt từ `App.js`
- [x] Xóa state `username`, `usernameSubmitted`
- [x] Xóa event `username` emit lên server
- [x] Xóa `socket.data.username` khỏi server
- [x] Thay player display bằng màu quân (Trắng/Đen) thay vì username

#### 1.3: Setup React Router
- [x] Cài `react-router-dom@6`
- [x] Route `/` → `Lobby` (tạo/nhập phòng)
- [x] Route `/room/:roomId` → `GameRoom`
- [x] Lobby sau khi tạo phòng → redirect sang `/room/:roomId`
- [x] Direct access `/room/:roomId` → join phòng ngay
- [x] Tạo `pages/Lobby.js` và `pages/GameRoom.js` (tách từ App.js)

#### 1.4: Backend — Restructure
- [x] Extract `roomManager.js` — tạo file mới, quản lý phòng + session token
- [x] Extract `gameLogic.js` — skeleton (validate move, build PGN, detect game over)
- [x] Extract `clockManager.js` — skeleton (pause/resume, time tracking)
- [x] Extract `aiClient.js` — skeleton (mock)
- [x] Thêm endpoint `GET /health`
- [x] Đổi server port: `8080` → **`3000`**

#### 1.5: Session Token Infrastructure
- [x] Server: sinh UUID v4 cho mỗi player khi join, lưu vào `room.sessionTokens[color]`
- [x] Map `sessionToken → { roomId, color }` trong `disconnectedSessions`
- [x] Client: lưu `sessionToken` vào `localStorage` key `'chess_session_token'`

**Estimated:** ~4–5 giờ

---

### Phase 2: Multiplayer Core — Giao tiếp Real-time đúng Spec

> **Mục tiêu:** Socket.IO events đúng contract, 2 người chơi đồng bộ bàn cờ

#### 2.1: Implement Room Manager đúng Spec
- [x] `createRoom()`: sinh mã phòng 6–8 ký tự, set `status: 'waiting'`, `players: { white: socket.id, black: null }`, sinh `sessionToken` cho người tạo
- [x] `joinRoom()`: gán phe Đen cho người thứ 2, sinh `sessionToken` cho người join, set `status: 'playing'`, emit `opponent_joined`
- [x] Từ chối người thứ 3: emit `room_full`
- [x] Fix: dùng `room.players` thay vì `room.length`

#### 2.2: Implement Socket Events đúng Contract (v6 spec)
- [x] Event `create_room` → `room_created` với `{ roomId, sessionToken }`
- [x] Event `join_room` → `joined` với `{ color, roomId, fen, sessionToken }` + `opponent_joined` cho người kia
- [x] Event `join_room` → `room_full` / `room_not_found` cho error
- [x] Event `make_move` → validate → broadcast `move_made` tới opponent
- [x] Từ chối nước đi nếu không phải lượt mình (`NOT_YOUR_TURN`)

#### 2.3: GameBoard — Auto-flip + Auto-promote
- [x] Pass `orientation` prop (white/black) vào `ChessBoard`
- [x] react-chessboard `boardOrientation={orientation}`
- [x] **Auto-promote Queen**: `chess.js` auto-promote 'q' khi đến hàng cuối
- [x] Player info hiển thị đúng bên: Đen (trái) | Trắng (phải)

**Estimated:** ~4–6 giờ

---

### Phase 3: Game Logic — Clock, PGN, Game Over

> **Mục tiêu:** Server-side clock 15+0 pause/resume, thu thập dữ liệu AI, game over detection đầy đủ

#### 3.1: Implement Clock Manager (Server-side)

- [ ] `initClock(roomId, timeControl)`: parse `15+0` → `{ initial: 900, increment: 0 }`
- [ ] `startClock(roomId)`: bắt đầu `setInterval` 1 giây
- [ ] `pauseClock(roomId, side)`: dừng decrement cho bên `side`
- [ ] `resumeClock(roomId, side)`: tiếp tục decrement cho bên `side`
- [ ] `switchClock(roomId)`: dừng bên hiện tại, chạy bên kia, record `timeSpent` vào `room.clockTimes[]`
- [ ] `pauseAllClock(roomId)`: pause cả 2 bên (khi cả 2 disconnect)
- [ ] `getTimes(roomId)`: trả `{ whiteTime, blackTime }`
- [ ] `getActiveSide(roomId)`: trả `'white' | 'black' | null`
- [ ] Khi đồng hồ về 0 → callback `'timeout'` → trigger `handleGameOver`
- [ ] `stopClock(roomId)`: dọn dẹp interval
- [ ] `resetClock(roomId)`: reset về initial

#### 3.2: Wire Clock vào Game Flow
- [ ] Khi opponent join → `initClock` + `startClock` (Trắng bắt đầu)
- [ ] Mỗi `make_move` → `switchClock` → record `timeSpent`
- [ ] Broadcast `clock_update` mỗi 1 giây tới cả 2 client (kèm `activeSide`)
- [ ] Khi `game_over` → `stopClock`
- [ ] Khi disconnect → `pauseClock(roomId, disconnectedColor)`



- [ ] Khi reconnect → `resumeClock(roomId, reconnectedColor)`

#### 3.3: Implement GameOver Detection (Server-side)
- [ ] Dùng chess.js kiểm tra `isCheckmate()`, `isStalemate()`
- [ ] `handleResign(color)`: bên color bấm → bên kia thắng
- [ ] `handleTimeout(roomId, color)`: trigger khi clock về 0
- [ ] Sau khi detect game over → gọi `handleGameOver(room, result, reason, io)`
- [ ] Đặt `finishedAt = Date.now()` để tính room cleanup timer

#### 3.4: Implement PGN Collection
- [ ] Mỗi nước đi → lưu `san` (VD: "e4", "Nf3", "O-O") vào `room.moves[]`
- [ ] Implement `buildPGN(moves[])`: "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
- [ ] Verify PGN có thể parse ngược bằng chess.js

#### 3.5: Frontend Clock Component (Server-sync)
- [ ] `GameClock.js`: props `{ whiteTime, blackTime, activeSide }`
- [ ] `activeSide`: `'white' | 'black' | null`
- [ ] Frontend suy ra status từ context:
  - `activeSide === 'white'` → Đen: paused, Trắng: running
  - `activeSide === 'black'` → Trắng: paused, Đen: running
  - Opponent disconnect → opponent: paused
- [ ] Format `MM:SS` (dùng `Math.floor(time / 60)`, `time % 60`)
- [ ] Running: màu cam. Paused: màu xám. Stopped: màu đỏ.
- [ ] Nhận `clock_update` từ server mỗi giây (payload có `activeSide`)

#### 3.6: Frontend Move History Component (SAN)
- [ ] `MoveHistory.js`: props `{ moves: string[] }`
- [ ] Format 2 cột: số nước | Trắng | Đen
- [ ] VD: "1. e4    e5" trên cùng 1 dòng
- [ ] Auto-scroll xuống cuối khi có nước mới

#### 3.7: Frontend — Resign + Exit Buttons
- [ ] Nút "Xin Thua" → emit `resign` event lên server
- [ ] Nút "Thoát Phòng" → emit `exit_room` event
- [ ] Server: `exit_room` khi đang `waiting` → xóa phòng. Khi đang `playing` → xử lý như resign.

#### 3.8: Waiting Overlay
- [ ] `WaitingOverlay.js`: hiện khi đã join nhưng `status === 'waiting'`
- [ ] Hiển thị link phòng để copy + nút "Copy Link"
- [ ] Có nút "Thoát Phòng"

**Estimated:** ~10–12 giờ

---

### Phase 4: Disconnect / Reconnect Handling

> **Mục tiêu:** Giữ room state, clock pause/resume, session token restore

#### 4.1: Handle Disconnect (Server-side)
- [ ] `socket.on('disconnect')`: lookup room by socketId → tìm color
- [ ] Set `room.players[color] = null`
- [ ] Call `pauseClock(roomId, color)` — clock của bên đó dừng
- [ ] Check: nếu `room.players.white === null && room.players.black === null`:
  - Cả 2 disconnect → đặt cleanup timer 30 giây
- [ ] Else: 1 người còn → emit `opponent_disconnected` tới player còn lại

#### 4.2: Handle Reconnect (Server-side)
- [ ] `socket.on('reconnect', { roomId, sessionToken })`: lookup `disconnectedSessions`
- [ ] Match sessionToken → get `roomId` và `color`
- [ ] Verify room còn tồn tại → restore
- [ ] Update `room.players[color] = socket.id`
- [ ] Call `resumeClock(roomId, color)` — clock của bên đó tiếp tục
- [ ] Cancel cleanup timer nếu có
- [ ] Emit `reconnected` tới player với full game state
- [ ] Emit `opponent_reconnected` tới opponent
- [ ] Xóa khỏi `disconnectedSessions`

#### 4.3: Handle Reconnect (Client-side)
- [ ] Socket.IO auto-reconnect: `socket.io.on('reconnect_attempt', ...)`
- [ ] On reconnect: check localStorage for `sessionToken`, emit `reconnect`
- [ ] Khi nhận `reconnected`: restore game state (fen, moves, clock, etc.)
- [ ] Khi nhận `opponent_reconnected`: tắt DisconnectedOverlay

#### 4.4: Frontend Disconnected Overlay
- [ ] `DisconnectedOverlay.js`: hiện khi nhận `opponent_disconnected`
- [ ] Text: "Đối thủ đã disconnect. Đang chờ reconnect..."
- [ ] Đồng hồ opponent hiển thị trạng thái PAUSED

#### 4.5: Frontend Reconnecting Overlay
- [ ] `ReconnectingOverlay.js`: hiện khi socket đang reconnect (Socket.IO `reconnect_attempt`)
- [ ] Text: "Mất kết nối. Đang kết nối lại..."

#### 4.6: Room Cleanup System
- [ ] `roomCleanupTimers` Map: `roomId → setTimeoutId`
- [ ] `scheduleRoomCleanup(roomId, delayMs)`: đặt timer → `cleanupRoom(roomId)`
- [ ] `cancelCleanupTimer(roomId)`: clear timer nếu có
- [ ] `cleanupRoom(roomId)`: xóa phòng khỏi `rooms`, xóa khỏi `disconnectedSessions`, stop clock
- [ ] Sau `game_over`: `scheduleFinishedRoomCleanup(roomId)` → 5 phút
- [ ] Khi cả 2 disconnect (Phase 4.1): `scheduleRoomCleanup(roomId, 30000)` → 30 giây
- [ ] Khi `waiting` timeout: `scheduleRoomCleanup(roomId, 300000)` → 5 phút

#### 4.7: Timeout khi Disconnect
- [ ] Nếu clock hết giờ trong lúc opponent disconnect → vẫn trigger `game_over` bình thường
- [ ] Server xử lý timeout như bình thường, opponent vẫn nhận `game_over` (nếu đang online)

**Estimated:** ~6–8 giờ

---

### Phase 5: AI Integration — Kết nối AI Engine + Result Modal

> **Mục tiêu:** Kết nối thực sự với MMD-G2, hiện ELO + Explanation

#### 5.1: Implement AI Client (Server-side)
- [ ] `aiClient.js`: HTTP POST dùng native `fetch`
- [ ] Build request body: `{ pgn, clock_times, result, time_control: "15+0" }`
- [ ] Env var `AI_ENGINE_URL` (default: `http://localhost:8000`)
- [ ] Timeout 30 giây dùng `AbortSignal.timeout(30000)`
- [ ] Parse response, validate `success` field
- [ ] Return `null` nếu lỗi (để caller xử lý fallback)

#### 5.2: Wire AI vào Game Over Flow
- [ ] `handleGameOver(room, result, reason, io)`:
  1. Emit `game_over` + `ai_loading` tới cả 2 client
  2. `buildPGN` + lấy `clockTimes`
  3. `await requestELOPrediction(room)`
  4. Nếu thành công → emit `ai_result`
  5. Nếu lỗi → emit `ai_error` với message fallback
  6. Đặt `finishedAt = Date.now()` cho cleanup timer

#### 5.3: Frontend AI Loading Overlay
- [ ] State: `aiLoading: boolean`
- [ ] Khi nhận `ai_loading` → set `aiLoading = true`
- [ ] Overlay toàn màn hình: spinner + "Đang phân tích ELO bằng AI..."
- [ ] Khi nhận `ai_result` hoặc `ai_error` → tắt overlay

#### 5.4: Frontend Result Modal
- [ ] `ResultModal.js`:
  - Header: kết quả ("Trắng Thắng / Đen Thắng / Hòa") + reason
  - 2 ô ELO lớn: Trắng | Đen
  - Stats: ECO code + name, CPL, Blunders
  - Explanation: scrollable text box
  - Nút `[ Chơi Lại ]`
- [ ] State từ `GameRoom`: `aiData`, `aiError`, `gameOverData`
- [ ] Render modal khi `aiData` hoặc `aiError` có giá trị

#### 5.5: Handle AI Error Fallback
- [ ] Khi nhận `ai_error`: hiện modal với kết quả cơ bản (Thắng/Thua/Hòa)
- [ ] Message: "Không thể kết nối AI Engine. Phân tích ELO tạm thời không khả dụng."
- [ ] Vẫn cho bấm "Chơi Lại" bình thường

**Estimated:** ~4–6 giờ

---

### Phase 6: Polish & QA — Hoàn thiện & Testing

> **Mục tiêu:** Tất cả checklist nghiệm thu pass, UX tốt

#### 6.1: Play Again Feature
- [ ] Server: emit `reset_game` tới cả 2 client khi bấm
- [ ] Frontend: reset `Game` state (chess instance, FEN, moves, clockTimes)
- [ ] Reset clock: `resetClock(roomId)` → `startClock(roomId)`
- [ ] Xóa `result`, `resultReason` khỏi room state, reset `finishedAt`
- [ ] Ẩn Result Modal

#### 6.2: UX Polish
- [ ] Theme tối (dark mode) cho Lobby và Game (MUI theme)
- [ ] **Check indicator**: dùng `chess.inCheck()` để detect. react-chessboard v4.5.0 không có built-in → custom overlay CSS:
  - Tính vị trí vua: `chess.board().king(chess.turn())`
  - Overlay position absolute hoặc custom square style để highlight ô vua màu đỏ
- [ ] Copy link button trong WaitingOverlay (dùng `navigator.clipboard.writeText`)

#### 6.3: Debug Logging
- [ ] Log mỗi nước đi: `{ move, timeSpent, fen }`
- [ ] Log khi `game_over`: `{ result, reason, pgn, clockTimes.length }`
- [ ] Log AI request/response (success và error)
- [ ] Log disconnect/reconnect events với session token
- [ ] Log room cleanup: khi nào phòng bị xóa, lý do
- [ ] Console đủ để debug mà không cần breakpoint

#### 6.4: Verify Server Port 3000
- [ ] Đổi server listen port: `8080` → `3000`
- [ ] Verify: chạy `npm start` → server listen on port 3000

#### 6.5: Checklist Nghiệm Thu
- [ ] Tất cả 23 items trong spec Requirements Section "Success Criteria" pass

**Estimated:** ~3–4 giờ

---

## Dependencies

### External Dependencies
- **AI Engine Server** (repo MMD-G2) phải chạy trên cổng 8000 để test end-to-end
- Hoặc AI Client cần handle graceful degradation khi AI Engine chưa có

### Internal Dependencies (Phase ordering)
```
Phase 1 (Foundation)
    ↓
Phase 2 (Multiplayer Core) ← cần Phase 1 xong
    ↓
Phase 3 (Game Logic) ← cần Phase 2 socket events đúng
    ↓
Phase 4 (Disconnect/Reconnect) ← cần Phase 2 (Room Manager) + Phase 3 (Clock)
    ↓
Phase 5 (AI Integration) ← cần Phase 3 (PGN collection)
    ↓
Phase 6 (Polish & QA)
```

### Trong Phase
- 1.3 (React Router) → 1.2 (Xóa auth): thứ tự tự do
- 1.5 (Session token) → 2.1 (Room Manager): token infrastructure trước
- 2.3 (Board flip + promote) → 2.2 (Socket events): tự do
- 3.5 (Clock UI) → 3.1 (Clock Manager): Clock Manager trước
- 4.2 (Reconnect server) → 4.3 (Reconnect client): server trước
- 5.3+5.4 (UI) → 5.2 (Wire AI): Wire AI trước
- 6.2 (UX) → 6.1 (Play Again): tự do

---

## Timeline & Estimates

| Phase | Tên | Estimate | Ghi chú |
|-------|-----|---------|---------|
| P1 | Foundation | 4–5 giờ | Fix bug, xóa auth, setup Router, session token |
| P2 | Multiplayer Core | 4–6 giờ | Socket events + Board |
| P3 | Game Logic | 10–12 giờ | Clock pause/resume + PGN + game over (nhiều nhất) |
| P4 | Disconnect/Reconnect | 6–8 giờ | Session token + clock pause/resume + cleanup |
| P5 | AI Integration | 4–6 giờ | AI client + Result Modal |
| P6 | Polish & QA | 3–4 giờ | Play Again + debug + checklist |
| **Total** | | **31–41 giờ** | ~1.5 tuần làm việc (6–8 ngày) |

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AI Engine chưa chạy được khi test | Cao | Thấp | AI Client handle null return + fallback UI đã implement |
| Clock drift (server vs client) | Thấp | Trung bình | Server-side clock + sync 1 giây |
| Socket event mismatch (server vs client) | Trung bình | Cao | Đặt constant/event name ở 1 chỗ, import chung |
| Session token collision | Thấp | Cao | Dùng UUID v4 (2^122 combinations) |
| Reconnect không restore đúng state | Trung bình | Cao | Test kỹ: disconnect giữa ván, reconnect giữa ván, reconnect sau game_over |
| PGN format không parse được bởi AI | Thấp | Cao | Test với chess.js parse ngược sau mỗi build |
| Room cleanup chạy quá sớm | Trung bình | Cao | Cancel cleanup timer ngay khi có reconnect hoặc action |
| Clock pause/resume logic phức tạp | Trung bình | Trung bình | Vẽ state diagram trước khi code |

---

## Resources Needed

| Resource | Chi tiết |
|----------|----------|
| **Team** | 1 web developer |
| **AI Engine URL** | `http://localhost:8000` (hoặc cấu hình qua env `AI_ENGINE_URL`) |
| **Node.js** | >= 18 (cho native `fetch`) |
| **NPM packages** | react-router-dom@6, concurrently (mới) — others đã có |
| **Tools** | 2 trình duyệt (hoặc 1 trình duyệt + 1 incognito) để test multiplayer |
| **AI Engine** | Repo MMD-G2 chạy `uvicorn src.ai_engine.main:app --port 8000` |
