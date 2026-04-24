---
phase: planning
title: Game Server Chess Multiplayer - Task Breakdown
description: Kế hoạch triển khai chi tiết cho module Game Server với các milestone và task breakdown.
---

# Planning: Game Server Chess Multiplayer

## Kiến trúc hiện tại

Module game server sử dụng **FastAPI + WebSocket** (thay vì Socket.IO) với **HTML/JS frontend** được serve trực tiếp từ server (thay vì Streamlit).

```
src/game_server/
├── main.py              # FastAPI + WebSocket server + HTML frontend
├── rooms.py             # Room Manager (quản lý phòng)
├── chess_engine.py      # Chess Engine (python-chess wrapper)
├── clock.py             # Clock Service (đồng hồ thi đấu)
├── game.py              # Game Manager (điều phối game)
└── integration.py       # AI prediction mock + XAI explanation
```

Chạy server: `python -m src.game_server` hoặc `uvicorn src.game_server.main:app`

## Milestones

- [x] **M1: Foundation** — FastAPI + WebSocket, Room Manager, HTML Frontend ✅
- [x] **M2: Chess Logic** — python-chess integration, move validation, PGN export ✅
- [x] **M3: Real-time Sync** — WebSocket communication, clock sync ✅
- [x] **M4: AI Integration** — AI prediction mock, XAI explanation mock ✅
- [x] **M5: Testing** — Unit tests, integration tests ✅

---
