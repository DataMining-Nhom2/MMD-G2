---
phase: implementation
title: Game Server Chess Multiplayer - Implementation Guide
description: Hướng dẫn triển khai chi tiết, patterns và best practices cho module Game Server với FastAPI + WebSocket.
---

# Implementation Guide: Game Server Chess Multiplayer

## Development Setup

### Prerequisites
- Python 3.11+
- conda environment `MMDS` đã activate

### Environment Setup

```bash
# Clone/Copy project
cd /home/sakana/Code/PTIT/MMDs/MMD-G2

# Tạo/activate conda environment
conda activate MMDS

# Cài đặt dependencies
pip install fastapi uvicorn python-chess pydantic websockets
```

### Project Structure

```
MMD-G2/
└── src/
    └── game_server/
        ├── __init__.py
        ├── main.py              # FastAPI + WebSocket server + HTML frontend
        ├── rooms.py             # Room Manager (quản lý phòng)
        ├── chess_engine.py      # Chess Engine (python-chess wrapper)
        ├── clock.py             # Clock Service (đồng hồ thi đấu)
        ├── game.py              # Game Manager (điều phối game)
        └── integration.py       # AI/XAI mock interfaces
```

### Running the Application

```bash
conda activate MMDS
cd /home/sakana/Code/PTIT/MMDs/MMD-G2

# Chạy server (FastAPI + WebSocket + HTML frontend)
python -m src.game_server

# Hoặc dùng uvicorn
uvicorn src.game_server.main:app --host 0.0.0.0 --port 8000
```

---

## Code Structure & Patterns

### Backend Patterns

#### 1. FastAPI + Socket.IO Server Setup

```python
# backend/server.py
import socketio
import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from socketio import ASGIApp

# Tạo Socket.IO async server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*'
)

# Tạo FastAPI app
app = FastAPI(title="MMD-G2 Game Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Health check
@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.get('/')
async def root():
    return {'message': 'MMD-G2 Game Server', 'version': '1.0.0'}

# Mount Socket.IO as ASGI app
socket_app = ASGIApp(sio, app)

# Import và đăng ký event handlers
from backend.game import room, game as game_module

# Socket.IO event handlers được định nghĩa trong các module riêng
```

#### 2. Room Manager với Room Code Generation

```python
# backend/game/room.py
import random
import string
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class RoomStatus(str, Enum):
    WAITING = "waiting"
    PLAYING = "playing"
    FINISHED = "finished"

class PlayerColor(str, Enum):
    WHITE = "white"
    BLACK = "black"

class Room(BaseModel):
    room_code: str
    status: RoomStatus = RoomStatus.WAITING
    white_player: Optional[str] = None  # Socket ID
    black_player: Optional[str] = None
    created_at: datetime = datetime.now()
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves: list[str] = []
    clock_white: float = 900.0
    clock_black: float = 900.0
    clock_times: list[float] = []
    time_control: str = "15+0"
    current_turn: PlayerColor = PlayerColor.WHITE
    game_result: Optional[str] = None

class RoomManager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def generate_room_code(self) -> str:
        """Tạo mã phòng 6 ký tự alphanumeric."""
        chars = string.ascii_uppercase + string.digits
        while True:
            code = ''.join(random.choices(chars, k=6))
            if code not in self.rooms:
                return code

    def create_room(self, time_control: str = "15+0") -> str:
        """Tạo phòng mới, trả về mã phòng."""
        room_code = self.generate_room_code()
        clock_seconds = self.parse_minutes(time_control)
        self.rooms[room_code] = Room(
            room_code=room_code,
            clock_white=clock_seconds,
            clock_black=clock_seconds,
            time_control=time_control
        )
        return room_code

    def join_room(self, room_code: str, sid: str) -> dict:
        """Tham gia phòng. Trả về kết quả."""
        if room_code not in self.rooms:
            return {'success': False, 'error': {'code': 'ROOM_NOT_FOUND', 'message': 'Phòng không tồn tại'}}

        room = self.rooms[room_code]

        if room.white_player and room.black_player:
            return {'success': False, 'error': {'code': 'ROOM_FULL', 'message': 'Phòng đã đầy'}}

        if room.status != RoomStatus.WAITING:
            return {'success': False, 'error': {'code': 'ROOM_PLAYING', 'message': 'Ván đấu đang diễn ra'}}

        # Gán màu
        if not room.white_player:
            room.white_player = sid
            color = PlayerColor.WHITE
        else:
            room.black_player = sid
            color = PlayerColor.BLACK
            room.status = RoomStatus.PLAYING

        return {
            'success': True,
            'color': color,
            'game_started': room.status == RoomStatus.PLAYING
        }

room_manager = RoomManager()
```

#### 3. Chess Engine Wrapper

```python
# backend/game/chess_engine.py
import chess

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()

    def load_fen(self, fen: str) -> bool:
        """Load position from FEN."""
        try:
            self.board = chess.Board(fen)
            return True
        except ValueError:
            return False

    def validate_move(self, move_san: str) -> bool:
        """Check if a move in SAN notation is legal."""
        try:
            move = self.board.parse_san(move_san)
            return self.board.is_legal(move)
        except ValueError:
            return False

    def make_move(self, move_san: str) -> dict:
        """Make a move and return result."""
        try:
            move = self.board.parse_san(move_san)
            if not self.board.is_legal(move):
                return {'success': False, 'error': 'Illegal move'}

            self.board.push(move)
            return {
                'success': True,
                'new_fen': self.board.fen(),
                'is_check': self.board.is_check(),
                'is_checkmate': self.board.is_checkmate(),
                'is_stalemate': self.board.is_stalemate(),
                'move_san': self.board.san(move)
            }
        except ValueError as e:
            return {'success': False, 'error': str(e)}

    def get_game_status(self) -> dict:
        """Kiểm tra trạng thái game."""
        if self.board.is_checkmate():
            winner = "black" if self.board.turn == chess.WHITE else "white"
            return {'is_over': True, 'result': winner, 'reason': 'checkmate'}
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return {'is_over': True, 'result': 'draw', 'reason': 'stalemate'}
        if self.board.is_fifty_moves():
            return {'is_over': True, 'result': 'draw', 'reason': 'fifty_moves'}
        return {'is_over': False, 'result': None, 'reason': None}

    def to_pgn(self) -> str:
        """Export moves as PGN."""
        pgn = chess.pgn.Game()
        node = pgn

        for i, move in enumerate(self.board.move_stack):
            if i == 0:
                node = pgn.add_variation(move)
            else:
                node = node.add_variation(move)

        # Add headers
        pgn.headers["Site"] = "MMD-G2 Local"
        pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")

        return str(pgn)
```

### Frontend Patterns

#### 1. Streamlit + Socket.IO Client

```python
# frontend/streamlit_app.py
import streamlit as st
import socketio

# Session state initialization
if 'room_code' not in st.session_state:
    st.session_state.room_code = None
if 'player_color' not in st.session_state:
    st.session_state.player_color = None
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'connected' not in st.session_state:
    st.session_state.connected = False

# Socket.IO client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    st.session_state.connected = True
    st.rerun()

@sio.on('disconnect')
def on_disconnect():
    st.session_state.connected = False
    st.rerun()

@sio.on('room_created')
def on_room_created(data):
    st.session_state.room_code = data['room_code']
    st.session_state.waiting = True
    st.rerun()

@sio.on('player_joined')
def on_player_joined(data):
    st.session_state.opponent_color = data['color']
    st.rerun()

@sio.on('game_started')
def on_game_started(data):
    st.session_state.game_started = True
    st.session_state.your_color = data['your_color']
    st.session_state.fen = data['fen']
    st.session_state.waiting = False
    st.rerun()

@sio.on('move_made')
def on_move_made(data):
    st.session_state.last_move = data
    st.session_state.fen = data['fen']
    st.rerun()

@sio.on('game_over')
def on_game_over(data):
    st.session_state.game_over = True
    st.session_state.game_result = data
    st.rerun()

@sio.on('error')
def on_error(data):
    st.error(f"{data['code']}: {data['message']}")

# Connect on load
if not sio.connected:
    try:
        sio.connect('http://localhost:8000')
    except:
        pass

def main():
    st.title("MMD-G2 Chess")

    if not st.session_state.room_code:
        show_lobby_ui()
    elif st.session_state.game_started:
        show_game_ui()
    else:
        show_waiting_ui()

if __name__ == "__main__":
    main()
```

#### 2. Chess Board Component (HTML/CSS)

```python
# frontend/components/chess_board.py
import streamlit as st

def render_chess_board(fen: str, interactive: bool = True, on_move=None):
    """
    Render a chess board from FEN string.

    Args:
        fen: FEN notation string
        interactive: If True, enable move input
        on_move: Callback function(move_san) when move is made

    Returns:
        str or None: Move in SAN notation if made
    """
    # Parse FEN để lấy pieces
    pieces = parse_fen(fen)

    # HTML cho bàn cờ
    html = """
    <style>
    .chess-board {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        width: 400px;
        height: 400px;
        border: 2px solid #333;
    }
    .square {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        cursor: pointer;
    }
    .white-square { background-color: #f0d9b5; }
    .black-square { background-color: #b58863; }
    .selected { background-color: #829769 !important; }
    .valid-move { background-color: #646f40 !important; }
    </style>
    <div class="chess-board">
    """

    # Render squares
    for rank in range(8, 0, -1):  # 8 to 1
        for file_idx, file in enumerate('abcdefgh'):
            square = f"{file}{rank}"
            is_white = (rank + file_idx) % 2 == 1
            piece = pieces.get(square, '')

            color_class = 'white-square' if is_white else 'black-square'
            html += f'<div class="square {color_class}" data-square="{square}">{piece}</div>'

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
```

---

## Implementation Notes

### Handling Special Chess Moves

#### Castling
```python
# python-chess tự handle, chỉ cần dùng SAN
board.parse_san("O-O")   # Kingside
board.parse_san("O-O-O") # Queenside
```

#### En Passant
```python
# Cần có flag trong FEN
# python-chess tự xử lý khi make move
```

#### Pawn Promotion
```python
# Khi pawn reach last rank, cần prompt promotion
# Backend: Trả về move với promotion piece
# Frontend: Hiển thị dialog chọn quân

# SAN format: e8=Q
if board.is_legal(move) and board.requires_promotion(move):
    promotion = chess.QUEEN  # Default hoặc cho user chọn
```

### Clock Implementation

```python
# backend/game/clock.py
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class ClockState:
    white_remaining: float
    black_remaining: float
    clock_times: list[float]
    current_turn: str
    last_move_time: float

def parse_time_control(tc: str) -> float:
    """Parse time control string like '15+0' to seconds."""
    minutes = int(tc.split('+')[0])
    return minutes * 60

def record_time(clock: ClockState, color: str) -> float:
    """Record thời gian suy nghĩ cho nước vừa đi."""
    elapsed = time.time() - clock.last_move_time
    clock.clock_times.append(elapsed)
    return elapsed

def update_clock(clock: ClockState, color: str, increment: float = 0) -> float:
    """Cập nhật clock sau khi đi nước."""
    elapsed = record_time(clock, color)

    # Trừ thời gian đã dùng + thêm increment
    if color == 'white':
        clock.white_remaining -= elapsed
        clock.white_remaining += increment
        clock.current_turn = 'black'
    else:
        clock.black_remaining -= elapsed
        clock.black_remaining += increment
        clock.current_turn = 'white'

    clock.last_move_time = time.time()

    # Không âm
    if clock.white_remaining < 0:
        clock.white_remaining = 0
    if clock.black_remaining < 0:
        clock.black_remaining = 0

    return elapsed
```

### Socket.IO Error Handling

```python
# backend/game/game.py
@sio.event
async def make_move(sid, data):
    room = room_manager.get_room_by_sid(sid)
    if not room:
        await sio.emit('error', {'code': 'NOT_IN_ROOM', 'message': 'Bạn không ở trong phòng'}, room=sid)
        return

    move = data.get('move')
    if not room_manager.is_your_turn(room, sid):
        await sio.emit('error', {'code': 'NOT_YOUR_TURN', 'message': 'Chưa đến lượt bạn'}, room=sid)
        return

    result = game_manager.handle_move(room, move)
    if not result['success']:
        await sio.emit('error', {'code': 'INVALID_MOVE', 'message': result['error']}, room=sid)
        return

    # Broadcast move to all players in room
    await sio.emit('move_made', result['data'], room=room.room_code)

    # Check game over
    if result['game_over']:
        await handle_game_over(room)
```

---

## Integration Points

### Calling AI Model

```python
# Khi game kết thúc
from src.game_server.integration import predict_elo, get_explanation

async def handle_game_over(room: Room):
    # 1. Export data
    pgn = chess_engine.to_pgn()
    clock_times = clock_state.clock_times

    # 2. Call AI (mock)
    elo_result = predict_elo(pgn, clock_times)

    # 3. Get explanation
    explanation = get_explanation(elo_result)

    # 4. Broadcast result
    await sio.emit('game_over', {
        'result': room.game_result,
        'reason': room.game_result_reason,
        'pgn': pgn,
        'clock_times': clock_times,
        'white_elo': elo_result['white_elo'],
        'black_elo': elo_result['black_elo'],
        'stats': elo_result['stats'],
        'explanation': explanation
    }, room=room.room_code)
```

### Mock Interface Contract

```python
# src/game_server/integration.py
# CONTRACT - ĐÂY LÀ INTERFACE

def predict_elo(pgn_string: str, clock_times: list[float]) -> dict:
    """
    Args:
        pgn_string: "1. e4 e5 2. Nf3 Nc6 ..." (SAN)
        clock_times: [5.2, 3.1, 12.0, ...] seconds

    Returns:
        {
            "white_elo": int,
            "black_elo": int,
            "stats": {
                "white_avg_cpl": float,
                "black_avg_cpl": float,
                "white_blunders": int,
                "black_blunders": int
            }
        }
    """
    pass

def get_explanation(prediction_result: dict) -> str:
    """
    Args:
        prediction_result: Output từ predict_elo()

    Returns:
        str: Lời giải thích bằng tiếng Việt
    """
    pass
```

---

## Error Handling

### Backend Errors

```python
# Error codes
ERROR_CODES = {
    'ROOM_NOT_FOUND': 'Phòng không tồn tại',
    'ROOM_FULL': 'Phòng đã đầy',
    'ROOM_PLAYING': 'Ván đấu đang diễn ra',
    'INVALID_MOVE': 'Nước đi không hợp lệ',
    'NOT_YOUR_TURN': 'Chưa đến lượt bạn',
    'INVALID_TIME_CONTROL': 'Thể thức thời gian không hợp lệ',
}

# Send error
await sio.emit('error', {'code': code, 'message': ERROR_CODES[code]}, room=sid)
```

### Frontend Errors

```python
@sio.on('error')
def on_error(data):
    st.error(f"{data['code']}: {data['message']}")
```

---

## Performance Considerations

### Socket.IO Message Size
- Giữ message nhỏ, chỉ gửi necessary data
- Không gửi full FEN mỗi move (chỉ gửi move + update)

### Clock Updates
- Frontend tự update display (countdown animation)
- Backend chỉ gửi snapshot khi có thay đổi (move made, timeout)

### Board State
- Frontend là source of truth cho display
- Backend là source of truth cho game logic
- Không cần sync full state liên tục

---

## Security Notes

### Input Validation
- Validate tất cả input với Pydantic
- Escape output khi hiển thị (XSS prevention)

### CORS
- Chỉ cho phép localhost origins (production: thay bằng specific origins)

---

## Socket.IO Client Testing

```javascript
// Test với socket.io-client
const socket = io('http://localhost:8000');

socket.on('connect', () => {
    console.log('Connected:', socket.id);
    socket.emit('create_room', { time_control: '15+0' });
});

socket.on('room_created', (data) => {
    console.log('Room created:', data.room_code);
    socket.emit('join_room', { room_code: data.room_code });
});

socket.on('game_started', (data) => {
    console.log('Game started!', data);
    socket.emit('make_move', { move: 'e4' });
});
```
