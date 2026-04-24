"""Game Server main - FastAPI với WebSocket support."""

import sys
import os

# Add src directory to path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.game_server.rooms import room_manager, Room, RoomStatus, PlayerColor
from src.game_server.integration import predict_elo, get_explanation

app = FastAPI(
    title="MMD-G2 Chess Game Server",
    description="WebSocket Chess Game với ELO Prediction",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Models ───────────────────────────────────────────────────────────────

class CreateRoomResponse(BaseModel):
    room_id: str
    url: str


# ─── HTML Frontend ────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MMD-G2 Chess Game</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; margin-bottom: 20px; color: #00d9ff; }
        
        /* Lobby */
        .lobby {
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-top: 50px;
        }
        .lobby input {
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            width: 250px;
        }
        .btn {
            background: #00d9ff;
            color: #000;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s;
        }
        .btn:hover { background: #00b8d9; transform: translateY(-2px); }
        .btn:disabled { background: #555; cursor: not-allowed; }
        .btn-secondary { background: #666; color: #fff; }
        
        /* Game Area */
        #gameArea { display: none; }
        .status {
            text-align: center;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            background: rgba(0,217,255,0.1);
        }
        .game-area {
            display: grid;
            grid-template-columns: 200px 480px 200px;
            gap: 20px;
            align-items: center;
            justify-content: center;
        }
        .player-info {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .player-info.white { border: 2px solid #fff; }
        .player-info.black { border: 2px solid #555; }
        .player-name { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
        .clock {
            font-size: 2.2em;
            font-family: 'Courier New', monospace;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
        }
        .clock.active { color: #00ff00; }
        .board-container { display: flex; justify-content: center; }
        #board { width: 480px; height: 480px; }
        
        /* Moves List */
        .moves-container {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 150px;
            overflow-y: auto;
        }
        .moves-list { font-family: monospace; font-size: 16px; line-height: 1.6; }
        
        /* Controls */
        .controls { text-align: center; margin-top: 20px; }
        
        /* Result Overlay */
        .result-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .result-overlay.show { display: flex; }
        .result-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00d9ff;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            max-width: 500px;
        }
        .result-card h2 { color: #00d9ff; margin-bottom: 25px; }
        .elo-display { display: flex; justify-content: space-around; margin: 20px 0; }
        .elo-box { padding: 20px; border-radius: 10px; min-width: 120px; }
        .elo-box.white { background: rgba(255,255,255,0.1); }
        .elo-box.black { background: rgba(0,0,0,0.4); }
        .elo-value { font-size: 2.5em; font-weight: bold; }
        .elo-label { font-size: 0.9em; color: #888; margin-top: 5px; }
        .explanation {
            text-align: left;
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        /* Turn Indicator */
        .turn-indicator {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1em;
        }
        .turn-indicator.my-turn { background: rgba(0,255,0,0.2); color: #00ff00; }
        .turn-indicator.waiting { background: rgba(255,255,0,0.1); color: #ffff00; }
        
        @media (max-width: 900px) {
            .game-area { grid-template-columns: 1fr; }
            #board { width: 320px; height: 320px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chess Game</h1>

        <!-- Lobby -->
        <div id="lobby" class="lobby">
            <button id="btnCreateRoom" class="btn">Tạo Phòng Mới</button>
            <br><br>
            <span>hoặc nhập mã phòng:</span>
            <br><br>
            <input type="text" id="roomInput" placeholder="Mã phòng (VD: abc12345)">
            <button id="btnJoinRoom" class="btn btn-secondary">Vào Phòng</button>
        </div>

        <!-- Game Area -->
        <div id="gameArea">
            <div class="status">
                <strong>Phòng:</strong> <span id="roomId"></span>
                <strong> | Màu của bạn:</strong> <span id="myColor" style="font-weight: bold;"></span>
            </div>

            <div id="turnIndicator" class="turn-indicator waiting">Đang chờ...</div>

            <div class="game-area">
                <div class="player-info black">
                    <div class="player-name"><span id="blackName">ĐEN</span></div>
                    <div id="blackClock" class="clock">05:00</div>
                </div>

                <div class="board-container">
                    <div id="board"></div>
                </div>

                <div class="player-info white">
                    <div class="player-name"><span id="whiteName">TRẮNG</span></div>
                    <div id="whiteClock" class="clock">05:00</div>
                </div>
            </div>

            <div class="controls">
                <button class="btn btn-secondary" onclick="resign()">Chấp Nhận Thua</button>
            </div>

            <div class="moves-container">
                <strong>Các nước đi:</strong>
                <div id="movesDisplay" class="moves-list">Chưa có nước đi</div>
            </div>
        </div>
    </div>

    <!-- Result Overlay -->
    <div id="resultOverlay" class="result-overlay">
        <div class="result-card">
            <h2 id="resultTitle">Kết Quả</h2>
            <div class="elo-display">
                <div class="elo-box white">
                    <div class="elo-value" id="whiteElo">-</div>
                    <div class="elo-label">TRẮNG</div>
                </div>
                <div class="elo-box black">
                    <div class="elo-value" id="blackElo">-</div>
                    <div class="elo-label">ĐEN</div>
                </div>
            </div>
            <div id="explanation" class="explanation"></div>
            <br>
            <button class="btn" onclick="closeResult()">Đóng</button>
        </div>
    </div>

    <script>
    // ─── State ───────────────────────────────────────────────────────────────
    var ws = null;
    var board = null;
    var game = new Chess();
    var myColor = null;
    var currentRoom = null;
    var isMyTurn = false;
    var gameStarted = false;
    var lastMoveStartTime = null;

    // Clock state (seconds)
    var whiteTime = 300;
    var blackTime = 300;
    var clockInterval = null;

    // ─── Init ────────────────────────────────────────────────────────────────
    $(document).ready(function() {
        // Check URL for room ID
        var urlParams = new URLSearchParams(window.location.search);
        var roomFromUrl = urlParams.get('room');
        if (roomFromUrl) {
            document.getElementById('roomInput').value = roomFromUrl;
        }

        // Button events
        document.getElementById('btnCreateRoom').onclick = createRoom;
        document.getElementById('btnJoinRoom').onclick = joinRoom;
    });

    // ─── Room Functions ───────────────────────────────────────────────────────
    function createRoom() {
        console.log('[Lobby] Creating room...');
        fetch('/api/create_room', { method: 'POST' })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                console.log('[Lobby] Room created:', data);
                currentRoom = data.room_id;
                document.getElementById('roomInput').value = data.room_id;
                // Navigate to room
                window.location.href = '/?room=' + data.room_id;
            })
            .catch(function(err) {
                console.error('[Lobby] Error:', err);
                alert('Lỗi tạo phòng');
            });
    }

    function joinRoom() {
        var roomId = document.getElementById('roomInput').value.trim();
        if (!roomId) {
            alert('Vui lòng nhập mã phòng');
            return;
        }
        console.log('[Lobby] Joining room:', roomId);
        window.location.href = '/?room=' + roomId;
    }

    // ─── WebSocket ───────────────────────────────────────────────────────────
    function connectWebSocket(roomId) {
        console.log('[WS] Connecting to room:', roomId);
        var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(protocol + '//' + location.host + '/ws/' + roomId);

        ws.onopen = function() {
            console.log('[WS] Connected');
            ws.send(JSON.stringify({ type: 'join' }));
        };

        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);
            console.log('[WS] Received:', data.type, data);
            handleMessage(data);
        };

        ws.onclose = function() {
            console.log('[WS] Disconnected');
        };

        ws.onerror = function(err) {
            console.error('[WS] Error:', err);
        };
    }

    // ─── Message Handler ──────────────────────────────────────────────────────
    function handleMessage(data) {
        switch (data.type) {
            case 'joined':
                // Player joined
                myColor = data.color;
                currentRoom = data.room_id;
                
                // Show game area
                document.getElementById('lobby').style.display = 'none';
                document.getElementById('gameArea').style.display = 'block';
                document.getElementById('roomId').textContent = currentRoom;
                document.getElementById('myColor').textContent = 
                    myColor === 'white' ? 'TRANG' : 'DEN';
                
                // Update player names
                if (myColor === 'white') {
                    document.getElementById('whiteName').textContent = 'BAN (Trang)';
                    document.getElementById('blackName').textContent = 'DOI THU';
                } else {
                    document.getElementById('blackName').textContent = 'BAN (Den)';
                    document.getElementById('whiteName').textContent = 'DOI THU';
                }
                
                // Initialize board
                game.load(data.fen);
                initBoard();
                
                // Update turn state
                updateTurnState(data.turn);
                
                // Check if game already has opponent
                if (data.has_opponent) {
                    gameStarted = true;
                    startClock();
                    updateTurnDisplay();
                }
                break;

            case 'opponent_joined':
                // Second player joined - game starts!
                alert('Doi thu da tham gia! Tran dau bat dau!');
                gameStarted = true;
                startClock();
                updateTurnDisplay();
                break;

            case 'move':
                // Opponent made a move
                game.load(data.fen);
                board.position(game.fen());
                addMoveToDisplay(data.move);
                updateTurnState(data.turn);
                break;

            case 'game_over':
                gameStarted = false;
                stopClock();
                showResult(data);
                break;

            case 'error':
                alert('Loi: ' + data.message);
                break;
        }
    }

    // ─── Turn State ──────────────────────────────────────────────────────────
    function updateTurnState(serverTurn) {
        // serverTurn is 'white' or 'black'
        isMyTurn = (serverTurn === myColor);
        console.log('[Turn] serverTurn:', serverTurn, '| myColor:', myColor, '| isMyTurn:', isMyTurn);
        updateTurnDisplay();
    }

    function updateTurnDisplay() {
        var indicator = document.getElementById('turnIndicator');
        if (!gameStarted) {
            indicator.textContent = 'Cho doi doi thu...';
            indicator.className = 'turn-indicator waiting';
            return;
        }
        
        if (isMyTurn) {
            indicator.textContent = 'Den luot ban!';
            indicator.className = 'turn-indicator my-turn';
        } else {
            indicator.textContent = 'Dang cho doi thu...';
            indicator.className = 'turn-indicator waiting';
        }
    }

    // ─── Board ───────────────────────────────────────────────────────────────
    function initBoard() {
        board = ChessBoard('board', {
            position: 'start',
            draggable: true,
            onDrop: handleMove,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
            orientation: myColor === 'black' ? 'black' : 'white'
        });
    }

    function handleMove(source, target) {
        console.log('[Move] Attempting:', source, '->', target);
        console.log('[Move] isMyTurn:', isMyTurn, '| gameStarted:', gameStarted);
        
        if (!isMyTurn || !gameStarted) {
            console.log('[Move] Rejected - not your turn');
            board.position(game.fen());
            return;
        }

        // Calculate time spent
        var now = Date.now();
        var timeSpent = lastMoveStartTime ? (now - lastMoveStartTime) / 1000 : 0;
        lastMoveStartTime = now;

        // Try move
        var move = game.move({ from: source, to: target, promotion: 'q' });
        if (move === null) {
            console.log('[Move] Invalid');
            board.position(game.fen());
            return;
        }

        console.log('[Move] Valid:', move.san, '| Time:', timeSpent.toFixed(2), 's');

        // Send to server
        ws.send(JSON.stringify({
            type: 'move',
            move: move.san,
            fen: game.fen(),
            clock_time: Math.round(timeSpent * 100) / 100
        }));

        // Update UI
        addMoveToDisplay(move.san);
        isMyTurn = false;
        updateTurnDisplay();
    }

    function addMoveToDisplay(move) {
        var display = document.getElementById('movesDisplay');
        if (display.textContent === 'Chua co nuoc di') {
            display.textContent = move;
        } else {
            display.textContent += ' ' + move;
        }
    }

    // ─── Clock ────────────────────────────────────────────────────────────────
    function startClock() {
        whiteTime = 300;
        blackTime = 300;
        lastMoveStartTime = Date.now();
        updateClockDisplay();

        clockInterval = setInterval(function() {
            if (!gameStarted) return;

            if (game.turn() === 'w') {
                whiteTime -= 0.1;
                document.getElementById('whiteClock').classList.add('active');
                document.getElementById('blackClock').classList.remove('active');
            } else {
                blackTime -= 0.1;
                document.getElementById('blackClock').classList.add('active');
                document.getElementById('whiteClock').classList.remove('active');
            }

            updateClockDisplay();

            if (whiteTime <= 0) {
                endGame('0-1', 'Het gio - Trang thua');
            } else if (blackTime <= 0) {
                endGame('1-0', 'Het gio - Den thua');
            }
        }, 100);
    }

    function stopClock() {
        if (clockInterval) {
            clearInterval(clockInterval);
            clockInterval = null;
        }
    }

    function updateClockDisplay() {
        document.getElementById('whiteClock').textContent = formatTime(whiteTime);
        document.getElementById('blackClock').textContent = formatTime(blackTime);
    }

    function formatTime(seconds) {
        var m = Math.floor(Math.max(0, seconds) / 60);
        var s = Math.floor(Math.max(0, seconds) % 60);
        return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
    }

    // ─── Game Flow ───────────────────────────────────────────────────────────
    function resign() {
        if (!confirm('Ban co that su muon chap nhan thua?')) return;
        var result = myColor === 'white' ? '0-1' : '1-0';
        endGame(result, 'Dau hang');
    }

    function endGame(result, reason) {
        if (!gameStarted && reason !== 'Dau hang') return;
        
        gameStarted = false;
        stopClock();

        console.log('[Game] End - Result:', result, '| Reason:', reason);

        ws.send(JSON.stringify({
            type: 'game_over',
            result: result,
            pgn: game.pgn()
        }));
    }

    function showResult(data) {
        document.getElementById('whiteElo').textContent = data.white_elo || '-';
        document.getElementById('blackElo').textContent = data.black_elo || '-';
        document.getElementById('explanation').innerHTML = data.explanation || '';

        var resultText = 'Ket thuc';
        if (data.result === '1-0') resultText = 'TRANG THANG!';
        else if (data.result === '0-1') resultText = 'DEN THANG!';
        else if (data.result === '1/2-1/2') resultText = 'HOA!';
        document.getElementById('resultTitle').textContent = resultText;

        document.getElementById('resultOverlay').classList.add('show');
    }

    function closeResult() {
        document.getElementById('resultOverlay').classList.remove('show');
    }

    // ─── Page Load - Connect if room in URL ──────────────────────────────────
    (function initFromURL() {
        var urlParams = new URLSearchParams(window.location.search);
        var roomId = urlParams.get('room');
        if (roomId) {
            console.log('[Init] Room from URL:', roomId);
            currentRoom = roomId;
            document.getElementById('roomInput').value = roomId;
            
            // Show game area immediately
            document.getElementById('lobby').style.display = 'none';
            document.getElementById('gameArea').style.display = 'block';
            document.getElementById('roomId').textContent = roomId;
            
            // Init board first (empty)
            initBoard();
            
            // Then connect WebSocket
            connectWebSocket(roomId);
        }
    })();
    </script>
</body>
</html>
"""


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/create_room", response_model=CreateRoomResponse)
async def create_room():
    """Tao phong choi moi."""
    room = room_manager.create_room()
    return CreateRoomResponse(
        room_id=room.id,
        url=f"/?room={room.id}"
    )


@app.get("/")
async def root():
    """Tra ve trang HTML."""
    return HTMLResponse(content=HTML_PAGE, media_type="text/html")


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """WebSocket endpoint cho multiplayer chess."""
    
    room = room_manager.get_room(room_id)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    await websocket.accept()

    player_color = None
    player_id = str(id(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            print(f"[WS] Player {player_id} sent: {msg_type}")

            # ── JOIN ──────────────────────────────────────────────────────
            if msg_type == "join":
                # Assign player to room
                player_color = room.waiting_player or "white"
                print(f"[WS] Assigned color: {player_color}")

                if player_color == "white":
                    room.white_player = player_id
                    room.websocket_white = websocket
                else:
                    room.black_player = player_id
                    room.websocket_black = websocket

                room.status = "playing" if room.is_full() else "waiting"

                # Determine turn from FEN
                fen_turn = room.fen.split()[1]
                turn_str = "white" if fen_turn == "w" else "black"

                # Send confirmation to joining player
                await websocket.send_json({
                    "type": "joined",
                    "color": player_color,
                    "room_id": room.id,
                    "fen": room.fen,
                    "turn": turn_str,
                    "has_opponent": room.is_full(),
                })

                # Notify existing player about new opponent
                if room.is_full():
                    # Tell WHITE player
                    if room.websocket_white and player_color == "black":
                        await room.websocket_white.send_json({
                            "type": "opponent_joined"
                        })
                    # Tell BLACK player
                    if room.websocket_black and player_color == "white":
                        await room.websocket_black.send_json({
                            "type": "opponent_joined"
                        })

            # ── MOVE ─────────────────────────────────────────────────────
            elif msg_type == "move":
                # Validate turn
                fen_turn = room.fen.split()[1]
                expected_color = "white" if fen_turn == "w" else "black"

                if player_color != expected_color:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Khong phai luot ban! (dang la luot {expected_color})"
                    })
                    continue

                move_san = data.get("move", "")
                move_fen = data.get("fen", "")
                clock_time = data.get("clock_time", 0)

                # Determine next turn
                next_fen_turn = move_fen.split()[1]
                next_turn = "white" if next_fen_turn == "w" else "black"

                # Broadcast to opponent
                opponent_ws = room.websocket_black if player_color == "white" else room.websocket_white
                if opponent_ws:
                    await opponent_ws.send_json({
                        "type": "move",
                        "move": move_san,
                        "fen": move_fen,
                        "turn": next_turn,
                        "clock_time": clock_time,
                    })

                # Update server state
                room.fen = move_fen
                room.moves.append(move_san)
                room.clock_times.append(clock_time)

            # ── GAME OVER ────────────────────────────────────────────────
            elif msg_type == "game_over":
                room.status = "finished"

                # Get ELO prediction
                elo_result = predict_elo(data.get("pgn", ""), list(room.clock_times))
                explanation = get_explanation(elo_result)

                # Broadcast to both players
                for ws_player in [room.websocket_white, room.websocket_black]:
                    if ws_player:
                        await ws_player.send_json({
                            "type": "game_over",
                            "result": data.get("result", "*"),
                            "pgn": data.get("pgn", ""),
                            "clock_times": list(room.clock_times),
                            "white_elo": elo_result["white_elo"],
                            "black_elo": elo_result["black_elo"],
                            "stats": elo_result["stats"],
                            "explanation": explanation,
                        })
                break

    except WebSocketDisconnect:
        print(f"[WS] Player {player_color} disconnected")
        if player_color == "white":
            room.white_player = None
            room.websocket_white = None
        else:
            room.black_player = None
            room.websocket_black = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
