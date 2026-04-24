---
phase: testing
title: Game Server Chess Multiplayer - Testing Strategy
description: Chiến lược testing, test cases và acceptance criteria cho module Game Server với Socket.IO.
---

# Testing Strategy: Game Server Chess Multiplayer

## Test Coverage Goals

| Level | Target | Scope |
|-------|--------|-------|
| **Unit Tests** | 100% | Chess engine, clock service, room manager |
| **Integration Tests** | Critical paths | Socket.IO flows, game over flow |
| **E2E Tests** | Key user journeys | Create room → Play → See results |

---

## Unit Tests

### Chess Engine Tests

#### `test_validate_legal_moves`
- **Mục tiêu:** Kiểm tra nước đi hợp lệ được chấp nhận
- **Test cases:**
  - [ ] Pawn move forward one square
  - [ ] Pawn move forward two squares from start
  - [ ] Knight move in L shape
  - [ ] Bishop diagonal move
  - [ ] Rook horizontal/vertical move
  - [ ] Queen combined moves
  - [ ] King adjacent move

#### `test_validate_illegal_moves`
- **Mục tiêu:** Kiểm tra nước đi bất hợp lệ bị từ chối
- **Test cases:**
  - [ ] Move piece through other piece
  - [ ] Move to square occupied by own piece
  - [ ] Move when in check (that doesn't resolve check)
  - [ ] Move piece not on turn

#### `test_special_moves`
- **Mục tiêu:** Kiểm tra các nước đi đặc biệt của FIDE
- **Test cases:**
  - [ ] Castling kingside (O-O) - White
  - [ ] Castling queenside (O-O-O) - White
  - [ ] Castling kingside (O-O) - Black
  - [ ] Castling blocked by piece between
  - [ ] Castling blocked by check
  - [ ] Castling through attacked square
  - [ ] En passant capture
  - [ ] En passant not available after one move
  - [ ] Pawn promotion to Queen
  - [ ] Pawn promotion to other pieces (if supported)

#### `test_game_status_detection`
- **Mục tiêu:** Kiểm tra phát hiện kết thúc ván
- **Test cases:**
  - [ ] Checkmate detected correctly
  - [ ] Stalemate detected correctly
  - [ ] Insufficient material detected
  - [ ] Game not over when not in terminal state

#### `test_fen_parsing`
- **Mục tiêu:** Kiểm tra FEN handling
- **Test cases:**
  - [ ] Load valid FEN
  - [ ] Reject invalid FEN
  - [ ] Export FEN matches import
  - [ ] Handle turn correctly in FEN

#### `test_pgn_export`
- **Mục tiêu:** Kiểm tra PGN export
- **Test cases:**
  - [ ] Export simple game
  - [ ] Export game with headers
  - [ ] PGN parseable by chess.com parser
  - [ ] PGN valid with headers

---

### Clock Service Tests

#### `test_clock_initialization`
- **Mục tiêu:** Kiểm tra khởi tạo clock
- **Test cases:**
  - [ ] Parse time control "15+0"
  - [ ] Parse time control "10+0"
  - [ ] Parse time control "5+3"
  - [ ] Initialize both clocks with 900 seconds

#### `test_clock_recording`
- **Mục tiêu:** Kiểm tra ghi nhận thời gian
- **Test cases:**
  - [ ] Record time for white move
  - [ ] Record time for black move
  - [ ] Clock times list length matches moves
  - [ ] Time deducted from correct player

#### `test_clock_timeout`
- **Mục tiêu:** Kiểm tra hết giờ
- **Test cases:**
  - [ ] Detect white timeout
  - [ ] Detect black timeout
  - [ ] Time remaining never negative

#### `test_clock_increment`
- **Mục tiêu:** Kiểm tra increment sau mỗi nước
- **Test cases:**
  - [ ] Increment added after move
  - [ ] No increment for 15+0 time control

---

### Room Manager Tests

#### `test_room_creation`
- **Mục tiêu:** Kiểm tra tạo phòng
- **Test cases:**
  - [ ] Create room returns 6-char code
  - [ ] Room code is alphanumeric uppercase
  - [ ] Room has initial state (WAITING)
  - [ ] Room ID is unique

#### `test_room_join`
- **Mục tiêu:** Kiểm tra tham gia phòng
- **Test cases:**
  - [ ] First player joins as white
  - [ ] Second player joins as black
  - [ ] Cannot join full room
  - [ ] Cannot join non-existent room
  - [ ] Game starts when second player joins

#### `test_room_code_generation`
- **Mục tiêu:** Kiểm tra tạo mã phòng
- **Test cases:**
  - [ ] Code is 6 characters
  - [ ] Code is uppercase alphanumeric
  - [ ] No duplicate codes generated

---

### Socket.IO Integration Tests

#### `test_socket_connection`
- **Mục tiêu:** Kiểm tra kết nối Socket.IO
- **Test cases:**
  - [ ] Client can connect to server
  - [ ] Client receives connect event
  - [ ] Multiple clients can connect

#### `test_create_room_flow`
- **Mục tiêu:** Kiểm tra flow tạo phòng
- **Test cases:**
  - [ ] Client emits create_room
  - [ ] Server returns room_created with 6-char code
  - [ ] Client receives room_created event

#### `test_join_room_flow`
- **Mục tiêu:** Kiểm tra flow tham gia phòng
- **Test cases:**
  - [ ] Client joins existing room
  - [ ] Client receives player_joined
  - [ ] Both clients receive game_started

#### `test_move_flow`
- **Mục tiêu:** Kiểm tra flow nước đi
- **Test cases:**
  - [ ] Valid move accepted
  - [ ] Invalid move rejected with error
  - [ ] Both clients receive move_made

#### `test_game_over_flow`
- **Mục tiêu:** Kiểm tra flow kết thúc game
- **Test cases:**
  - [ ] Game over detected by checkmate
  - [ ] Game over detected by timeout
  - [ ] Both clients receive game_over
  - [ ] ELO and explanation included

---

## Integration Tests

### Integration: Create and Join Room

```
Test: test_create_join_room_flow
Steps:
1. Client A connects via Socket.IO
2. Client A emits create_room
3. Receive room_created with 6-char code
4. Client B connects via Socket.IO
5. Client B emits join_room with code
6. Both clients receive player_joined
7. Both clients receive game_started

Expected: Full flow completes without errors
```

### Integration: Move Flow

```
Test: test_move_flow
Precondition: Room with 2 players, game started

Steps:
1. Client A (white) emits make_move("e4")
2. Server validates move
3. Server updates room state
4. Both clients receive move_made
5. Board updated on both clients
6. Clock updated on both clients

Expected: Move synced to both clients
```

### Integration: Game Over by Checkmate

```
Test: test_game_over_checkmate
Precondition: Room with setup for checkmate

Steps:
1. Play moves leading to checkmate
2. Server detects checkmate
3. Both clients receive game_over
4. PGN and clock_times included
5. predict_elo called
6. get_explanation called
7. ELO and explanation sent to clients

Expected: Full game_over flow completes
```

### Integration: Invalid Move Rejection

```
Test: test_invalid_move_rejected
Precondition: Room with 2 players

Steps:
1. Client sends make_move with invalid move
2. Server validates and rejects
3. Client receives error event

Expected: Move rejected, error returned
```

---

## End-to-End Tests

### E2E: Full Game Session

```
Scenario: Two players play a complete game

Steps:
1. Player A creates room → receives 6-char code
2. Player B joins room via code
3. Game starts, white to move
4. Player A moves e4
5. Player B sees move, moves e5
6. Players alternate until checkmate
7. Both see ELO prediction
8. Both see explanation

Expected: Complete working game
```

### E2E: Clock Timeout

```
Scenario: Player loses on time

Steps:
1. Create room with 15+0 time control
2. Start game
3. Wait without moving until clock reaches 0
4. Other player wins by timeout

Expected: Game ends, timeout winner declared
```

---

## Test Data

### Sample FEN Positions
```python
# Starting position
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# After 1. e4
FEN_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Checkmate position (Scholars Mate)
SCHOLARS_MATE = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
```

### Sample PGN
```python
SAMPLE_PGN = """
[Event "Test Game"]
[Site "MMD-G2 Local"]
[Date "2024.01.01"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""
```

### Sample Clock Times
```python
SAMPLE_CLOCK_TIMES = [
    5.2,   # White move 1
    3.1,   # Black move 1
    12.0,  # White move 2
    8.5,   # Black move 2
    2.1,   # White move 3
    45.3,  # Black move 3 (long think)
    4.2,   # White move 4
    6.8,   # Black move 4
]
```

### Room Codes
```python
VALID_CODES = [
    "ABC123",
    "XYZ789",
    "TEST01",
]

INVALID_CODES = [
    "AB",      # Too short
    "abcdef",  # Lowercase
    "AB12",    # Too short
    "AB!@#$",  # Special characters
]
```

---

## Test Reporting & Coverage

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src.game_server --cov-report=html

# Run specific test file
pytest tests/test_chess_engine.py -v
```

### Coverage Targets

| Module | Coverage Target |
|--------|-----------------|
| `src/game_server/chess_engine.py` | 100% |
| `src/game_server/clock.py` | 100% |
| `src/game_server/rooms.py` | 100% |
| `src/game_server/main.py` | 80% (WebSocket endpoints) |

### Manual Testing Checklist

#### Room Management
- [ ] Can create new room
- [ ] Receive 6-char room code
- [ ] Can join room via code
- [ ] Cannot join full room
- [ ] Cannot join non-existent room

#### Chess Moves
- [ ] Can move pawns
- [ ] Can move knights
- [ ] Can move bishops
- [ ] Can move rooks
- [ ] Can move queen
- [ ] Can move king
- [ ] Illegal moves rejected
- [ ] Wrong turn moves rejected

#### Special Moves
- [ ] Castling kingside works
- [ ] Castling queenside works
- [ ] Castling blocked by pieces
- [ ] Castling through check blocked
- [ ] En passant works
- [ ] En passant not available after move
- [ ] Promotion works (select Queen)
- [ ] Checkmate detected
- [ ] Stalemate detected

#### Clock
- [ ] Clock starts on game start (15 minutes)
- [ ] Clock stops on move
- [ ] Clock starts for opponent
- [ ] Time recorded for each move
- [ ] Timeout detected correctly
- [ ] Increment applied correctly (for 5+3)

#### Real-time Sync
- [ ] Move appears on opponent's board
- [ ] Board state consistent
- [ ] Clock sync between clients
- [ ] Game over message received

#### Results
- [ ] ELO displayed for both players
- [ ] Stats displayed (CPL, Blunders)
- [ ] Explanation displayed
- [ ] PGN valid

---

## Browser/Device Testing

### Desktop (Primary)
- [ ] Chrome 90+
- [ ] Firefox 88+
- [ ] Safari 14+
- [ ] Edge 90+

---

## Bug Tracking

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| **Critical** | Game cannot start or end | Immediate |
| **High** | Move validation broken | 24h |
| **Medium** | UI issue, work around exists | 1 week |
| **Low** | Cosmetic, enhancement | Next release |

### Regression Strategy
- Run full test suite before each commit
- Manual test critical paths before merge
- Keep integration tests stable (flaky tests = broken trust)
