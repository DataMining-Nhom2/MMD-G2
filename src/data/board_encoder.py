"""Board Encoder — Chuyển đổi trạng thái bàn cờ thành tensor 12×8×8.

Mỗi vị trí bàn cờ được mã hóa thành 12 mặt phẳng nhị phân (binary planes):
  - Plane 0-5:  Quân trắng (Pawn, Knight, Bishop, Rook, Queen, King)
  - Plane 6-11: Quân đen   (Pawn, Knight, Bishop, Rook, Queen, King)

Giá trị: 1.0 nếu có quân tại ô đó, 0.0 nếu không.

Tham khảo: Paper arXiv:2409.11506 — CNN input format.
"""

from __future__ import annotations

import chess
import numpy as np


# Map loại quân → index plane (0-5)
PIECE_TYPE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode trạng thái bàn cờ thành tensor 12×8×8.

    Args:
        board: Đối tượng chess.Board hiện tại.

    Returns:
        np.ndarray shape (12, 8, 8), dtype float32.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        plane_idx = PIECE_TYPE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6

        # chess.SQUARES: a1=0, b1=1, ..., h8=63
        # Chuyển sang row/col: row = square // 8, col = square % 8
        row = square // 8
        col = square % 8
        planes[plane_idx, row, col] = 1.0

    return planes


def replay_game_to_boards(moves_san: str) -> list[np.ndarray]:
    """Replay ván cờ từ SAN string → danh sách board state tensors.

    Args:
        moves_san: Chuỗi nước đi SAN (ví dụ "1. e4 e5 2. Nf3 Nc6 ...").

    Returns:
        Danh sách np.ndarray shape (12, 8, 8), mỗi phần tử là trạng thái
        bàn cờ SAU KHI thực hiện nước đi tương ứng.
    """
    board = chess.Board()
    boards = []

    # Parse SAN string → danh sách nước đi
    # Loại bỏ số thứ tự nước: "1.", "2.", "1..."
    import re
    tokens = moves_san.split()
    for token in tokens:
        # Bỏ qua số thứ tự (vd: "1.", "12.", "1...")
        if re.match(r'^\d+\.', token):
            continue
        # Bỏ qua kết quả ván (1-0, 0-1, 1/2-1/2)
        if token in ('1-0', '0-1', '1/2-1/2', '*'):
            continue

        try:
            move = board.parse_san(token)
            board.push(move)
            boards.append(encode_board(board))
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            # Nước đi lỗi → dừng replay tại đây
            break

    return boards


def replay_game_to_boards_from_moves_list(board: chess.Board, moves: list[chess.Move]) -> list[np.ndarray]:
    """Replay từ danh sách Move objects (nhanh hơn parse SAN).

    Args:
        board: Board khởi tạo (thường là chess.Board()).
        moves: Danh sách chess.Move objects.

    Returns:
        Danh sách np.ndarray shape (12, 8, 8).
    """
    boards = []
    for move in moves:
        board.push(move)
        boards.append(encode_board(board))
    return boards
