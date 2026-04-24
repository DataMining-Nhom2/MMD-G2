"""
Chess Engine - Wrapper cho python-chess
Xử lý logic cờ vua theo luật FIDE
"""
import chess
import chess.pgn
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoveResult:
    """Kết quả của một nước đi"""
    success: bool
    new_fen: str = ""
    san: str = ""
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    error: str = ""


@dataclass
class GameStatus:
    """Trạng thái ván cờ"""
    is_over: bool
    result: Optional[str]  # "white", "black", "draw"
    reason: Optional[str]  # "checkmate", "stalemate", "timeout", "resignation"


class ChessEngine:
    """Engine xử lý cờ vua"""

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """Reset bàn cờ về vị trí ban đầu"""
        self.board = chess.Board()
        return self.board.fen()

    def load_fen(self, fen: str) -> bool:
        """
        Load position từ FEN
        Returns: True nếu FEN hợp lệ
        """
        try:
            self.board = chess.Board(fen)
            return True
        except ValueError:
            return False

    def get_fen(self) -> str:
        """Lấy FEN hiện tại"""
        return self.board.fen()

    def is_valid_position(self) -> bool:
        """Kiểm tra vị trí hiện tại có hợp lệ không"""
        return self.board.is_valid()

    def validate_move(self, move_san: str) -> bool:
        """
        Kiểm tra nước đi có hợp lệ không
        Args:
            move_san: Nước đi theo SAN notation (ví dụ: "e4", "Nf3", "O-O")
        Returns:
            True nếu nước đi hợp lệ
        """
        try:
            move = self.board.parse_san(move_san)
            return self.board.is_legal(move)
        except ValueError:
            return False

    def make_move(self, move_san: str) -> MoveResult:
        """
        Thực hiện nước đi
        Args:
            move_san: Nước đi theo SAN notation
        Returns:
            MoveResult với thông tin kết quả
        """
        try:
            move = self.board.parse_san(move_san)

            if not self.board.is_legal(move):
                return MoveResult(
                    success=False,
                    error=f"Nước đi '{move_san}' không hợp lệ"
                )

            # Lưu SAN trước khi push
            san = self.board.san(move)

            # Thực hiện nước đi
            self.board.push(move)

            return MoveResult(
                success=True,
                new_fen=self.board.fen(),
                san=san,
                is_check=self.board.is_check(),
                is_checkmate=self.board.is_checkmate(),
                is_stalemate=self.board.is_stalemate()
            )

        except ValueError as e:
            return MoveResult(
                success=False,
                error=f"Nước đi '{move_san}' không hợp lệ: {str(e)}"
            )

    def get_legal_moves(self) -> list[str]:
        """
        Lấy danh sách tất cả nước đi hợp lệ
        Returns:
            List các nước đi theo SAN
        """
        return [self.board.san(move) for move in self.board.legal_moves]

    def get_legal_moves_from_square(self, square: str) -> list[str]:
        """
        Lấy danh sách nước đi hợp lệ từ một ô
        Args:
            square: Tên ô (ví dụ: "e2")
        Returns:
            List các nước đi theo SAN
        """
        try:
            sq = chess.square_from_name(square.upper())
            piece = self.board.piece_at(sq)

            if piece is None:
                return []

            moves = []
            for move in self.board.legal_moves:
                if move.from_square == sq:
                    moves.append(self.board.san(move))

            return moves
        except ValueError:
            return []

    def get_game_status(self) -> GameStatus:
        """
        Kiểm tra trạng thái ván cờ
        Returns:
            GameStatus với thông tin kết quả
        """
        if self.board.is_checkmate():
            # Người thắng là người vừa đi (turn hiện tại là người thua)
            winner = "black" if self.board.turn == chess.WHITE else "white"
            return GameStatus(
                is_over=True,
                result=winner,
                reason="checkmate"
            )

        if self.board.is_stalemate():
            return GameStatus(
                is_over=True,
                result="draw",
                reason="stalemate"
            )

        if self.board.is_insufficient_material():
            return GameStatus(
                is_over=True,
                result="draw",
                reason="insufficient_material"
            )

        if self.board.is_fifty_moves():
            return GameStatus(
                is_over=True,
                result="draw",
                reason="fifty_moves"
            )

        if self.board.is_seventyfive_moves():
            return GameStatus(
                is_over=True,
                result="draw",
                reason="seventyfive_moves"
            )

        if self.board.can_claim_threefold_repetition():
            return GameStatus(
                is_over=True,
                result="draw",
                reason="threefold_repetition"
            )

        return GameStatus(
            is_over=False,
            result=None,
            reason=None
        )

    def get_turn(self) -> str:
        """Lấy lượt đi hiện tại"""
        return "white" if self.board.turn == chess.WHITE else "black"

    def is_in_check(self) -> bool:
        """Kiểm tra có đang bị chiếu không"""
        return self.board.is_check

    def to_pgn(self) -> str:
        """
        Export ván cờ thành PGN
        """
        pgn = chess.pgn.Game()

        # Add headers
        pgn.headers["Event"] = "MMD-G2 Chess"
        pgn.headers["Site"] = "Local"
        pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn.headers["White"] = "Player White"
        pgn.headers["Black"] = "Player Black"

        # Build game tree
        node = pgn
        for move in self.board.move_stack:
            node = node.add_variation(move)

        # Set result
        if self.board.is_checkmate():
            winner = "0-1" if self.board.turn == chess.WHITE else "1-0"
            pgn.headers["Result"] = winner
        else:
            pgn.headers["Result"] = "*"

        return str(pgn)

    def get_move_history(self) -> list[str]:
        """
        Lấy lịch sử các nước đi
        Returns:
            List các nước đi theo SAN
        """
        # Use push/pop to correctly get SAN for each historical move
        history = []
        temp_board = chess.Board()

        for move in self.board.move_stack:
            try:
                san = temp_board.san(move)
                history.append(san)
                temp_board.push(move)
            except ValueError:
                # Fallback: use UCI notation
                history.append(temp_board.uci(move))
                temp_board.push(move)

        return history

    def get_piece_at(self, square: str) -> Optional[str]:
        """
        Lấy quân cờ tại một ô
        Args:
            square: Tên ô (ví dụ: "e1")
        Returns:
            Ký hiệu quân cờ hoặc None
        """
        try:
            sq = chess.square_from_name(square.upper())
            piece = self.board.piece_at(sq)
            if piece:
                return piece.symbol()
            return None
        except ValueError:
            return None
