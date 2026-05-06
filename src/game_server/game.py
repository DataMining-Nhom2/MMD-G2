"""
Game Manager - Điều phối trạng thái ván cờ
Kết hợp Room Manager, Chess Engine và Clock Service
"""

from typing import Optional
from dataclasses import dataclass

from src.game_server.rooms import room_manager, Room, PlayerColor
from src.game_server.chess_engine import ChessEngine
from src.game_server.clock import clock_service


@dataclass
class GameResult:
    """Kết quả ván cờ"""

    result: str  # "white", "black", "draw"
    reason: str  # "checkmate", "stalemate", "timeout", "resignation"
    pgn: str
    clock_times: list[float]
    white_elo: int = 0
    black_elo: int = 0
    stats: dict = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = {}


class GameManager:
    """Manager điều phối game"""

    def __init__(self):
        # Lịch sử engine cho các phòng
        self.engines: dict[str, ChessEngine] = {}

    def get_engine(self, room_code: str) -> ChessEngine:
        """Lấy hoặc tạo chess engine cho phòng"""
        if room_code not in self.engines:
            self.engines[room_code] = ChessEngine()
        return self.engines[room_code]

    def reset_engine(self, room_code: str):
        """Reset engine cho phòng mới"""
        self.engines[room_code] = ChessEngine()

    def delete_engine(self, room_code: str):
        """Xóa engine khi phòng kết thúc"""
        if room_code in self.engines:
            del self.engines[room_code]

    def start_game(self, room_code: str):
        """
        Bắt đầu ván cờ
        """
        # Reset engine
        self.reset_engine(room_code)

        # Reset room state
        room = room_manager.get_room(room_code)
        if room:
            room.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            room.moves = []
            room.current_turn = PlayerColor.WHITE
            room.game_result = None

        # Create clock
        # clock_service.create_clock(room_code, room.time_control if room else "15+0")
        clock_service.create_clock(room_code, "15+0")
        # Start clock
        clock_service.start_clock(room_code)

    def handle_move(self, sid: str, move_san: str) -> dict:
        """
        Xử lý nước đi
        Returns: dict với success, error, và các thông tin move
        """
        # Get room
        room_code = room_manager.get_room_by_sid(sid)
        if not room_code:
            return {"success": False, "error": "Bạn không ở trong phòng nào"}

        room = room_manager.get_room(room_code)
        if not room:
            return {"success": False, "error": "Phòng không tồn tại"}

        # Get player color
        color = room.get_player_color(sid)
        if not color:
            return {"success": False, "error": "Không tìm thấy người chơi"}

        # Check turn
        if room.current_turn != color:
            return {"success": False, "error": "Chưa đến lượt của bạn"}

        # Get engine
        engine = self.get_engine(room_code)

        # Load current position
        engine.load_fen(room.fen)

        # Validate move
        if not engine.validate_move(move_san):
            return {"success": False, "error": f'Nước đi "{move_san}" không hợp lệ'}

        # Record time
        elapsed = clock_service.make_move(room_code, color.value)

        # Make move
        result = engine.make_move(move_san)

        if not result.success:
            return {"success": False, "error": result.error}

        # Update room state
        room.fen = result.new_fen
        room.moves.append(result.san)
        room.current_turn = (
            PlayerColor.BLACK if color == PlayerColor.WHITE else PlayerColor.WHITE
        )

        return {
            "success": True,
            "san": result.san,
            "fen": result.new_fen,
            "is_check": result.is_check,
            "is_checkmate": result.is_checkmate,
            "is_stalemate": result.is_stalemate,
            "moves": room.moves,
        }

    def check_game_over(self, room_code: str) -> dict:
        """
        Kiểm tra ván cờ có kết thúc không
        """
        room = room_manager.get_room(room_code)
        if not room:
            return {"is_over": False}

        engine = self.get_engine(room_code)
        engine.load_fen(room.fen)

        status = engine.get_game_status()

        # Check timeout
        timeout = clock_service.is_timeout(room_code)
        if timeout:
            winner = "black" if timeout == "white" else "white"
            return {"is_over": True, "result": winner, "reason": "timeout"}

        if status.is_over:
            return {"is_over": True, "result": status.result, "reason": status.reason}

        return {"is_over": False}

    def resign(self, sid: str) -> dict:
        """
        Người chơi xin thua
        """
        room_code = room_manager.get_room_by_sid(sid)
        if not room_code:
            return {"success": False, "error": "Bạn không ở trong phòng nào"}

        room = room_manager.get_room(room_code)
        if not room:
            return {"success": False, "error": "Phòng không tồn tại"}

        color = room.get_player_color(sid)
        if not color:
            return {"success": False, "error": "Không tìm thấy người chơi"}

        # Winner là đối thủ
        winner = "black" if color == PlayerColor.WHITE else "white"

        return {"success": True, "result": winner}

    async def end_game(self, room_code: str, result: str, reason: str) -> dict:
        """
        Kết thúc ván cờ - trả về dict kết quả để WebSocket endpoint broadcast.
        """
        room = room_manager.get_room(room_code)
        if not room:
            return {}

        room.game_result = result
        room.status = "finished"
        clock_service.delete_clock(room_code)

        engine = self.get_engine(room_code)
        pgn = engine.to_pgn()
        clock_times = clock_service.get_clock_times(room_code)

        from src.game_server.integration import predict_elo, get_explanation

        elo_result = predict_elo(pgn, clock_times)
        explanation = get_explanation(elo_result)

        return {
            "result": result,
            "reason": reason,
            "pgn": pgn,
            "clock_times": clock_times,
            "white_elo": elo_result["white_elo"],
            "black_elo": elo_result["black_elo"],
            "stats": elo_result["stats"],
            "explanation": explanation,
        }

    def cleanup_room(self, room_code: str):
        """
        Dọn dẹp khi phòng kết thúc
        """
        self.delete_engine(room_code)
        clock_service.delete_clock(room_code)


# Singleton instance
game_manager = GameManager()
