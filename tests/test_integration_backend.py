"""
Integration tests cho Backend
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.rooms import RoomManager, RoomStatus, PlayerColor
from src.game_server.chess_engine import ChessEngine
from src.game_server.clock import ClockService
from src.game_server.game import GameManager
from src.game_server.integration import predict_elo, get_explanation


class TestRoomFlow:
    """Test flow tạo và tham gia phòng"""

    def test_full_room_lifecycle(self):
        manager = RoomManager()

        room = manager.create_room("15+0")
        assert len(room.id) == 6

        room_ref = manager.get_room(room.id)
        assert room_ref.status == RoomStatus.WAITING.value

        result1 = manager.join_room(room.id, "player1_sid")
        assert result1['success'] is True
        assert result1['color'] == PlayerColor.WHITE

        result2 = manager.join_room(room.id, "player2_sid")
        assert result2['success'] is True
        assert result2['color'] == PlayerColor.BLACK
        assert result2['game_started'] is True

        room_ref = manager.get_room(room.id)
        assert room_ref.status == RoomStatus.PLAYING.value


class TestChessEngineFlow:
    """Test flow chơi cờ"""

    def test_scholars_mate(self):
        engine = ChessEngine()
        moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]
        for m in moves:
            result = engine.make_move(m)
            assert result.success is True

        status = engine.get_game_status()
        assert status.is_over is True
        assert status.result == "white"


class TestClockAndGameIntegration:
    """Test tích hợp Clock"""

    def test_clock_creation(self):
        service = ClockService()
        room_code = "TEST_ROOM"
        service.create_clock(room_code, "15+0")
        clock = service.get_clock(room_code)
        assert clock.white_remaining == 900.0
        service.delete_clock(room_code)


class TestAIPredictionIntegration:
    """Test AI prediction"""

    def test_predict_elo_structure(self):
        result = predict_elo("1. e4 e5", [5.0, 3.0])
        assert "white_elo" in result
        assert "black_elo" in result
        assert "stats" in result

    def test_predict_elo_deterministic(self):
        result1 = predict_elo("1. e4 e5", [5.0, 3.0])
        result2 = predict_elo("1. e4 e5", [5.0, 3.0])
        assert result1["white_elo"] == result2["white_elo"]

    def test_get_explanation(self):
        prediction = {
            "white_elo": 1500,
            "black_elo": 1400,
            "stats": {"white_avg_cpl": 10.5, "black_avg_cpl": 15.0, "white_blunders": 1, "black_blunders": 3}
        }
        explanation = get_explanation(prediction)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestErrorHandling:
    """Test xử lý lỗi"""

    def test_join_nonexistent_room(self):
        manager = RoomManager()
        result = manager.join_room("NOTEXIST", "player_sid")
        assert result['success'] is False
        assert result['error']['code'] == 'ROOM_NOT_FOUND'

    def test_invalid_chess_move(self):
        engine = ChessEngine()
        result = engine.make_move("e5")
        assert result.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
