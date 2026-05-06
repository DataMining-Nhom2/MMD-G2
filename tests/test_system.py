"""
System Tests - Chess Game Server
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.rooms import RoomManager, RoomStatus, PlayerColor
from src.game_server.chess_engine import ChessEngine
from src.game_server.clock import ClockService
from src.game_server.integration import predict_elo, get_explanation


class TestEndToEndGameFlow:
    """Test flow hoàn chỉnh"""

    def test_scholars_mate_full_flow(self):
        """Test Scholars Mate từ đầu đến cuối"""
        room_mgr = RoomManager()
        room = room_mgr.create_room("15+0")

        assert len(room.id) == 6

        result_white = room_mgr.join_room(room.id, "white_sid")
        assert result_white['color'] == PlayerColor.WHITE

        result_black = room_mgr.join_room(room.id, "black_sid")
        assert result_black['color'] == PlayerColor.BLACK

        room_ref = room_mgr.get_room(room.id)
        assert room_ref.status == RoomStatus.PLAYING.value

        engine = ChessEngine()
        for m in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
            result = engine.make_move(m)
            assert result.success is True

        status = engine.get_game_status()
        assert status.is_over is True
        assert status.result == "white"

    def test_full_game_with_ai(self):
        """Test game với AI prediction"""
        engine = ChessEngine()
        engine.make_move("e4")
        engine.make_move("e5")
        engine.make_move("Nf3")

        pgn = engine.to_pgn()
        clock_times = [5.0, 3.0, 2.0, 4.0]

        elo = predict_elo(pgn, clock_times)
        assert 800 <= elo["white_elo"] <= 2200

        explanation = get_explanation(elo)
        assert isinstance(explanation, str)


class TestRoomManagementSystem:
    """Test quản lý phòng"""

    def test_multiple_rooms(self):
        manager = RoomManager()
        rooms = [manager.create_room() for _ in range(5)]
        codes = [r.id for r in rooms]
        assert len(set(codes)) == 5

    def test_player_tracking(self):
        manager = RoomManager()
        room = manager.create_room()
        manager.join_room(room.id, "player1")
        manager.set_player_sid(room.id, "white", "sid1")
        assert manager.get_room_by_sid("sid1") == room.id


class TestClockSystem:
    """Test đồng hồ"""

    def test_clock_lifecycle(self):
        service = ClockService()
        room_code = "TEST"
        service.create_clock(room_code, "10+0")
        service.start_clock(room_code)
        service.make_move(room_code, "white")
        times = service.get_clock_times(room_code)
        assert len(times) == 1
        service.delete_clock(room_code)

    def test_increment_time_control(self):
        service = ClockService()
        service.create_clock("INC", "5+3")
        clock = service.get_clock("INC")
        assert clock.increment == 3.0


class TestChessSpecialPositions:
    """Test vị trí đặc biệt"""

    def test_castling(self):
        engine = ChessEngine()
        engine.load_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        result = engine.make_move("O-O")
        assert result.success is True

    def test_promotion(self):
        engine = ChessEngine()
        engine.load_fen("8/P7/8/8/8/8/8/8 w KQkq - 0 1")
        result = engine.make_move("a8=Q")
        assert result.success is True


class TestPerformance:
    """Test hiệu năng"""

    def test_room_creation_speed(self):
        import time
        manager = RoomManager()
        start = time.time()
        for _ in range(100):
            manager.create_room()
        elapsed = time.time() - start
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
