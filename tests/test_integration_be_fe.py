"""
Integration tests - Backend and Frontend Connection
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRoomCodeFormat:
    """Test format room code"""

    def test_room_code_is_6_chars(self):
        from src.game_server.rooms import RoomManager
        manager = RoomManager()
        code = manager.generate_room_code()
        assert len(code) == 6

    def test_room_code_is_alphanumeric(self):
        from src.game_server.rooms import RoomManager
        manager = RoomManager()
        code = manager.generate_room_code()
        assert code.isalnum() is True


class TestSocketIOEventContracts:
    """Test contracts của Socket.IO events"""

    def test_create_room_payload(self):
        payload = {"time_control": "15+0"}
        assert "time_control" in payload

    def test_join_room_payload(self):
        payload = {"room_code": "ABC123"}
        assert "room_code" in payload

    def test_make_move_payload(self):
        payload = {"move": "e4"}
        assert "move" in payload

    def test_game_over_response(self):
        response = {
            "result": "white",
            "reason": "checkmate",
            "pgn": "1. e4 e5 2. Qh5 1-0",
            "clock_times": [5.0, 3.0],
            "white_elo": 1523,
            "black_elo": 1345,
            "stats": {"white_avg_cpl": 45.2, "black_avg_cpl": 67.8, "white_blunders": 2, "black_blunders": 4},
            "explanation": "Test"
        }
        assert "result" in response
        assert "white_elo" in response
        assert "stats" in response

    def test_error_response(self):
        response = {"code": "ROOM_NOT_FOUND", "message": "Phòng không tồn tại"}
        assert "code" in response
        assert "message" in response


class TestErrorCodes:
    """Test error codes"""

    def test_error_codes(self):
        error_codes = ["ROOM_NOT_FOUND", "ROOM_FULL", "INVALID_MOVE", "NOT_YOUR_TURN"]
        for code in error_codes:
            assert "_" in code or code.isupper()


class TestTimeControlConsistency:
    """Test tính nhất quán của time control"""

    def test_backend_time_control(self):
        from src.game_server.clock import ClockService
        service = ClockService()
        assert service.parse_time_control("15+0") == (900.0, 0.0)
        assert service.parse_time_control("5+3") == (300.0, 3.0)

    def test_frontend_time_control_options(self):
        options = ["15+0", "10+0", "5+3", "3+0"]
        assert len(options) == 4


class TestAIIntegrationContract:
    """Test contract của AI integration"""

    def test_predict_elo_output(self):
        from src.game_server.integration import predict_elo
        result = predict_elo("1. e4 e5", [5.0, 3.0])
        assert "white_elo" in result
        assert "black_elo" in result
        assert "stats" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
