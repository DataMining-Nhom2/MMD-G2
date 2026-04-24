"""
Frontend Tests - Chess Board UI Components
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.chess_engine import ChessEngine


class TestPieceUnicode:
    """Test piece Unicode mapping"""

    def test_unicode_pieces(self):
        pieces_map = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        for symbol, unicode_char in pieces_map.items():
            assert len(unicode_char) == 1


class TestClockFormatting:
    """Test clock formatting"""

    def test_format_time(self):
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        assert format_time(900) == "15:00"
        assert format_time(60) == "01:00"
        assert format_time(0) == "00:00"


class TestMoveHistory:
    """Test move history"""

    def test_pair_moves(self):
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4"]
        pairs = []
        for i in range(0, len(moves), 2):
            white = moves[i]
            black = moves[i + 1] if i + 1 < len(moves) else ""
            pairs.append((white, black))

        assert len(pairs) == 3
        assert pairs[0] == ("e4", "e5")
        assert pairs[2] == ("Bc4", "")


class TestGameResult:
    """Test game result"""

    def test_white_win(self):
        engine = ChessEngine()
        for m in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
            engine.make_move(m)
        status = engine.get_game_status()
        assert status.result == "white"

    def test_black_win(self):
        engine = ChessEngine()
        for m in ["f3", "e5", "g4", "Qh4"]:
            engine.make_move(m)
        status = engine.get_game_status()
        assert status.result == "black"


class TestErrorMessages:
    """Test error codes"""

    def test_room_not_found(self):
        from src.game_server.rooms import RoomManager
        manager = RoomManager()
        result = manager.join_room("INVALID", "sid")
        assert result['success'] is False
        assert result['error']['code'] == 'ROOM_NOT_FOUND'


class TestTimeControlOptions:
    """Test time control"""

    def test_time_control_options(self):
        options = ["15+0", "10+0", "5+3", "3+0"]
        assert len(options) == 4
        assert "15+0" in options


class TestAIOutputContract:
    """Test AI output contract"""

    def test_predict_elo_output(self):
        from src.game_server.integration import predict_elo
        result = predict_elo("1. e4 e5", [5.0, 3.0])
        assert isinstance(result["white_elo"], int)
        assert isinstance(result["black_elo"], int)
        assert "stats" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
