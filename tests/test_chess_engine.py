"""
Unit tests cho Chess Engine - Verified
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.chess_engine import ChessEngine, MoveResult, GameStatus


class TestChessEngineInit:
    """Test khởi tạo ChessEngine"""

    def test_default_initialization(self):
        engine = ChessEngine()
        assert engine.board is not None
        assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in engine.board.fen()

    def test_reset(self):
        engine = ChessEngine()
        engine.reset()
        assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in engine.get_fen()


class TestValidateMoves:
    """Test validate moves"""

    def test_pawn_forward(self):
        engine = ChessEngine()
        assert engine.validate_move("e4") is True

    def test_knight_move(self):
        engine = ChessEngine()
        assert engine.validate_move("Nf3") is True
        assert engine.validate_move("Nc3") is True

    def test_illegal_move(self):
        engine = ChessEngine()
        assert engine.validate_move("e5") is False  # blocked
        assert engine.validate_move("INVALID") is False


class TestMakeMoves:
    """Test thực hiện nước đi"""

    def test_make_pawn(self):
        engine = ChessEngine()
        result = engine.make_move("e4")
        assert result.success is True
        assert result.san == "e4"

    def test_make_illegal(self):
        engine = ChessEngine()
        result = engine.make_move("e5")
        assert result.success is False


class TestSpecialMoves:
    """Test nước đi đặc biệt"""

    def test_castling_kingside(self):
        engine = ChessEngine()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]
        for m in moves:
            assert engine.make_move(m).success is True

    def test_castling_queenside(self):
        engine = ChessEngine()
        engine.load_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        result = engine.make_move("O-O-O")
        assert result.success is True

    def test_en_passant(self):
        engine = ChessEngine()
        for m in ["e4", "a6", "e5", "d5", "exd6"]:
            assert engine.make_move(m).success is True


class TestGameStatus:
    """Test trạng thái game"""

    def test_not_over_at_start(self):
        engine = ChessEngine()
        status = engine.get_game_status()
        assert status.is_over is False

    def test_checkmate(self):
        engine = ChessEngine()
        for m in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
            engine.make_move(m)
        status = engine.get_game_status()
        assert status.is_over is True
        assert status.result == "white"


class TestPGNExport:
    """Test xuất PGN"""

    def test_pgn_basic(self):
        engine = ChessEngine()
        engine.make_move("e4")
        engine.make_move("e5")
        pgn = engine.to_pgn()
        assert "e4" in pgn
        assert "Event" in pgn

    def test_pgn_checkmate(self):
        engine = ChessEngine()
        for m in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
            engine.make_move(m)
        pgn = engine.to_pgn()
        assert "1-0" in pgn


class TestFENParsing:
    """Test FEN parsing"""

    def test_load_valid_fen(self):
        engine = ChessEngine()
        assert engine.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") is True

    def test_reject_invalid_fen(self):
        engine = ChessEngine()
        assert engine.load_fen("invalid") is False


class TestHelperMethods:
    """Test helper methods"""

    def test_get_legal_moves(self):
        engine = ChessEngine()
        moves = engine.get_legal_moves()
        assert len(moves) > 0
        assert isinstance(moves, list)

    def test_get_game_status(self):
        engine = ChessEngine()
        status = engine.get_game_status()
        assert isinstance(status, GameStatus)

    def test_get_turn(self):
        engine = ChessEngine()
        assert engine.get_turn() == "white"

    def test_get_fen(self):
        engine = ChessEngine()
        fen = engine.get_fen()
        assert isinstance(fen, str)
        assert len(fen) > 0

    def test_get_move_history(self):
        engine = ChessEngine()
        engine.make_move("e4")
        history = engine.get_move_history()
        assert len(history) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
