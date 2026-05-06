"""
Unit Tests for AI Engine — src/ai_engine/

Run with:
    pytest tests/unit/ -v
    pytest tests/integration/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_engine.eco_classifier import classify_eco
from src.ai_engine.stockfish_analyzer import analyze_game
from src.ai_engine.model_predictor import predict_elo


class TestECOClassifier:
    """Test suite for eco_classifier.py"""

    def test_classify_sicilian_defense(self):
        """1.e4 c5 should be classified as Sicilian Defense"""
        pgn = "1. e4 c5"
        result = classify_eco(pgn)
        assert result["code"] in ["B20", "B50", "B21", "B22", "B23", "B27", "B30"]
        assert "Sicilian" in result["name"] or "Defense" in result["name"]

    def test_classify_italian_game(self):
        """1.e4 e5 2.Nf3 Nc6 3.Bc4 should be Italian Game"""
        pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4"
        result = classify_eco(pgn)
        assert result["code"] == "C50"
        assert result["name"] == "Italian Game"

    def test_classify_ruyi_lopez(self):
        """Ruy Lopez (1.e4 e5 2.Nf3 Nc6 3.Bb5) should be C60"""
        pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5"
        result = classify_eco(pgn)
        assert result["code"] == "C60"

    def test_classify_slav_defense(self):
        """Slav Defense (1.d4 d5 2.c4 c6)"""
        pgn = "1. d4 d5 2. c4 c6"
        result = classify_eco(pgn)
        assert result["code"] == "D10"

    def test_classify_kings_indian_defense(self):
        """King's Indian Defense"""
        pgn = "1. d4 Nf6 2. c4 g6"
        result = classify_eco(pgn)
        assert result["code"] == "E60"

    def test_classify_french_defense(self):
        """French Defense (1.e4 e6)"""
        pgn = "1. e4 e6"
        result = classify_eco(pgn)
        assert result["code"] == "C00"
        assert result["name"] == "French Defense"

    def test_classify_queens_gambit_accepted(self):
        """Queen's Gambit Accepted"""
        pgn = "1. d4 d5 2. c4 dxc4"
        result = classify_eco(pgn)
        assert result["code"] == "D20"

    def test_classify_empty_pgn(self):
        """Empty PGN should return Uncommon Opening"""
        result = classify_eco("")
        assert result["code"] == "A00"
        assert result["name"] == "Uncommon Opening"

    def test_return_type(self):
        """Result should be a dict with code and name"""
        result = classify_eco("1. e4 e5")
        assert isinstance(result, dict)
        assert "code" in result
        assert "name" in result

    def test_pgn_with_numbers_and_dots_parsed(self):
        """PGN with move numbers should be parsed correctly"""
        result = classify_eco("1. e4 e5 2. Nf3 Nc6")
        assert isinstance(result["code"], str)
        assert isinstance(result["name"], str)


class TestStockfishAnalyzer:
    """Test suite for stockfish_analyzer.py"""

    def test_analyze_returns_dict(self):
        result = analyze_game("1. e4 e5 2. Nf3 Nc6")
        assert isinstance(result, dict)

    def test_analyze_has_required_fields(self):
        result = analyze_game("1. e4 e5 2. Nf3 Nc6")
        assert "white_avg_cpl" in result
        assert "black_avg_cpl" in result
        assert "white_blunders" in result
        assert "black_blunders" in result
        assert "total_moves" in result

    def test_cpl_values_are_non_negative(self):
        result = analyze_game("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")
        assert result["white_avg_cpl"] >= 0
        assert result["black_avg_cpl"] >= 0

    def test_blunder_counts_are_non_negative(self):
        result = analyze_game("1. e4 e5")
        assert result["white_blunders"] >= 0
        assert result["black_blunders"] >= 0

    def test_total_moves_matches_pgn(self):
        pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
        result = analyze_game(pgn)
        assert result["total_moves"] == 6

    def test_empty_pgn_has_total_moves(self):
        result = analyze_game("")
        assert "total_moves" in result
        assert result["total_moves"] >= 0


class TestModelPredictor:
    """Test suite for model_predictor.py"""

    def test_predict_elo_returns_dict(self):
        features = analyze_game("1. e4 e5")
        result = predict_elo("1. e4 e5", [5.0, 3.0], features)
        assert isinstance(result, dict)

    def test_predict_has_white_and_black_elo(self):
        features = analyze_game("1. e4 e5")
        result = predict_elo("1. e4 e5", [5.0, 3.0], features)
        assert "white_elo" in result
        assert "black_elo" in result

    def test_elo_in_valid_range(self):
        features = analyze_game("1. e4 e5 2. Nf3 Nc6")
        result = predict_elo("1. e4 e5 2. Nf3 Nc6", [5.0, 3.0, 12.0, 8.0], features)
        assert 400 <= result["white_elo"] <= 3000
        assert 400 <= result["black_elo"] <= 3000

    def test_predict_returns_integers(self):
        features = analyze_game("1. e4")
        result = predict_elo("1. e4", [5.0], features)
        assert isinstance(result["white_elo"], int)
        assert isinstance(result["black_elo"], int)

    def test_predict_with_empty_clock_times(self):
        features = analyze_game("1. e4")
        result = predict_elo("1. e4", [], features)
        assert "white_elo" in result
        assert "black_elo" in result
