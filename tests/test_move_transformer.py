"""Unit tests cho MoveTransformer (Phase 3)."""

import unittest
import sys
from pathlib import Path

import polars as pl

# Đảm bảo import được package src khi chạy pytest trực tiếp từ workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import MoveTransformer


class TestMoveTransformer(unittest.TestCase):
    """Kiểm tra tokenizer, bigram, entropy và board-state features."""

    def test_tokenize_strips_move_numbers_and_result(self) -> None:
        transformer = MoveTransformer(n_ply=10, svd_dim=4)
        text = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0"
        tokens = transformer._tokenize(text, n_ply=6)
        self.assertEqual(tokens, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])

    def test_bigram_generation(self) -> None:
        transformer = MoveTransformer(n_ply=10, svd_dim=4)
        tokens = ["e4", "e5", "Nf3"]
        bigrams = transformer._bigrams(tokens)
        self.assertEqual(bigrams, ["e4 e5", "e5 Nf3"])

    def test_entropy_non_negative(self) -> None:
        transformer = MoveTransformer(n_ply=10, svd_dim=4)
        value = transformer._entropy(["e4", "e5", "e4", "e5"])
        self.assertGreaterEqual(value, 0.0)

    def test_transform_outputs_expected_shape(self) -> None:
        transformer = MoveTransformer(n_ply=10, svd_dim=8)
        series = pl.Series(
            [
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
                "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
            ]
        )
        out = transformer.transform(series)

        # svd_dim + 10 cột thủ công
        self.assertEqual(out.shape, (2, 18))

    def test_board_state_features_castling_detected(self) -> None:
        transformer = MoveTransformer(n_ply=20, svd_dim=4)
        tokens = transformer._tokenize("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O Nf6")
        ks, qs, checks, pawn_ratio = transformer._board_state_features(tokens)
        self.assertEqual(ks, 1.0)
        self.assertEqual(qs, 0.0)
        self.assertGreaterEqual(checks, 0.0)
        self.assertGreaterEqual(pawn_ratio, 0.0)

    def test_fit_enables_tfidf_svd_transform(self) -> None:
        transformer = MoveTransformer(n_ply=10, svd_dim=6)
        train_series = pl.Series(
            [
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
                "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
                "1. e4 c5 2. Nf3 d6 3. d4 cxd4",
                "1. c4 e5 2. Nc3 Nf6 3. g3 d5",
            ]
        )
        transformer.fit(train_series, sample_size=10)

        self.assertIsNotNone(transformer.tfidf_vectorizer)
        # Với dữ liệu ngắn có thể không fit được SVD; test chính là pipeline không lỗi.
        out = transformer.transform(train_series)
        self.assertEqual(out.shape, (4, 16))  # svd_dim + 10 cột thủ công


if __name__ == "__main__":
    unittest.main()
