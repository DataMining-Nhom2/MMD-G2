"""Unit tests cho TabularTransformer (Phase 2)."""

import unittest
import sys
from pathlib import Path

import polars as pl

# Đảm bảo import được package src khi chạy pytest trực tiếp từ workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import TabularTransformer


class TestTabularTransformer(unittest.TestCase):
    """Kiểm tra các biến đổi tabular chính theo planning Phase 2."""

    def _build_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ECO": ["A00", "B12", "A00", "C45", None],
                "GameFormat": ["Blitz", "Rapid", "Blitz", "Bullet", None],
                "BaseTime": [180, 600, 15_000, 30, 300],
                "Increment": [0, 5, 250, 1, None],
                "NumMoves": [20, 150, 80, 5, 200],
                "WhiteElo": [1200, 1500, 2000, 900, 1100],
                "BlackElo": [1300, 1450, 1800, 950, 900],
                "Moves": ["e4 e5", "d4 d5", "c4 e5", "Nf3 d5", "e4 c5"],
                "ModelBand": [1, 2, 3, 0, 1],
            }
        )

    def test_eco_and_category_one_hot(self) -> None:
        df = self._build_df()
        transformer = TabularTransformer(eco_top_n=2, realtime_mode=True)
        transformer.fit(df)
        out = transformer.transform(df)

        self.assertIn("eco_A00", out.columns)
        eco_cols = [
            c
            for c in out.columns
            if c.startswith("eco_") and not c.startswith("eco_cat_")
        ]
        self.assertEqual(len(eco_cols), 2)
        self.assertIn("eco_cat_A", out.columns)
        self.assertIn("eco_cat_B", out.columns)
        self.assertIn("eco_cat_C", out.columns)
        self.assertIn("eco_cat_D", out.columns)
        self.assertIn("eco_cat_E", out.columns)

    def test_game_format_handles_unknown(self) -> None:
        df = self._build_df()
        transformer = TabularTransformer(eco_top_n=2, realtime_mode=True)
        transformer.fit(df)
        out = transformer.transform(df)

        self.assertIn("gf_Unknown", out.columns)
        self.assertGreater(out["gf_Unknown"].sum(), 0.0)

    def test_numeric_transforms(self) -> None:
        df = self._build_df()
        transformer = TabularTransformer(eco_top_n=2, realtime_mode=True)
        transformer.fit(df)
        out = transformer.transform(df)

        self.assertIn("basetime_log", out.columns)
        self.assertIn("increment_log", out.columns)
        self.assertIn("num_moves_norm", out.columns)
        max_norm = float(out["num_moves_norm"].max() or 0.0)
        self.assertTrue(max_norm <= 1.0)

    def test_elodiff_only_when_not_realtime(self) -> None:
        df = self._build_df()

        transformer_realtime = TabularTransformer(eco_top_n=2, realtime_mode=True)
        transformer_realtime.fit(df)
        out_realtime = transformer_realtime.transform(df)
        self.assertNotIn("elo_diff_norm", out_realtime.columns)

        transformer_offline = TabularTransformer(eco_top_n=2, realtime_mode=False)
        transformer_offline.fit(df)
        out_offline = transformer_offline.transform(df)
        self.assertIn("elo_diff_norm", out_offline.columns)


if __name__ == "__main__":
    unittest.main()
