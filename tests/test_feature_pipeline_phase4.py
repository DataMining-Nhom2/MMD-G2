"""Unit tests cho FeaturePipeline Phase 4 (feature store + quality gate)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_config import FeatureConfig
from src.feature_engineering import FeaturePipeline


class TestFeaturePipelinePhase4(unittest.TestCase):
    """Kiểm tra các năng lực chính của Phase 4."""

    @staticmethod
    def _build_df(n: int, short_rows: int = 0) -> pl.DataFrame:
        moves = []
        for i in range(n):
            if i < short_rows:
                moves.append("1. e4 e5")  # 2 ply -> bị loại
            else:
                moves.append("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")

        return pl.DataFrame(
            {
                "ECO": ["A00"] * n,
                "GameFormat": ["Blitz"] * n,
                "BaseTime": [180] * n,
                "Increment": [0] * n,
                "NumMoves": [20] * n,
                "Moves": moves,
                "EloAvg": [1500] * n,
            }
        )

    def test_temporal_split_files(self) -> None:
        pipeline = FeaturePipeline()
        train, val = pipeline.temporal_split_files(
            [
                "data/processed/lichess_2025-12_ml.parquet",
                "data/processed/lichess_2026-01_ml.parquet",
                "data/processed/other.parquet",
            ]
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)

    def test_quality_gate_pass_and_warning(self) -> None:
        pipeline = FeaturePipeline()

        pass_df = self._build_df(100, short_rows=1)  # 1%
        _, pass_stats = pipeline.apply_data_quality_gate(pass_df, split_name="train")
        self.assertEqual(pass_stats["status"], "pass")

        warn_df = self._build_df(100, short_rows=2)  # 2%
        _, warn_stats = pipeline.apply_data_quality_gate(warn_df, split_name="train")
        self.assertEqual(warn_stats["status"], "warning")

    def test_quality_gate_fail(self) -> None:
        pipeline = FeaturePipeline()
        fail_df = self._build_df(100, short_rows=4)  # 4%
        with self.assertRaises(ValueError):
            pipeline.apply_data_quality_gate(fail_df, split_name="train")

    def test_process_split_and_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "lichess_2025-12_ml.parquet"
            output_file = tmp_path / "train_features.parquet"

            df = self._build_df(20, short_rows=0)
            df.write_parquet(input_file)

            cfg = FeatureConfig(
                batch_size=5,
                train_features_file=output_file,
                val_features_file=tmp_path / "val_features.parquet",
                tfidf_vocab_file=tmp_path / "tfidf_vocabulary.pkl",
                svd_components_file=tmp_path / "svd_components.pkl",
                feature_columns_file=tmp_path / "feature_columns.json",
            )
            pipeline = FeaturePipeline(config=cfg)

            stats = pipeline.process_split_to_parquet(
                input_files=[str(input_file)],
                output_file=output_file,
                split_name="train",
            )
            self.assertEqual(stats["status"], "pass")
            self.assertTrue(output_file.exists())

            out_df = pl.read_parquet(output_file)
            self.assertIn("ModelBand", out_df.columns)

            pipeline.save_metadata(out_df.columns)
            self.assertTrue(cfg.feature_columns_file.exists())
            self.assertTrue(cfg.tfidf_vocab_file.exists())
            self.assertTrue(cfg.svd_components_file.exists())


if __name__ == "__main__":
    unittest.main()
