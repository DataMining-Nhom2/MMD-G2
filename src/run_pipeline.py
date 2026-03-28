"""Chạy full Feature Pipeline V2 trên sample 30k ván.

Theo Planning V2:
- Fit TabularTransformer trên toàn bộ 30k (để lấy đủ top-100 ECO vocab).
- Chạy StockfishTransformer V2 phân tích 11 features cho 30k ván.
- Lưu kết quả ra data/features/sample_30k_features_v2.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from src.feature_engineering import FeaturePipeline
from src.feature_config import (
    FeatureConfig,
    SAMPLE_SOURCE_FILE,
    SAMPLE_FEATURES_V2_FILE,
)


def main() -> None:
    t0 = time.time()
    print(f"{'═' * 60}")
    print("  CHẠY FEATURE PIPELINE V2 TRÊN SAMPLE 30K")
    print("  (11 features: Nhóm A + B + C)")
    print(f"{'─' * 60}")

    # Bước 1: Load sample
    print(f"  Đang load: {SAMPLE_SOURCE_FILE}")
    df = pl.read_parquet(str(SAMPLE_SOURCE_FILE))
    print(f"  Rows: {df.height}, Cols: {df.width}")

    # Bước 2: Khởi tạo pipeline
    config = FeatureConfig(
        stockfish_path=str(Path(__file__).resolve().parent.parent / ".tmp" / "stockfish_binary"),
        stockfish_depth=10,
        eco_top_n=100,
    )
    pipeline = FeaturePipeline(config=config)

    # Bước 3: Fit (chỉ TabularTransformer cần fit, Stockfish là stateless)
    print("\n  Đang fit TabularTransformer (ECO vocab)...")
    pipeline.fit(df)
    print(f"  ECO vocab size: {len(pipeline.tabular.eco_vocab)}")

    # Bước 4: Transform (bao gồm Stockfish V2 — ước tính ~25-30 phút)
    print(f"\n  Đang transform {df.height} ván (V2: WDL + PV + Phase CPL)...")
    print(f"  (Ước tính: ~{df.height * 0.05 / 60:.0f} phút)")
    result = pipeline.transform(df)

    # Bước 5: Lưu kết quả V2 (KHÔNG ghi đè file V1)
    SAMPLE_FEATURES_V2_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(SAMPLE_FEATURES_V2_FILE), compression="zstd")

    # Bước 6: Lưu metadata
    feature_cols = [c for c in result.columns if c != "ModelBand"]
    pipeline.save_metadata(feature_cols)

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✅ HOÀN THÀNH PIPELINE V2!")
    print(f"  Output: {SAMPLE_FEATURES_V2_FILE}")
    print(f"  Rows: {result.height}, Cols: {result.width}")
    print(f"  Thời gian: {elapsed/60:.1f} phút ({elapsed/3600:.1f} giờ)")
    print(f"\n  Schema:")
    for name, dtype in result.schema.items():
        print(f"    {name}: {dtype}")

    # Kiểm tra NaN distribution
    print(f"\n  NaN distribution:")
    for col in ["opening_cpl", "midgame_cpl", "endgame_cpl"]:
        if col in result.columns:
            nan_count = result[col].is_nan().sum()
            nan_pct = nan_count / result.height * 100
            print(f"    {col}: {nan_count} NaN ({nan_pct:.1f}%)")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
