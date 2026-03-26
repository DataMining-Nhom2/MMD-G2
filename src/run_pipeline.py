"""Chạy full Feature Pipeline trên sample 30k ván.

Theo Planning Task 4.4:
- Fit TabularTransformer trên toàn bộ 30k (để lấy đủ top-100 ECO vocab).
- Chạy StockfishTransformer phân tích CPL cho 30k ván.
- Lưu kết quả ra data/features/sample_30k_features.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from src.feature_engineering import FeaturePipeline
from src.feature_config import FeatureConfig, SAMPLE_SOURCE_FILE, SAMPLE_FEATURES_FILE


def main() -> None:
    t0 = time.time()
    print(f"{'═' * 60}")
    print("  CHẠY FEATURE PIPELINE TRÊN SAMPLE 30K")
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

    # Bước 4: Transform (bao gồm Stockfish CPL — tốn ~1.4 giờ)
    print(f"\n  Đang transform {df.height} ván...")
    print(f"  (Ước tính: ~{df.height * 0.17 / 60:.0f} phút)")
    result = pipeline.transform(df)

    # Bước 5: Lưu kết quả
    SAMPLE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(SAMPLE_FEATURES_FILE), compression="zstd")

    # Bước 6: Lưu metadata
    feature_cols = [c for c in result.columns if c != "ModelBand"]
    pipeline.save_metadata(feature_cols)

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✅ HOÀN THÀNH!")
    print(f"  Output: {SAMPLE_FEATURES_FILE}")
    print(f"  Rows: {result.height}, Cols: {result.width}")
    print(f"  Thời gian: {elapsed/60:.1f} phút ({elapsed/3600:.1f} giờ)")
    print(f"\n  Schema:")
    for name, dtype in result.schema.items():
        print(f"    {name}: {dtype}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
