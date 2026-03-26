"""Cấu hình trung tâm cho Feature Engineering pipeline.

Phiên bản mới (Stockfish CPL):
- Loại bỏ hoàn toàn: TFIDF_*, SVD_*, N_MOVES_*
- Loại bỏ hoàn toàn: GameFormat, BaseTime, Increment khỏi required columns
- Thêm mới: STOCKFISH_* (đường dẫn engine, depth, threads)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_DIR


# ── Cấu hình Stockfish Engine ──────────────────────────────
STOCKFISH_PATH_DEFAULT = str(
    Path(__file__).resolve().parent.parent / ".tmp" / "stockfish_binary"
)
STOCKFISH_DEPTH_DEFAULT = 10
STOCKFISH_THREADS_DEFAULT = 1
STOCKFISH_HASH_MB_DEFAULT = 128

# ── Cấu hình tabular ───────────────────────────────────────
ECO_TOP_N_DEFAULT = 100
BATCH_SIZE_DEFAULT = 30_000  # Sample 30k, 1 batch đủ

# ── Bins mục tiêu ModelBand theo EloAvg ─────────────────────
MODEL_BINS = (0, 1000, 1400, 1800, 2200)
MODEL_BAND_LABELS = (
    "Beginner",
    "Intermediate",
    "Advanced",
    "Expert",
    "Master",
)
MODEL_BAND_LABEL_TO_ID = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2,
    "Expert": 3,
    "Master": 4,
}

# ── Cột đầu vào cốt lõi (ĐÃ TINH GỌN) ────────────────────
# Loại bỏ: GameFormat, BaseTime, Increment (Design Decision 4)
INPUT_COLUMNS_REQUIRED = (
    "ECO",
    "NumMoves",
    "Moves",
)

# Cột cần cho target encoder
TARGET_SOURCE_COLUMNS = ("ModelBand", "EloAvg")

# ── Đường dẫn mặc định ─────────────────────────────────────
SAMPLE_SOURCE_FILE = DATA_PROCESSED / "sample_30k.parquet"
FEATURES_DIR = DATA_DIR / "features"
SAMPLE_FEATURES_FILE = FEATURES_DIR / "sample_30k_features.parquet"
FEATURE_COLUMNS_FILE = FEATURES_DIR / "feature_columns.json"


@dataclass(slots=True)
class FeatureConfig:
    """Cấu hình thực thi cho FeaturePipeline."""

    # Tabular
    batch_size: int = BATCH_SIZE_DEFAULT
    eco_top_n: int = ECO_TOP_N_DEFAULT

    # Stockfish Engine
    stockfish_path: str = STOCKFISH_PATH_DEFAULT
    stockfish_depth: int = STOCKFISH_DEPTH_DEFAULT
    stockfish_threads: int = STOCKFISH_THREADS_DEFAULT
    stockfish_hash_mb: int = STOCKFISH_HASH_MB_DEFAULT

    # Chung
    random_seed: int = 42

    # Input/output
    sample_source_file: Path = SAMPLE_SOURCE_FILE
    sample_features_file: Path = SAMPLE_FEATURES_FILE
    feature_columns_file: Path = FEATURE_COLUMNS_FILE

    required_columns: tuple[str, ...] = INPUT_COLUMNS_REQUIRED
