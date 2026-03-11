"""Cấu hình trung tâm cho Feature Engineering pipeline.

Mục tiêu:
- Gom toàn bộ hằng số/cấu hình FE vào một nơi.
- Tránh hard-code rải rác trong các module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_DIR


# Cấu hình feature mặc định
N_MOVES_DEFAULT = 15
ECO_TOP_N_DEFAULT = 100
TFIDF_SAMPLE_SIZE_DEFAULT = 500_000
SVD_DIM_DEFAULT = 50
BATCH_SIZE_DEFAULT = 10_000_000
TFIDF_MAX_FEATURES_DEFAULT = 500
TFIDF_NGRAM_MIN_DEFAULT = 1
TFIDF_NGRAM_MAX_DEFAULT = 2
TFIDF_MIN_DF_DEFAULT = 2
TFIDF_MAX_DF_DEFAULT = 0.95


# Bins mục tiêu ModelBand theo EloAvg: [0, 1000, 1400, 1800, 2200, +inf)
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


# Cột đầu vào cốt lõi cho pipeline
INPUT_COLUMNS_REQUIRED = (
    "ECO",
    "GameFormat",
    "BaseTime",
    "Increment",
    "NumMoves",
    "Moves",
)


# Cột cần cho target encoder (ít nhất một trong hai cột phải tồn tại)
TARGET_SOURCE_COLUMNS = ("ModelBand", "EloAvg")


# Đường dẫn mặc định
TRAIN_SOURCE_FILE = DATA_PROCESSED / "lichess_2025-12_ml.parquet"
VAL_SOURCE_FILE = DATA_PROCESSED / "lichess_2026-01_ml.parquet"

FEATURES_DIR = DATA_DIR / "features"
TRAIN_FEATURES_FILE = FEATURES_DIR / "train_features.parquet"
VAL_FEATURES_FILE = FEATURES_DIR / "val_features.parquet"
TFIDF_VOCAB_FILE = FEATURES_DIR / "tfidf_vocabulary.pkl"
SVD_COMPONENTS_FILE = FEATURES_DIR / "svd_components.pkl"
FEATURE_COLUMNS_FILE = FEATURES_DIR / "feature_columns.json"


@dataclass(slots=True)
class FeatureConfig:
    """Cấu hình thực thi cho FeaturePipeline."""

    batch_size: int = BATCH_SIZE_DEFAULT
    eco_top_n: int = ECO_TOP_N_DEFAULT
    n_ply: int = N_MOVES_DEFAULT
    tfidf_sample_size: int = TFIDF_SAMPLE_SIZE_DEFAULT
    tfidf_max_features: int = TFIDF_MAX_FEATURES_DEFAULT
    tfidf_ngram_min: int = TFIDF_NGRAM_MIN_DEFAULT
    tfidf_ngram_max: int = TFIDF_NGRAM_MAX_DEFAULT
    tfidf_min_df: int | float = TFIDF_MIN_DF_DEFAULT
    tfidf_max_df: int | float = TFIDF_MAX_DF_DEFAULT
    svd_dim: int = SVD_DIM_DEFAULT
    realtime_mode: bool = True
    random_seed: int = 42

    min_required_ply: int = 5
    drop_rate_pass_threshold: float = 1.5
    drop_rate_fail_threshold: float = 3.0

    # Feature flags cho các nhóm cột
    enable_tabular_features: bool = True
    enable_move_features: bool = True
    enable_eco_features: bool = True
    enable_game_format_features: bool = True
    enable_numeric_features: bool = True

    # Input/output
    train_source_file: Path = TRAIN_SOURCE_FILE
    val_source_file: Path = VAL_SOURCE_FILE
    train_features_file: Path = TRAIN_FEATURES_FILE
    val_features_file: Path = VAL_FEATURES_FILE
    tfidf_vocab_file: Path = TFIDF_VOCAB_FILE
    svd_components_file: Path = SVD_COMPONENTS_FILE
    feature_columns_file: Path = FEATURE_COLUMNS_FILE

    required_columns: tuple[str, ...] = INPUT_COLUMNS_REQUIRED
