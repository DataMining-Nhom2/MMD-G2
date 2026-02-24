"""
src/config.py
=============
Single source of truth cho tất cả đường dẫn trong project.

Cách dùng trong bất kỳ file nào (sau pip install -e .):

    from src.config import DATA_RAW, DATA_PROCESSED, EDA_OUTPUTS

    df = pd.read_csv(DATA_RAW / "accepted_2007_to_2018Q4.csv")
    pgn = DATA_RAW / "lichess_db_standard_rated_2025-12.pgn.zst"

Không cần hardcode path, không cần os.chdir(), không cần sys.path.append().
==============
TODO: change in the future: use environment variables or a config file for more flexibility.
"""

from pathlib import Path
import os

# ── Root của project (thư mục chứa pyproject.toml) ──────────
# _file_ = .../project/src/config.py
# .parent   = .../project/src/
# .parent   = .../project/          ← ROOT
ROOT = Path(__file__).resolve().parent.parent

# ── Cho phép override ROOT bằng env var nếu cần ─────────────
# Ví dụ: export PROJECT_ROOT=/mnt/data/chess-ml
if os.getenv("PROJECT_ROOT"):
    ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()

# ── Data ────────────────────────────────────────────────────
DATA_DIR       = ROOT / "data"
DATA_RAW       = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# File cụ thể — đổi tên tháng ở đây 1 lần duy nhất
PGN_ZST        = DATA_RAW / "lichess_db_standard_rated_2025-12.pgn.zst"
LOAN_CSV       = DATA_RAW / "accepted_2007_to_2018Q4.csv"

# Output của convert script
PGN_JSONL      = DATA_PROCESSED / "lichess_db_standard_rated_2025-12_converted.jsonl"
PGN_ERROR_LOG  = DATA_PROCESSED / "convert_errors.log"
PGN_SKIP_LOG   = DATA_PROCESSED / "convert_skipped.log"

# Output Parquet (Phương án B: JSONL → Parquet)
PARQUET_ML     = DATA_PROCESSED / "lichess_2025-12_ml.parquet"

# ── EDA ─────────────────────────────────────────────────────
EDA_DIR        = ROOT / "eda"
EDA_OUTPUTS    = EDA_DIR / "outputs"

# ── Source ──────────────────────────────────────────────────
SRC_DIR        = ROOT / "src"

# ── Docs ────────────────────────────────────────────────────
DOCS_DIR       = ROOT / "docs"

# ── Tự động tạo thư mục output nếu chưa có ─────────────────
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
EDA_OUTPUTS.mkdir(parents=True, exist_ok=True)


# ── Kiểm tra nhanh khi chạy trực tiếp ───────────────────────
if __name__ == "__main__":
    paths = {
        "ROOT"          : ROOT,
        "DATA_RAW"      : DATA_RAW,
        "DATA_PROCESSED": DATA_PROCESSED,
        "EDA_OUTPUTS"   : EDA_OUTPUTS,
        "PGN_ZST"       : PGN_ZST,
        "LOAN_CSV"      : LOAN_CSV,
    }
    print(f"{'─'*55}")
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗ NOT FOUND"
        print(f"  {name:<18} {exists}  {path}")
    print(f"{'─'*55}")