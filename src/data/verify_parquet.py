"""
verify_parquet.py  —  Kiểm tra chất lượng Parquet output
==========================================================
Đối chiếu file Parquet cuối cùng với yêu cầu trong CHESS_DATA_STRATEGY.md
để đảm bảo pipeline đã thực hiện đầy đủ.

Cách chạy:
  conda activate mining
  python src/verify_parquet.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from src.config import DATA_PROCESSED, PGN_JSONL

PARQUET_FILE = DATA_PROCESSED / "lichess_2026-01_ml.parquet"

# ═══════════════════════════════════════════════════════════
# Checklist theo CHESS_DATA_STRATEGY.md
# ═══════════════════════════════════════════════════════════

# Gold Features bắt buộc phải có (hoặc dạng biến đổi tương đương)
REQUIRED_GOLD = {
    "WhiteElo":        "Target dự đoán Elo trắng",
    "BlackElo":        "Target dự đoán Elo đen",
    "Result":          "Kết quả ván đấu (categorical)",
    "ResultNumeric":   "Result mapped → số (1.0/0.5/0.0) cho ML",
    "Moves":           "Chuỗi nước đi SAN — input chính",
    "ECO":             "Mã khai cuộc",
    "WhiteRatingDiff": "Thay đổi Elo trắng sau trận",
    "BlackRatingDiff": "Thay đổi Elo đen sau trận",
    "Termination":     "Lý do kết thúc (Normal, Time forfeit...)",
}

# TimeControl đã được parse thành 3 cột derived
DERIVED_FROM_TIMECONTROL = {
    "BaseTime":   "Thời gian gốc (giây), parse từ TimeControl",
    "Increment":  "Thời gian cộng thêm (giây), parse từ TimeControl",
    "GameFormat": "Phân loại: UltraBullet/Bullet/Blitz/Rapid/Classical",
}

# Cột bổ sung (từ preprocessing pipeline)
EXTRA_FEATURES = {
    "EloAvg":   "Trung bình Elo 2 bên — tiện filter theo skill level",
    "NumMoves": "Số nước đi — tiện filter downstream",
}

# Cột phải bị loại bỏ (Garbage theo strategy)
MUST_BE_REMOVED = [
    "Site", "Event", "Round", "Date", "UTCDate", "UTCTime", "Time",
    "Opening", "White", "Black", "GameID",
    "TimeControl",  # Đã parse → 3 cột mới
]

# Kiểu dữ liệu mong đợi
EXPECTED_DTYPES = {
    "WhiteElo":        pl.Int16,
    "BlackElo":        pl.Int16,
    "EloAvg":          pl.Int16,
    "NumMoves":        pl.Int16,
    "WhiteRatingDiff": pl.Int16,
    "BlackRatingDiff": pl.Int16,
    "BaseTime":        pl.Int16,
    "Increment":       pl.Int16,
    "ResultNumeric":   pl.Float32,
    "Result":          pl.Categorical,
    "ECO":             pl.Categorical,
    "Termination":     pl.Categorical,
    "GameFormat":      pl.Categorical,
    "Moves":           pl.String,
}

# Result values hợp lệ
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
VALID_RESULT_NUMERIC = {1.0, 0.5, 0.0}

# GameFormat hợp lệ
VALID_FORMATS = {"UltraBullet", "Bullet", "Blitz", "Rapid", "Classical", "Unknown"}


def print_header(title: str) -> None:
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def print_check(passed: bool, msg: str) -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon}  {msg}")


def run_verification() -> None:
    """Chạy toàn bộ kiểm tra và trả về tổng kết PASS/FAIL."""

    total_checks = 0
    passed_checks = 0
    failed_details = []

    def check(passed: bool, msg: str) -> None:
        nonlocal total_checks, passed_checks
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            failed_details.append(msg)
        print_check(passed, msg)

    # ────────────────────────────────────────────────────
    # 0. File tồn tại
    # ────────────────────────────────────────────────────
    print_header("0. KIỂM TRA FILE")

    parquet_exists = PARQUET_FILE.exists()
    check(parquet_exists, f"File Parquet tồn tại: {PARQUET_FILE.name}")

    if not parquet_exists:
        print("\n  ✗✗✗  KHÔNG TÌM THẤY FILE PARQUET — DỪNG KIỂM TRA  ✗✗✗")
        return

    file_size_gb = os.path.getsize(PARQUET_FILE) / (1024**3)
    check(file_size_gb > 1.0, f"Dung lượng hợp lý: {file_size_gb:.2f} GB (> 1 GB)")

    # Đọc schema (lazy, không load data)
    lf = pl.scan_parquet(PARQUET_FILE)
    schema = lf.collect_schema()
    col_names = list(schema.keys())

    # ────────────────────────────────────────────────────
    # 1. Gold Features — bắt buộc có
    # ────────────────────────────────────────────────────
    print_header("1. GOLD FEATURES (Bắt buộc — CHESS_DATA_STRATEGY.md)")

    for col, desc in REQUIRED_GOLD.items():
        check(col in col_names, f"{col:<20} — {desc}")

    # ────────────────────────────────────────────────────
    # 2. Cột derived từ TimeControl
    # ────────────────────────────────────────────────────
    print_header("2. CỘT DERIVED TỪ TimeControl")

    for col, desc in DERIVED_FROM_TIMECONTROL.items():
        check(col in col_names, f"{col:<20} — {desc}")

    # ────────────────────────────────────────────────────
    # 3. Cột bổ sung
    # ────────────────────────────────────────────────────
    print_header("3. CỘT BỔ SUNG (preprocessing)")

    for col, desc in EXTRA_FEATURES.items():
        check(col in col_names, f"{col:<20} — {desc}")

    # ────────────────────────────────────────────────────
    # 4. Cột phải bị loại bỏ (Garbage)
    # ────────────────────────────────────────────────────
    print_header("4. CỘT ĐÃ LOẠI BỎ (Garbage)")

    for col in MUST_BE_REMOVED:
        check(col not in col_names, f"{col:<20} — đã bị loại bỏ")

    # ────────────────────────────────────────────────────
    # 5. Kiểu dữ liệu
    # ────────────────────────────────────────────────────
    print_header("5. KIỂU DỮ LIỆU (dtype optimization)")

    for col, expected in EXPECTED_DTYPES.items():
        if col in schema:
            actual = schema[col]
            check(
                actual == expected,
                f"{col:<20} → {actual} {'==' if actual == expected else '!='} {expected}",
            )
        else:
            check(False, f"{col:<20} — THIẾU CỘT, không kiểm tra dtype được")

    # ────────────────────────────────────────────────────
    # 6. Tổng số dòng
    # ────────────────────────────────────────────────────
    print_header("6. SỐ LƯỢNG DỮ LIỆU")

    total_rows = lf.select(pl.len()).collect().item()
    check(total_rows > 0, f"Tổng số dòng: {total_rows:,}")

    # So sánh với JSONL (nếu có)
    if PGN_JSONL.exists():
        # Đếm nhanh dòng JSONL
        jsonl_lines = 0
        with open(PGN_JSONL, "rb") as f:
            buf_size = 1 << 20
            while True:
                buf = f.read(buf_size)
                if not buf:
                    break
                jsonl_lines += buf.count(b"\n")

        match_pct = total_rows / jsonl_lines * 100 if jsonl_lines > 0 else 0
        check(
            match_pct > 99.0,
            f"Khớp JSONL: {total_rows:,} / {jsonl_lines:,} dòng ({match_pct:.2f}%)",
        )
    else:
        print(f"  ⓘ  Không tìm thấy JSONL để so sánh số dòng")

    # ────────────────────────────────────────────────────
    # 7. Kiểm tra giá trị — sample 100K dòng
    # ────────────────────────────────────────────────────
    print_header("7. KIỂM TRA GIÁ TRỊ (sample 100K dòng)")

    sample = lf.head(100_000).collect()

    # 7a. Result values
    result_vals = set(sample["Result"].unique().to_list())
    check(
        result_vals.issubset(VALID_RESULTS),
        f"Result values hợp lệ: {result_vals}",
    )

    # 7b. ResultNumeric values
    rn_vals = set(sample["ResultNumeric"].drop_nulls().unique().to_list())
    check(
        rn_vals.issubset(VALID_RESULT_NUMERIC),
        f"ResultNumeric values: {rn_vals}",
    )

    # 7c. ResultNumeric null count
    rn_null = sample["ResultNumeric"].null_count()
    rn_null_pct = rn_null / len(sample) * 100
    check(
        rn_null_pct < 1.0,
        f"ResultNumeric null: {rn_null:,} / {len(sample):,} ({rn_null_pct:.2f}%)",
    )

    # 7d. Elo range hợp lý (Lichess: ~600 – ~3500)
    for elo_col in ["WhiteElo", "BlackElo"]:
        elo_min = sample[elo_col].min()
        elo_max = sample[elo_col].max()
        elo_null = sample[elo_col].null_count()
        reasonable = (elo_min is not None and elo_min >= 0 and
                      elo_max is not None and elo_max <= 4000)
        check(
            reasonable,
            f"{elo_col}: range [{elo_min}, {elo_max}], nulls={elo_null:,}",
        )

    # 7e. NumMoves > 0 (đã filter min_moves=5)
    nm_min = sample["NumMoves"].min()
    check(
        nm_min is not None and nm_min >= 5,
        f"NumMoves min = {nm_min} (phải >= 5 theo filter)",
    )

    # 7f. Moves không rỗng
    moves_empty = sample.filter(
        pl.col("Moves").is_null() | (pl.col("Moves").str.len_chars() == 0)
    ).height
    check(
        moves_empty == 0,
        f"Moves rỗng/null: {moves_empty} / {len(sample):,}",
    )

    # 7g. ECO format (thường là A00-E99)
    eco_sample = sample["ECO"].drop_nulls().head(10).to_list()
    eco_valid = all(
        len(e) >= 2 and e[0].isalpha() and e[1:3].isdigit()
        for e in eco_sample if e
    )
    check(eco_valid, f"ECO format hợp lệ (mẫu: {eco_sample[:5]})")

    # 7h. GameFormat values
    gf_vals = set(sample["GameFormat"].unique().to_list())
    check(
        gf_vals.issubset(VALID_FORMATS),
        f"GameFormat values: {gf_vals}",
    )

    # 7i. BaseTime, Increment không âm
    for col in ["BaseTime", "Increment"]:
        col_min = sample[col].drop_nulls().min()
        check(
            col_min is not None and col_min >= 0,
            f"{col} min = {col_min} (không âm)",
        )

    # ────────────────────────────────────────────────────
    # 8. Thống kê phân phối nhanh
    # ────────────────────────────────────────────────────
    print_header("8. THỐNG KÊ PHÂN PHỐI (toàn bộ file)")

    # 8a. Phân bố Result
    result_dist = (
        lf.group_by("Result")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .collect()
    )
    print(f"\n  Phân bố Result:")
    for row in result_dist.iter_rows(named=True):
        pct = row["count"] / total_rows * 100
        print(f"    {row['Result']:<10} {row['count']:>12,}  ({pct:5.1f}%)")

    # 8b. Phân bố GameFormat
    format_dist = (
        lf.group_by("GameFormat")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .collect()
    )
    print(f"\n  Phân bố GameFormat:")
    for row in format_dist.iter_rows(named=True):
        pct = row["count"] / total_rows * 100
        print(f"    {row['GameFormat']:<14} {row['count']:>12,}  ({pct:5.1f}%)")

    # 8c. Phân bố Termination
    term_dist = (
        lf.group_by("Termination")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .collect()
    )
    print(f"\n  Phân bố Termination:")
    for row in term_dist.iter_rows(named=True):
        pct = row["count"] / total_rows * 100
        print(f"    {row['Termination']:<25} {row['count']:>12,}  ({pct:5.1f}%)")

    # 8d. Thống kê Elo
    elo_stats = (
        lf.select(
            pl.col("WhiteElo").mean().alias("WhiteElo_mean"),
            pl.col("WhiteElo").median().alias("WhiteElo_median"),
            pl.col("WhiteElo").std().alias("WhiteElo_std"),
            pl.col("BlackElo").mean().alias("BlackElo_mean"),
            pl.col("BlackElo").median().alias("BlackElo_median"),
            pl.col("BlackElo").std().alias("BlackElo_std"),
            pl.col("EloAvg").mean().alias("EloAvg_mean"),
            pl.col("EloAvg").min().alias("EloAvg_min"),
            pl.col("EloAvg").max().alias("EloAvg_max"),
            pl.col("NumMoves").mean().alias("NumMoves_mean"),
            pl.col("NumMoves").median().alias("NumMoves_median"),
        ).collect()
    )
    print(f"\n  Thống kê Elo:")
    for col_name in elo_stats.columns:
        val = elo_stats[col_name][0]
        if val is not None:
            print(f"    {col_name:<22} {val:>10.1f}")

    # ────────────────────────────────────────────────────
    # 9. Kiểm tra nén & hiệu suất
    # ────────────────────────────────────────────────────
    print_header("9. HIỆU SUẤT NÉN")

    jsonl_size_gb = os.path.getsize(PGN_JSONL) / (1024**3) if PGN_JSONL.exists() else 0
    parquet_size_gb = file_size_gb
    compression_ratio = (1 - parquet_size_gb / jsonl_size_gb) * 100 if jsonl_size_gb > 0 else 0

    print(f"  JSONL  : {jsonl_size_gb:.2f} GB")
    print(f"  Parquet: {parquet_size_gb:.2f} GB")
    print(f"  Tỷ lệ nén: {compression_ratio:.1f}%")

    bytes_per_row = (parquet_size_gb * 1024**3) / total_rows if total_rows > 0 else 0
    print(f"  Bytes/dòng: {bytes_per_row:.0f}")

    check(
        compression_ratio > 50,
        f"Tỷ lệ nén > 50%: {compression_ratio:.1f}%",
    )

    # ════════════════════════════════════════════════════
    # TỔNG KẾT
    # ════════════════════════════════════════════════════
    print(f"\n{'═'*64}")
    print(f"  TỔNG KẾT KIỂM TRA")
    print(f"{'─'*64}")
    print(f"  Tổng checks : {total_checks}")
    print(f"  PASSED      : {passed_checks}")
    print(f"  FAILED      : {total_checks - passed_checks}")

    if failed_details:
        print(f"\n  Chi tiết FAILED:")
        for msg in failed_details:
            print(f"    ✗  {msg}")

    status = "PASS ✓✓✓" if passed_checks == total_checks else "FAIL ✗✗✗"
    print(f"\n  KẾT QUẢ: {status}")
    print(f"{'═'*64}")

    # Exit code cho CI/CD
    sys.exit(0 if passed_checks == total_checks else 1)


if __name__ == "__main__":
    run_verification()
