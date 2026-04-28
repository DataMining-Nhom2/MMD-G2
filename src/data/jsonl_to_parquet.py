"""
jsonl_to_parquet.py  —  Phương án B
=====================================
Chuyển đổi JSONL (67 GB) → Parquet tối ưu cho ML.

Tối ưu cho: i9-14900HX, 16 GB RAM
Thời gian ước tính: ~20-40 phút

Kiến trúc:
  [JSONL 67GB] → Polars scan_ndjson (streaming/lazy)
              → Drop cột thừa (Event, Date, Time, Opening, White, Black, GameID)
              → Ép kiểu tối ưu (Int16, Int8, Categorical)
              → Map Result → số (1.0 / 0.5 / 0.0)
              → Parse TimeControl → base_time + increment
              → Ghi Parquet (snappy, row_group_size=500K)

Cách chạy:
  conda activate mining
  python src/jsonl_to_parquet.py
"""

import sys
import time
import os
from pathlib import Path

# Đảm bảo import config từ project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from src.config import (
    PGN_JSONL, DATA_PROCESSED,
)

# ╔══════════════════════════════════════════════════════════╗
# ║                        CẤU HÌNH                         ║
# ╚══════════════════════════════════════════════════════════╝

INPUT_JSONL  = PGN_JSONL
OUTPUT_DIR   = DATA_PROCESSED
OUTPUT_FILE  = OUTPUT_DIR / "lichess_2025-12_ml.parquet"

# -- Cột giữ lại (Gold Features theo CHESS_DATA_STRATEGY.md) --
# Loại bỏ: Event, Date, Time, White, Black, GameID, Opening
KEEP_COLUMNS = [
    "Result",
    "WhiteElo",
    "BlackElo",
    "EloAvg",
    "NumMoves",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "ECO",
    "TimeControl",
    "Termination",
    "Moves",
]

# -- Xử lý chunk (tránh OOM trên 16GB RAM) --
# Đọc JSONL theo batch, mỗi batch xử lý rồi ghi append vào Parquet
BATCH_SIZE     = 500_000   # 500K dòng/batch (~4-5 GB RAM peak)
ROW_GROUP_SIZE = 500_000   # Row group size trong Parquet

# -- Map Result sang dạng số cho ML --
RESULT_MAP = {
    "1-0"     : 1.0,   # Trắng thắng
    "0-1"     : 0.0,   # Đen thắng
    "1/2-1/2" : 0.5,   # Hòa
}


# ╔══════════════════════════════════════════════════════════╗
# ║                   HÀM XỬ LÝ CHÍNH                       ║
# ╚══════════════════════════════════════════════════════════╝

def parse_time_control(tc: str) -> tuple[int | None, int | None]:
    """
    Parse TimeControl string (ví dụ "300+3") thành (base_seconds, increment).
    Trả về (None, None) nếu không parse được.
    """
    if not tc or tc == "-":
        return None, None
    parts = tc.split("+")
    try:
        base = int(parts[0])
        inc  = int(parts[1]) if len(parts) > 1 else 0
        return base, inc
    except (ValueError, IndexError):
        return None, None


def classify_time_control(base: int | None, inc: int | None) -> str:
    """
    Phân loại format cờ dựa trên thời gian:
      - UltraBullet: < 30s
      - Bullet: 30s - <3 phút
      - Blitz: 3-8 phút
      - Rapid: 8-25 phút
      - Classical: > 25 phút
    Công thức Lichess: estimated_time = base + 40 * increment
    """
    if base is None:
        return "Unknown"
    estimated = base + 40 * (inc or 0)
    if estimated < 30:
        return "UltraBullet"
    elif estimated < 180:
        return "Bullet"
    elif estimated < 480:
        return "Blitz"
    elif estimated < 1500:
        return "Rapid"
    else:
        return "Classical"


def transform_batch(df: pl.DataFrame) -> pl.DataFrame:
    """
    Áp dụng tất cả biến đổi cho một batch DataFrame.
    """
    # 1. Chỉ giữ cột cần thiết (bỏ qua cột không tồn tại)
    existing_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df.select(existing_cols)

    # 2. Map Result → số
    df = df.with_columns(
        pl.col("Result")
          .replace_strict(RESULT_MAP, default=None)
          .cast(pl.Float32)
          .alias("ResultNumeric"),
    )

    # 3. Ép kiểu WhiteRatingDiff, BlackRatingDiff → Int16
    #    (trong JSONL chúng là string như "+5", "-6")
    df = df.with_columns(
        pl.col("WhiteRatingDiff").cast(pl.Int16, strict=False),
        pl.col("BlackRatingDiff").cast(pl.Int16, strict=False),
    )

    # 4. Ép kiểu Elo → Int16, NumMoves → Int16
    df = df.with_columns(
        pl.col("WhiteElo").cast(pl.Int16, strict=False),
        pl.col("BlackElo").cast(pl.Int16, strict=False),
        pl.col("EloAvg").cast(pl.Int16, strict=False),
        pl.col("NumMoves").cast(pl.Int16, strict=False),
    )

    # 5. Parse TimeControl → base_time, increment, game_format
    tc_parsed = df["TimeControl"].to_list()
    bases, incs, formats = [], [], []
    for tc in tc_parsed:
        b, i = parse_time_control(tc)
        bases.append(b)
        incs.append(i)
        formats.append(classify_time_control(b, i))

    df = df.with_columns(
        pl.Series("BaseTime", bases, dtype=pl.Int16),
        pl.Series("Increment", incs, dtype=pl.Int16),
        pl.Series("GameFormat", formats, dtype=pl.Utf8),
    )

    # 6. Cast categorical columns
    df = df.with_columns(
        pl.col("Result").cast(pl.Categorical),
        pl.col("ECO").cast(pl.Categorical),
        pl.col("Termination").cast(pl.Categorical),
        pl.col("GameFormat").cast(pl.Categorical),
    )

    # 7. Loại bỏ cột TimeControl gốc (đã parse xong)
    df = df.drop("TimeControl")

    return df


def count_lines_fast(filepath: Path) -> int:
    """Đếm nhanh số dòng của file lớn (dùng buffer thô)."""
    count = 0
    buf_size = 1 << 20  # 1 MB
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            count += buf.count(b"\n")
    return count


def convert_jsonl_to_parquet() -> None:
    """Pipeline chính: đọc JSONL theo batch → transform → ghi Parquet."""

    if not INPUT_JSONL.exists():
        print(f"  Lỗi: Không tìm thấy file '{INPUT_JSONL}'")
        return

    file_size_gb = os.path.getsize(INPUT_JSONL) / (1024**3)

    print(f"\n{'═'*64}")
    print(f"  JSONL → Parquet Converter  (Phương án B)")
    print(f"{'─'*64}")
    print(f"  Input    : {INPUT_JSONL}")
    print(f"           : {file_size_gb:.1f} GB")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  Batch    : {BATCH_SIZE:,} dòng/batch")
    print(f"{'═'*64}")

    # Bước 1: Đếm tổng số dòng để hiển thị tiến độ
    print(f"\n  [1/3] Đếm số dòng trong JSONL...", end=" ", flush=True)
    t0 = time.time()
    total_lines = count_lines_fast(INPUT_JSONL)
    print(f"{total_lines:,} dòng ({time.time()-t0:.1f}s)")

    total_batches = (total_lines + BATCH_SIZE - 1) // BATCH_SIZE

    # Bước 2: Đọc + transform + ghi Parquet
    print(f"  [2/3] Chuyển đổi {total_batches} batch → Parquet...")
    start_time  = time.time()
    rows_done   = 0
    batch_idx   = 0
    parquet_parts = []  # Danh sách các file part tạm

    # Đọc JSONL theo batch bằng Polars scan
    # Polars ndjson reader: đọc toàn bộ file. Với 67GB, ta dùng
    # phương pháp: đọc chunk bằng Python → feed vào Polars DataFrame
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        while True:
            lines = []
            for _ in range(BATCH_SIZE):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    lines.append(line.encode("utf-8"))

            if not lines:
                break

            batch_idx += 1
            batch_start = time.time()

            # Parse JSON lines → Polars DataFrame
            # Polars có thể đọc từ bytes NDJSON
            ndjson_bytes = b"\n".join(lines)
            df = pl.read_ndjson(ndjson_bytes)

            # Áp dụng biến đổi
            df = transform_batch(df)

            # Ghi ra file Parquet part tạm
            part_file = OUTPUT_DIR / f"_part_{batch_idx:04d}.parquet"
            df.write_parquet(
                part_file,
                compression="snappy",
                row_group_size=ROW_GROUP_SIZE,
            )
            parquet_parts.append(part_file)

            rows_done += len(lines)
            elapsed    = time.time() - start_time
            batch_time = time.time() - batch_start
            speed      = rows_done / elapsed if elapsed > 0 else 0
            pct        = rows_done / total_lines * 100

            print(
                f"    Batch {batch_idx:>3}/{total_batches}"
                f"  | {rows_done:>12,}/{total_lines:,} ({pct:5.1f}%)"
                f"  | {speed:>8,.0f} dòng/s"
                f"  | batch {batch_time:.1f}s"
                f"  | tổng {elapsed/60:.1f} phút"
            )

            # Giải phóng RAM
            del df, ndjson_bytes, lines

    # Bước 3: Merge tất cả parts thành 1 file Parquet duy nhất
    print(f"\n  [3/3] Ghép {len(parquet_parts)} parts → file Parquet cuối cùng...")
    merge_start = time.time()

    # Đọc lazy tất cả parts rồi sink_parquet
    lazy_frames = [pl.scan_parquet(p) for p in parquet_parts]
    combined    = pl.concat(lazy_frames)
    combined.sink_parquet(
        OUTPUT_FILE,
        compression="snappy",
        row_group_size=ROW_GROUP_SIZE,
    )

    merge_time = time.time() - merge_start
    print(f"    Ghép xong ({merge_time:.1f}s)")

    # Dọn dẹp file parts tạm
    for p in parquet_parts:
        p.unlink(missing_ok=True)
    print(f"    Đã xóa {len(parquet_parts)} file tạm.")

    # Tổng kết
    total_time   = time.time() - start_time
    output_size  = os.path.getsize(OUTPUT_FILE) / (1024**3)
    compression  = (1 - output_size / file_size_gb) * 100

    print(f"\n{'═'*64}")
    print(f"  HOÀN THÀNH!")
    print(f"{'─'*64}")
    print(f"  Tổng dòng xử lý : {rows_done:,}")
    print(f"  Thời gian        : {total_time/60:.1f} phút ({total_time:.0f}s)")
    print(f"  Tốc độ TB        : {rows_done/total_time:,.0f} dòng/s")
    print(f"  Input JSONL      : {file_size_gb:.2f} GB")
    print(f"  Output Parquet   : {output_size:.2f} GB")
    print(f"  Tỷ lệ nén       : {compression:.1f}%")
    print(f"  File             : {OUTPUT_FILE}")
    print(f"{'═'*64}")


# ╔══════════════════════════════════════════════════════════╗
# ║                  KIỂM TRA NHANH                          ║
# ╚══════════════════════════════════════════════════════════╝

def verify_parquet(filepath: Path, sample_rows: int = 5) -> None:
    """Đọc Parquet và in thông tin tóm tắt để kiểm tra."""
    if not filepath.exists():
        print(f"  File không tồn tại: {filepath}")
        return

    print(f"\n{'═'*64}")
    print(f"  KIỂM TRA PARQUET OUTPUT")
    print(f"{'─'*64}")

    # Đọc schema (không load data)
    lf = pl.scan_parquet(filepath)
    schema = lf.collect_schema()

    print(f"  Cột ({len(schema)}):")
    for name, dtype in schema.items():
        print(f"    {name:<20} {dtype}")

    # Đọc vài dòng mẫu
    sample = lf.head(sample_rows).collect()
    print(f"\n  {sample_rows} dòng đầu tiên:")
    print(sample)

    # Thống kê tổng
    total_rows = lf.select(pl.len()).collect().item()
    print(f"\n  Tổng số dòng   : {total_rows:,}")

    size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"  Dung lượng file : {size_gb:.2f} GB")
    print(f"{'═'*64}")


# ╔══════════════════════════════════════════════════════════╗
# ║                        MAIN                              ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    convert_jsonl_to_parquet()
    print()
    verify_parquet(OUTPUT_FILE)
