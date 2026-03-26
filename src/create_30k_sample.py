"""Trích xuất mẫu 30.000 ván cờ phân bổ đều theo 5 nhóm ELO.

Chiến lược tối ưu RAM:
- Đọc từng file Parquet một (không gộp 2 file 45GB).
- Dùng pyarrow iter_batches để đọc chunk nhỏ 1M rows.
- Gom lại đủ quota mỗi band thì dừng sớm.
"""

from __future__ import annotations

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import pyarrow.parquet as pq

from src.config import DATA_PROCESSED
from src.feature_config import MODEL_BINS

# ── Cấu hình ──────────────────────────────────────────
N_PER_CLASS = 6_000
NUM_CLASSES = 5
TOTAL_SAMPLE = N_PER_CLASS * NUM_CLASSES  # 30.000

READ_BATCH_SIZE = 1_000_000  # Đọc 1M rows/lần cho nhẹ RAM

# Giữ TẤT CẢ cột trong output (để tham khảo/debug)
# Nhưng sẽ LỌC dữ liệu theo Termination và GameFormat
KEEP_COLUMNS = [
    "WhiteElo",
    "BlackElo",
    "EloAvg",
    "NumMoves",
    "ECO",
    "Termination",
    "GameFormat",
    "BaseTime",
    "Increment",
    "Moves",
]

# Điều kiện lọc: CHỈ lấy ván kết thúc bình thường, KHÔNG cờ chớp
EXCLUDED_TERMINATIONS = {"Time forfeit"}
EXCLUDED_FORMATS = {"Bullet", "UltraBullet", "Blitz"}

INPUT_FILES = [
    DATA_PROCESSED / "lichess_2025-12_ml.parquet",
    DATA_PROCESSED / "lichess_2026-01_ml.parquet",
]

OUTPUT_FILE = DATA_PROCESSED / "sample_30k.parquet"


def elo_to_band(elo: int) -> int:
    """Map EloAvg sang ModelBand id."""
    if elo < MODEL_BINS[1]:
        return 0
    if elo < MODEL_BINS[2]:
        return 1
    if elo < MODEL_BINS[3]:
        return 2
    if elo < MODEL_BINS[4]:
        return 3
    return 4


def main() -> None:
    t0 = time.time()
    print(f"{'═' * 60}")
    print("  TẠO MẪU 30.000 VÁN CỜ PHÂN BỔ ĐỀU THEO ELO")
    print(f"{'─' * 60}")

    random.seed(42)

    # Reservoir sampling — giữ tối đa N_PER_CLASS cho mỗi band
    # Dùng reservoir sampling để không cần load toàn bộ data vào RAM
    reservoirs: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}

    existing_files = [f for f in INPUT_FILES if f.exists()]
    if not existing_files:
        print("  LỖI: Không tìm thấy file Parquet nguồn!")
        return

    print(f"  Nguồn: {len(existing_files)} file(s)")

    for file_path in existing_files:
        print(f"\n  Đang xử lý: {file_path.name}")
        pf = pq.ParquetFile(str(file_path))
        available_cols = [c for c in KEEP_COLUMNS if c in pf.schema.names]

        batch_idx = 0
        for record_batch in pf.iter_batches(
            batch_size=READ_BATCH_SIZE, columns=available_cols
        ):
            batch_idx += 1
            df = pl.from_arrow(record_batch)

            # Lấy các cột cần thiết để lọc
            elo_avg = df["EloAvg"].to_list()
            terminations = df["Termination"].to_list() if "Termination" in df.columns else [None] * df.height
            formats = df["GameFormat"].to_list() if "GameFormat" in df.columns else [None] * df.height

            for row_idx in range(df.height):
                elo = elo_avg[row_idx]
                if elo is None:
                    continue

                # LỌC: Bỏ ván chết đồng hồ
                term = str(terminations[row_idx]) if terminations[row_idx] is not None else ""
                if term in EXCLUDED_TERMINATIONS:
                    continue

                # LỌC: Bỏ ván cờ chớp / siêu chớp
                fmt = str(formats[row_idx]) if formats[row_idx] is not None else ""
                if fmt in EXCLUDED_FORMATS:
                    continue

                band = elo_to_band(int(elo))
                counts[band] += 1
                n = counts[band]

                if len(reservoirs[band]) < N_PER_CLASS:
                    # Chưa đầy quota → thêm trực tiếp
                    reservoirs[band].append(df.row(row_idx, named=True))
                else:
                    # Reservoir sampling: thay thế ngẫu nhiên
                    j = random.randint(0, n - 1)
                    if j < N_PER_CLASS:
                        reservoirs[band][j] = df.row(row_idx, named=True)

            # In tiến độ
            filled = sum(min(len(r), N_PER_CLASS) for r in reservoirs.values())
            print(
                f"    Batch {batch_idx}: "
                f"đã gom {filled}/{TOTAL_SAMPLE} ván "
                f"[{', '.join(f'B{i}:{len(reservoirs[i])}' for i in range(NUM_CLASSES))}]",
                flush=True,
            )

            # Kiểm tra đã đủ chưa (tối thiểu mỗi band có đủ quota)
            all_full = all(len(reservoirs[i]) >= N_PER_CLASS for i in range(NUM_CLASSES))
            # Sau khi đủ quota, tiếp tục thêm vài batch nữa để reservoir sampling
            # có cơ hội thay thế (diversify). Dừng sau khi gom ít nhất 3x quota.
            if all_full and all(counts[i] >= N_PER_CLASS * 3 for i in range(NUM_CLASSES)):
                print("    ✓ Đủ quota cho tất cả bands, dừng sớm!")
                break

    # Gộp tất cả
    all_rows: list[dict] = []
    band_names = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    for band_id in range(NUM_CLASSES):
        n = len(reservoirs[band_id])
        print(f"  Band {band_id} ({band_names[band_id]}): {n} ván")
        all_rows.extend(reservoirs[band_id])

    final_df = pl.DataFrame(all_rows)

    # Thêm cột ModelBand
    final_df = final_df.with_columns(
        pl.col("EloAvg")
        .cast(pl.Int32)
        .map_elements(elo_to_band, return_dtype=pl.Int8)
        .alias("ModelBand")
    )

    # Shuffle
    final_df = final_df.sample(fraction=1.0, seed=42)

    # Kiểm tra
    print(f"\n  Tổng mẫu: {final_df.height}")
    print(final_df.group_by("ModelBand").len().sort("ModelBand"))

    # Lưu
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(str(OUTPUT_FILE), compression="zstd")

    elapsed = time.time() - t0
    print(f"\n  ✅ Đã lưu: {OUTPUT_FILE}")
    print(f"  Thời gian: {elapsed:.1f}s")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
