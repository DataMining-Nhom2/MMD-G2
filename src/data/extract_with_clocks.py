"""Trích xuất 30.000 ván cờ từ file PGN.zst gốc — GIỮ NGUYÊN clock time (%clk).

Script này stream file PGN.zst, parse từng ván, reservoir sampling 6k/band × 5 bands.
Khác với create_30k_sample.py (đọc từ parquet đã bị strip clock), script này
đọc trực tiếp từ PGN gốc để giữ lại thông tin %clk cho mỗi nước đi.

Output: data/processed/sample_30k_dl.parquet
  - Moves (SAN string)
  - ClockSeq (JSON string: danh sách thời gian còn lại per-move, giây)
  - TimeSpentSeq (JSON string: danh sách thời gian suy nghĩ per-move, giây)
  - WhiteElo, BlackElo, EloAvg, NumMoves, TimeControl, Result, Termination, ECO, GameFormat
  - ModelBand (0-4)

Cách chạy:
  conda activate MMDS
  python -m src.data.extract_with_clocks
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chess.pgn
import polars as pl
import zstandard as zstd

from src.config import DATA_RAW, DATA_PROCESSED

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════

# File PGN.zst gốc
PGN_ZST_FILE = DATA_RAW / "lichess_db_standard_rated_2026-01.pgn.zst"

# Output
OUTPUT_FILE = DATA_PROCESSED / "sample_30k_dl.parquet"

# Sampling
N_PER_CLASS = 6_000
NUM_CLASSES = 5
TOTAL_SAMPLE = N_PER_CLASS * NUM_CLASSES  # 30.000

# ELO bins — giống feature_config.py
MODEL_BINS = (0, 1000, 1400, 1800, 2200)

# Filter tối thiểu (giữ gần hết, chỉ bỏ ván quá ngắn hoặc thiếu ELO)
MIN_MOVES = 5  # Bỏ ván dưới 5 ply (forfeit / abandoned)

# Dừng sớm sau khi sampling đủ diversified
OVERSAMPLE_FACTOR = 3  # Tiếp tục thêm 3x quota để reservoir đa dạng hơn

# I/O
ZST_READ_BLOCK = 1 << 18  # 256 KB

# Logging
LOG_INTERVAL = 50_000

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

_CLK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]')

_GAME_SEP_RE = re.compile(r'\n\n(?=\[Event )')


def _safe_int(value: str) -> int | None:
    """Elo string → int, None nếu không hợp lệ."""
    try:
        v = int(value)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def elo_to_band(elo: int) -> int:
    """Map EloAvg sang ModelBand id (0-4)."""
    if elo < MODEL_BINS[1]:
        return 0
    if elo < MODEL_BINS[2]:
        return 1
    if elo < MODEL_BINS[3]:
        return 2
    if elo < MODEL_BINS[4]:
        return 3
    return 4


def parse_clock_from_comment(comment: str) -> float | None:
    """Parse thời gian còn lại từ PGN comment.

    Ví dụ comment: '[%clk 0:09:54]' → 594.0 (giây)
    """
    match = _CLK_RE.search(comment)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def categorize_time_control(time_control: str) -> str:
    """Phân loại thể thức dựa trên TimeControl string (VD: '180+0').

    Theo chuẩn Lichess:
      UltraBullet: < 29s
      Bullet: < 179s
      Blitz: < 479s
      Rapid: < 1499s
      Classical: >= 1500s
    """
    try:
        parts = time_control.split('+')
        initial = int(parts[0])
        increment = int(parts[1]) if len(parts) > 1 else 0
        estimated_duration = initial + 40 * increment

        if estimated_duration < 29:
            return 'UltraBullet'
        elif estimated_duration < 179:
            return 'Bullet'
        elif estimated_duration < 479:
            return 'Blitz'
        elif estimated_duration < 1499:
            return 'Rapid'
        else:
            return 'Classical'
    except (ValueError, IndexError):
        return 'Unknown'


def parse_game_with_clocks(pgn_text: str) -> dict | None:
    """Parse một ván cờ từ PGN text, giữ lại clock time.

    Returns:
        dict chứa tất cả thông tin hoặc None nếu ván bị lọc.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None

        headers = game.headers

        # Filter: kết quả
        result = headers.get("Result", "")
        if result == "*":
            return None

        # Filter: ELO
        white_elo = _safe_int(headers.get("WhiteElo", ""))
        black_elo = _safe_int(headers.get("BlackElo", ""))
        if white_elo is None or black_elo is None:
            return None

        # Duyệt qua mainline() để lấy cả moves VÀ clock comments
        moves_san_parts = []
        clock_seq = []  # Thời gian còn lại (giây) tại mỗi nước
        board = game.board()

        move_count = 0
        for node in game.mainline():
            move = node.move
            san = board.san(move)
            moves_san_parts.append(san)
            board.push(move)
            move_count += 1

            # Parse clock từ comment
            clock_seconds = parse_clock_from_comment(node.comment)
            clock_seq.append(clock_seconds)

        num_moves = move_count
        if num_moves < MIN_MOVES:
            return None

        # Tính time_spent từ clock sequence
        # time_spent = clock_trước - clock_sau (cho cùng 1 bên)
        # Nước lẻ (1, 3, 5...) = Trắng, nước chẵn (2, 4, 6...) = Đen
        time_spent_seq = []
        white_clocks = [clock_seq[i] for i in range(0, len(clock_seq), 2)]
        black_clocks = [clock_seq[i] for i in range(1, len(clock_seq), 2)]

        for i, clk in enumerate(clock_seq):
            if clk is None:
                time_spent_seq.append(None)
                continue

            # Xác định nước trước đó CỦA CÙNG BÊN
            side_idx = i // 2  # Vị trí trong dãy clock của bên đó
            if i % 2 == 0:  # Trắng
                prev_clk = white_clocks[side_idx - 1] if side_idx > 0 else None
            else:  # Đen
                prev_clk = black_clocks[side_idx - 1] if side_idx > 0 else None

            if prev_clk is not None and clk is not None:
                spent = prev_clk - clk
                # Increment có thể làm clock tăng → spent âm
                time_spent_seq.append(max(0.0, spent))
            else:
                time_spent_seq.append(None)

        # Build SAN string  
        # Tạo SAN chuẩn có đánh số (1. e4 e5 2. Nf3 ...)
        san_parts_numbered = []
        for i, san in enumerate(moves_san_parts):
            if i % 2 == 0:
                san_parts_numbered.append(f"{i // 2 + 1}. {san}")
            else:
                san_parts_numbered.append(san)
        moves_san = " ".join(san_parts_numbered)

        # Metadata
        elo_avg = (white_elo + black_elo) // 2
        time_control = headers.get("TimeControl", "")
        game_format = categorize_time_control(time_control)

        # Tính BaseTime và Increment
        try:
            tc_parts = time_control.split('+')
            base_time = int(tc_parts[0])
            increment = int(tc_parts[1]) if len(tc_parts) > 1 else 0
        except (ValueError, IndexError):
            base_time = 0
            increment = 0

        site_url = headers.get("Site", "")
        game_id = site_url.rsplit("/", 1)[-1] if "/" in site_url else ""

        has_clock = any(c is not None for c in clock_seq)

        return {
            "GameID": game_id,
            "WhiteElo": white_elo,
            "BlackElo": black_elo,
            "EloAvg": elo_avg,
            "NumMoves": num_moves,
            "ECO": headers.get("ECO", ""),
            "TimeControl": time_control,
            "BaseTime": base_time,
            "Increment": increment,
            "GameFormat": game_format,
            "Result": result,
            "Termination": headers.get("Termination", ""),
            "Moves": moves_san,
            "ClockSeq": json.dumps(clock_seq),
            "TimeSpentSeq": json.dumps(time_spent_seq),
            "HasClock": has_clock,
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# MAIN: Stream PGN.zst → Reservoir Sampling → Parquet
# ══════════════════════════════════════════════════════════

def stream_pgn_zst(pgn_path: Path):
    """Generator: yield từng PGN text block từ file .pgn.zst."""
    with open(pgn_path, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f_in, read_size=1 << 24) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            leftover = ''

            while True:
                block = text_stream.read(ZST_READ_BLOCK)
                if not block:
                    break

                leftover += block

                if '\n\n[Event ' in leftover:
                    parts = _GAME_SEP_RE.split(leftover)
                    leftover = parts[-1]

                    for pgn_text in parts[:-1]:
                        pgn_text = pgn_text.strip()
                        if pgn_text:
                            yield pgn_text

            # Flush phần còn lại
            if leftover.strip():
                yield leftover.strip()


def main() -> None:
    t0 = time.time()
    print(f"{'═' * 64}")
    print("  TRÍCH XUẤT 30K VÁN CỜ TỪ PGN.ZST — CÓ CLOCK TIME")
    print(f"{'─' * 64}")
    print(f"  Input    : {PGN_ZST_FILE}")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  Sampling : {N_PER_CLASS} ván/band × {NUM_CLASSES} bands = {TOTAL_SAMPLE}")
    print(f"  Filter   : min_moves={MIN_MOVES}, KHÔNG filter thể thức/time forfeit")
    print(f"{'═' * 64}\n")

    if not PGN_ZST_FILE.exists():
        print(f"  ❌ KHÔNG tìm thấy file: {PGN_ZST_FILE}")
        return

    random.seed(42)

    # Reservoir sampling
    reservoirs: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}

    total_read = 0
    total_valid = 0
    total_with_clock = 0

    for pgn_text in stream_pgn_zst(PGN_ZST_FILE):
        total_read += 1

        game_dict = parse_game_with_clocks(pgn_text)
        if game_dict is None:
            continue

        total_valid += 1
        if game_dict["HasClock"]:
            total_with_clock += 1

        elo_avg = game_dict["EloAvg"]
        band = elo_to_band(elo_avg)
        counts[band] += 1
        n = counts[band]

        if len(reservoirs[band]) < N_PER_CLASS:
            reservoirs[band].append(game_dict)
        else:
            j = random.randint(0, n - 1)
            if j < N_PER_CLASS:
                reservoirs[band][j] = game_dict

        # Logging
        if total_read % LOG_INTERVAL == 0:
            filled = sum(min(len(r), N_PER_CLASS) for r in reservoirs.values())
            elapsed = time.time() - t0
            speed = total_read / elapsed if elapsed > 0 else 0
            clock_pct = (total_with_clock / total_valid * 100) if total_valid > 0 else 0
            print(
                f"  Đã đọc {total_read:>10,} ván | "
                f"Hợp lệ: {total_valid:,} | "
                f"Clock: {clock_pct:.0f}% | "
                f"Gom: {filled}/{TOTAL_SAMPLE} | "
                f"[{', '.join(f'B{i}:{len(reservoirs[i])}' for i in range(NUM_CLASSES))}] | "
                f"{speed:,.0f} ván/s",
                flush=True,
            )

        # Dừng sớm khi đủ quota + oversample
        all_full = all(len(reservoirs[i]) >= N_PER_CLASS for i in range(NUM_CLASSES))
        if all_full and all(counts[i] >= N_PER_CLASS * OVERSAMPLE_FACTOR for i in range(NUM_CLASSES)):
            print(f"\n  ✓ Đủ quota cho tất cả bands sau {total_read:,} ván. Dừng sớm!")
            break

    # Gộp kết quả
    band_names = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    all_rows: list[dict] = []

    print(f"\n{'─' * 64}")
    print("  Kết quả sampling:")
    for band_id in range(NUM_CLASSES):
        n = len(reservoirs[band_id])
        clk_count = sum(1 for r in reservoirs[band_id] if r["HasClock"])
        print(f"    Band {band_id} ({band_names[band_id]:>14}): {n:,} ván | Clock: {clk_count:,}")
        all_rows.extend(reservoirs[band_id])

    # Tạo DataFrame
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

    # Thống kê
    print(f"\n{'─' * 64}")
    print(f"  Tổng mẫu        : {final_df.height:,}")
    print(f"  Có clock time    : {final_df.filter(pl.col('HasClock')).height:,}")
    print(f"  Phân bổ thể thức:")
    for fmt, cnt in final_df.group_by("GameFormat").len().sort("len", descending=True).iter_rows():
        print(f"    {fmt:>14}: {cnt:,}")
    print(f"  Phân bổ ELO band:")
    for band, cnt in final_df.group_by("ModelBand").len().sort("ModelBand").iter_rows():
        print(f"    Band {band} ({band_names[band]:>14}): {cnt:,}")

    # Lưu
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(str(OUTPUT_FILE), compression="zstd")

    elapsed = time.time() - t0
    print(f"\n  ✅ Đã lưu: {OUTPUT_FILE}")
    print(f"  Tổng ván đã đọc : {total_read:,}")
    print(f"  Thời gian        : {elapsed / 60:.1f} phút")
    print(f"{'═' * 64}")


if __name__ == "__main__":
    main()
