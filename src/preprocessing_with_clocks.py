"""Preprocessing nâng cấp: Giữ lại clock times (%clk) từ PGN gốc.

Mục đích:
    Tạo lại mẫu dữ liệu 30k ván VỚI per-move clock times.
    Script này cần được chạy từ file PGN.zst gốc (Lichess database dump).

Tại sao cần re-preprocess:
    Script preprocessing.py gốc dùng game.board().variation_san(moves_list)
    để export chuỗi nước đi → STRIP HẾT comments (bao gồm %clk annotations).
    Script này thay bằng cách đi qua game.mainline() để giữ lại clock data.

Điều kiện chạy (yêu cầu):
    - File PGN.zst gốc từ Lichess database: https://database.lichess.org/
    - Lichess PGN exports có format: 1. e4 { [%clk 0:15:00] } e5 { [%clk 0:14:59] } ...
    - Chạy: python -m src.preprocessing_with_clocks

Output thêm so với sample_30k.parquet:
    - Cột 'ClockSeq':     list[float] — giây còn lại sau mỗi ply (từ %clk)
    - Cột 'TimeSpentSeq': list[float] — giây spent mỗi ply (tính từ ClockSeq)

File output:
    data/processed/sample_30k_with_clocks.parquet

Sau đó chạy extract_sequences.py để có CPL sequence + time sequence kết hợp.
"""

from __future__ import annotations

import io
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess.pgn
import polars as pl
import pyarrow.parquet as pq

try:
    import zstandard as zstd  # type: ignore

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from src.config import DATA_PROCESSED
from src.feature_config import MODEL_BINS
from src.clock_parser import (
    extract_clock_from_game,
    compute_time_spent_from_remaining,
)

# ── Cấu hình ──────────────────────────────────────────────────────────────
N_PER_CLASS = 6_000
NUM_CLASSES = 5
TOTAL_SAMPLE = N_PER_CLASS * NUM_CLASSES  # 30,000

# Tên file output — phân biệt với sample_30k.parquet cũ (không có clock)
OUTPUT_FILE = DATA_PROCESSED / "sample_30k_with_clocks.parquet"

# Đường dẫn file PGN.zst — thay đổi theo máy của bạn
# Download từ: https://database.lichess.org/#standard_games
PGN_ZST_FILES = [
    DATA_PROCESSED.parent / "raw" / "lichess_db_standard_rated_2025-12.pgn.zst",
    DATA_PROCESSED.parent / "raw" / "lichess_db_standard_rated_2026-01.pgn.zst",
]

# Bỏ ván Bullet/Blitz và Time forfeit (giống create_30k_sample.py)
EXCLUDED_FORMATS = {"Bullet", "UltraBullet", "Blitz"}
EXCLUDED_TERMINATIONS = {"Time forfeit"}


# ── Helper ────────────────────────────────────────────────────────────────


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


def _safe_int(value: str) -> int | None:
    try:
        v = int(value)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def parse_game_with_clocks(pgn_text: str) -> dict | None:
    """Parse một PGN text → dict có cả ClockSeq và TimeSpentSeq.

    Khác với preprocessing.py gốc:
        - Dùng game.mainline() thay vì variation_san() để giữ lại comments
        - Export cả MovesSAN (không có comments) VÀ ClockSeq riêng biệt

    Returns:
        dict nếu ván hợp lệ, None nếu bị lọc.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None

    if game is None:
        return None

    headers = game.headers

    # ── Lọc kết quả ──────────────────────────────────────────────────────
    result = headers.get("Result", "")
    if result == "*":
        return None

    # ── Lọc nước đi ──────────────────────────────────────────────────────
    # Lấy mainline nodes (chứa clock info)
    nodes = list(game.mainline())
    num_moves = len(nodes)
    if num_moves < 5:
        return None

    # ── Lọc thể thức ─────────────────────────────────────────────────────
    time_control = headers.get("TimeControl", "")
    termination = headers.get("Termination", "")

    # Tính game format từ TimeControl
    base_time = 0
    increment = 0
    if "+" in time_control:
        parts = time_control.split("+")
        try:
            base_time = int(parts[0])
            increment = int(parts[1])
        except ValueError:
            pass

    total_time = base_time + increment * 40  # Ước tính ~ 40 nước
    if total_time < 180:  # < 3 phút = Bullet/Blitz
        return None
    if termination == "Time forfeit":
        return None

    # ── Lấy Elo ──────────────────────────────────────────────────────────
    white_elo = _safe_int(headers.get("WhiteElo", ""))
    black_elo = _safe_int(headers.get("BlackElo", ""))
    if white_elo is None or black_elo is None:
        return None

    elo_avg = (white_elo + black_elo) // 2

    # ── Extract SAN moves (không có comments) ─────────────────────────────
    board = game.board()
    moves_list = [node.move for node in nodes if node.move is not None]
    moves_san = board.variation_san(moves_list)

    # ── Extract clock sequence LẠI từ nodes (có comments) ────────────────
    clock_remaining = extract_clock_from_game(game)
    time_spent = compute_time_spent_from_remaining(clock_remaining, float(increment))

    return {
        "WhiteElo": white_elo,
        "BlackElo": black_elo,
        "EloAvg": elo_avg,
        "NumMoves": num_moves,
        "ECO": headers.get("ECO", ""),
        "Termination": termination,
        "BaseTime": base_time,
        "Increment": increment,
        "Moves": moves_san,  # Plain SAN (dùng cho Stockfish)
        "ClockSeq": clock_remaining,  # Giây còn lại mỗi ply
        "TimeSpentSeq": time_spent,  # Giây spent mỗi ply
    }


def main() -> None:
    """Tạo lại 30k sample WITH clock times từ PGN.zst gốc."""
    if not HAS_ZSTD:
        print("❌ Cần cài đặt zstandard: pip install zstandard")
        sys.exit(1)

    # Kiểm tra file PGN tồn tại
    available_files = [f for f in PGN_ZST_FILES if f.exists()]
    if not available_files:
        print("❌ Không tìm thấy file PGN.zst gốc!")
        print("   Cần download từ: https://database.lichess.org/#standard_games")
        print("   Đặt tại:")
        for f in PGN_ZST_FILES:
            print(f"     {f}")
        print()
        print("⚠ Lưu ý: Dữ liệu hiện tại (sample_30k.parquet) không có %clk.")
        print("  Để có time sequence, chạy script này với file PGN.zst gốc.")
        sys.exit(1)

    t0 = time.time()
    random.seed(42)

    print("═" * 60)
    print("  TẠO MẪU 30K VỚI CLOCK TIMES")
    print("─" * 60)

    # Reservoir sampling theo band
    reservoirs: dict[int, list[dict]] = {i: [] for i in range(NUM_CLASSES)}
    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    total_seen = 0

    for pgn_path in available_files:
        print(f"\n  Đọc: {pgn_path}")

        with open(pgn_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                while True:
                    try:
                        # python-chess đọc từng game
                        game_text = ""
                        line = text_stream.readline()
                        while line and not line.startswith("[Event ") or not game_text:
                            game_text += line
                            line = text_stream.readline()
                            if line.startswith("[Event ") and game_text.strip():
                                break
                        if not game_text.strip():
                            break
                    except Exception:
                        break

                    game_dict = parse_game_with_clocks(game_text)
                    if game_dict is None:
                        continue

                    band = elo_to_band(game_dict["EloAvg"])
                    total_seen += 1
                    counts[band] += 1

                    # Reservoir sampling
                    if len(reservoirs[band]) < N_PER_CLASS:
                        reservoirs[band].append(game_dict)
                    else:
                        j = random.randint(0, counts[band] - 1)
                        if j < N_PER_CLASS:
                            reservoirs[band][j] = game_dict

        # Kiểm tra đủ chưa
        if all(len(r) >= N_PER_CLASS for r in reservoirs.values()):
            print("  ✅ Đủ mẫu cho tất cả các band!")
            break

    # Gộp và lưu
    all_records = []
    for band_records in reservoirs.values():
        all_records.extend(band_records)

    df = pl.DataFrame(all_records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUTPUT_FILE), compression="zstd")

    elapsed = time.time() - t0
    print(f"\n  ✅ HOÀN THÀNH!")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Rows: {len(all_records)}")
    print(f"  Thời gian: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
