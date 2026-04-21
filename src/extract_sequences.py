"""Trích xuất chuỗi CPL từng nước đi và thời gian từng nước đi.

Phase V4 — Sequence Data Extraction (theo hướng paper arXiv:2409.11506):
- CPL từng ply (không phải aggregate): list[float]
- Time spent từng ply từ %clk annotations: list[float]

Thay đổi so với V2 (11 aggregate features):
- V2: avg_cpl, blunder_rate, opening_cpl, avg_wdl_loss, ... (tổng hợp → mất tính chuỗi)
- V4: cpl_seq = [12.5, 45.0, 0.0, 23.1, ...] (raw sequence per ply)

Yêu cầu dữ liệu:
- CPL sequence: chạy được ngay trên sample_30k.parquet (Moves = plain SAN)
- Time sequence: cần Moves chứa %clk annotations (cần re-preprocess từ PGN gốc)
  → Xem src/clock_parser.py để hiểu cách extract %clk khi re-preprocess

Output:
    data/features/sample_30k_sequences.parquet
    Columns:
        EloAvg          int32
        ModelBand       int8
        NumMoves        int16
        cpl_seq         list[float32]  — CPL từng ply (có NaN khi là mate position)
        time_spent_seq  list[float32]  — giây spent/ply (empty nếu không có %clk)
        seq_length      int32          — số ply được phân tích (loại bỏ lỗi)
        has_clock       bool           — True nếu %clk có trong dữ liệu
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
import numpy as np
import polars as pl
from tqdm import tqdm

from src.feature_config import (
    MODEL_BINS,
    MODEL_BAND_LABEL_TO_ID,
    SAMPLE_SOURCE_FILE,
)

# ── Cấu hình ──────────────────────────────────────────────────────────────
STOCKFISH_PATH = str(
    Path(__file__).resolve().parent.parent / ".tmp" / "stockfish_binary"
)
STOCKFISH_DEPTH = 10
STOCKFISH_THREADS = 1
STOCKFISH_HASH_MB = 128

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "features"
    / "sample_30k_sequences.parquet"
)

# Regex để tìm %clk annotations: { [%clk H:MM:SS] }
_CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")


# ╔══════════════════════════════════════════════════════════╗
# ║              CLOCK TIME EXTRACTION                       ║
# ╚══════════════════════════════════════════════════════════╝


def parse_clock_remaining(moves_with_clk: str) -> list[float]:
    """Parse danh sách giây còn lại từ %clk annotations trong chuỗi PGN.

    Ví dụ input:
        "1. e4 { [%clk 0:15:00] } e5 { [%clk 0:15:00] } 2. Nf3 { [%clk 0:14:58] }"
    Output:
        [900.0, 900.0, 898.0]  # giây còn lại sau mỗi ply

    Lưu ý:
        - Dữ liệu hiện tại (sample_30k.parquet) không có %clk → trả về []
        - Cần re-preprocess từ PGN gốc để có %clk (xem src/clock_parser.py)
    """
    times: list[float] = []
    for match in _CLK_PATTERN.finditer(moves_with_clk):
        h = int(match.group(1))
        m = int(match.group(2))
        s = int(match.group(3))
        times.append(float(h * 3600 + m * 60 + s))
    return times


def compute_time_spent(
    clock_remaining: list[float],
    increment: float = 0.0,
) -> list[float]:
    """Tính giây SPENT mỗi ply từ danh sách giây còn lại.

    Quy tắc: Trắng đi ply 0, 2, 4, ... — Đen đi ply 1, 3, 5, ...
    time_spent[i] = clock_remaining[i-2] - clock_remaining[i] + increment

    Args:
        clock_remaining: giây còn lại sau mỗi ply
        increment: giây increment mỗi nước (từ cột Increment)

    Returns:
        Danh sách giây spent / ply (0.0 cho 2 ply đầu)
    """
    n = len(clock_remaining)
    spent = [0.0] * n
    for i in range(2, n):
        spent[i] = max(0.0, clock_remaining[i - 2] - clock_remaining[i] + increment)
    return spent


# ╔══════════════════════════════════════════════════════════╗
# ║              STOCKFISH CPL SEQUENCE                      ║
# ╚══════════════════════════════════════════════════════════╝


def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> float | None:
    """Chuyển đổi PovScore → centipawns từ góc nhìn bên vừa đi.

    Trả về None nếu là mate score.
    """
    cp = score.pov(turn).score()
    return float(cp) if cp is not None else None


def analyze_game_cpl_sequence(
    moves_san: str,
    engine: chess.engine.SimpleEngine,
    depth: int = STOCKFISH_DEPTH,
) -> list[float]:
    """Phân tích 1 ván cờ → trả về CPL theo từng ply (không aggregate).

    Quy trình:
        - Duyệt từng nước đi theo thứ tự
        - Gọi Stockfish analyse() TRƯỚC và SAU mỗi nước
        - CPL = max(0, score_before - score_after) từ góc nhìn bên đi
        - NaN cho nước mate hoặc lỗi parse

    Returns:
        cpl_seq: CPL từng ply (có thể chứa NaN)
    """
    board = chess.Board()
    cpl_seq: list[float] = []

    # Parse SAN tokens (lọc số thứ tự và kết quả)
    raw_tokens = (moves_san or "").split()
    san_moves: list[str] = [
        t
        for t in raw_tokens
        if not t.endswith(".") and t not in ("1-0", "0-1", "1/2-1/2", "*")
    ]

    limit = chess.engine.Limit(depth=depth)

    for san in san_moves:
        try:
            move = board.parse_san(san)
        except Exception:
            cpl_seq.append(float("nan"))
            continue

        turn = board.turn  # Bên sắp đi

        # ── Phân tích TRƯỚC nước đi ──────────────────────────────────
        try:
            info_before = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
            cp_before = _score_to_cp(info_before["score"], turn)
        except Exception:
            cp_before = None

        # ── Thực hiện nước đi ────────────────────────────────────────
        board.push(move)

        # ── Phân tích SAU nước đi ────────────────────────────────────
        try:
            info_after = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
            cp_after = _score_to_cp(info_after["score"], turn)
        except Exception:
            cp_after = None

        # ── Tính CPL từng nước ──────────────────────────────────────
        if cp_before is not None and cp_after is not None:
            cpl_seq.append(max(0.0, float(cp_before) - float(cp_after)))
        else:
            cpl_seq.append(float("nan"))

    return cpl_seq


# ╔══════════════════════════════════════════════════════════╗
# ║              WORKER (MULTIPROCESSING)                    ║
# ╚══════════════════════════════════════════════════════════╝


def _worker_chunk(
    args: tuple[list[str], list[float], str, int, int, int],
) -> list[dict]:
    """Worker subprocess: xử lý một chunk ván cờ.

    Args:
        args: (chunk_san, chunk_increment, engine_path, depth, threads, hash_mb)

    Returns:
        List dict với keys: cpl_seq, time_spent_seq, seq_length, has_clock
    """
    chunk_san, chunk_increment, engine_path, depth, threads, hash_mb = args

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    results = []
    try:
        for moves_san, increment in zip(chunk_san, chunk_increment):
            try:
                # CPL sequence từ Stockfish
                cpl_seq = analyze_game_cpl_sequence(moves_san, engine, depth)

                # Time sequence từ %clk (nếu có)
                clock_remaining = parse_clock_remaining(moves_san)
                has_clock = len(clock_remaining) > 0
                time_spent_seq = (
                    compute_time_spent(clock_remaining, increment) if has_clock else []
                )

                # Số ply hợp lệ (không NaN)
                valid_count = sum(1 for c in cpl_seq if not (c != c))  # IEEE NaN check

                results.append(
                    {
                        "cpl_seq": cpl_seq,
                        "time_spent_seq": time_spent_seq,
                        "seq_length": valid_count,
                        "has_clock": has_clock,
                    }
                )
            except Exception:
                results.append(
                    {
                        "cpl_seq": [],
                        "time_spent_seq": [],
                        "seq_length": 0,
                        "has_clock": False,
                    }
                )
    finally:
        engine.quit()

    return results


# ╔══════════════════════════════════════════════════════════╗
# ║              SEQUENCE EXTRACTOR                          ║
# ╚══════════════════════════════════════════════════════════╝


class SequenceExtractor:
    """Trích xuất CPL sequence và time sequence từ Parquet data.

    Thay thế StockfishTransformer V2 (11 aggregate features) bằng
    raw sequence output theo hướng paper arXiv:2409.11506.
    """

    def __init__(
        self,
        engine_path: str = STOCKFISH_PATH,
        depth: int = STOCKFISH_DEPTH,
        threads: int = STOCKFISH_THREADS,
        hash_mb: int = STOCKFISH_HASH_MB,
        chunk_size: int = 100,
    ) -> None:
        self.engine_path = engine_path
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.chunk_size = chunk_size

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Chạy extraction trên toàn bộ DataFrame.

        Args:
            df: DataFrame chứa ít nhất các cột Moves, EloAvg, NumMoves
                Tùy chọn: Increment (dùng để tính time_spent_seq)

        Returns:
            DataFrame với các cột sequence mới
        """
        moves_list = df["Moves"].to_list()
        increment_list = (
            df["Increment"].cast(pl.Float32).to_list()
            if "Increment" in df.columns
            else [0.0] * len(moves_list)
        )

        # Chia chunk
        chunks_san = [
            moves_list[i : i + self.chunk_size]
            for i in range(0, len(moves_list), self.chunk_size)
        ]
        chunks_inc = [
            increment_list[i : i + self.chunk_size]
            for i in range(0, len(increment_list), self.chunk_size)
        ]

        args_list = [
            (
                chunk_san,
                chunk_inc,
                self.engine_path,
                self.depth,
                self.threads,
                self.hash_mb,
            )
            for chunk_san, chunk_inc in zip(chunks_san, chunks_inc)
        ]

        max_workers = max(1, (os.cpu_count() or 4) - 2)
        print(
            f"\n  Khởi tạo {max_workers} tiến trình Stockfish (depth={self.depth})..."
        )

        all_results: list[dict] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunk_res in tqdm(
                executor.map(_worker_chunk, args_list),
                total=len(args_list),
                desc="CPL Sequence Extraction",
                unit="chunk",
            ):
                all_results.extend(chunk_res)

        # Build output DataFrame
        cpl_seqs = [r["cpl_seq"] for r in all_results]
        time_seqs = [r["time_spent_seq"] for r in all_results]
        seq_lengths = [r["seq_length"] for r in all_results]
        has_clocks = [r["has_clock"] for r in all_results]

        # Tạo các cột metadata gốc cần giữ lại
        keep_cols = ["EloAvg", "ModelBand", "NumMoves"]
        meta_df = df.select([c for c in keep_cols if c in df.columns])

        # Tạo sequence DataFrame
        seq_df = pl.DataFrame(
            {
                "cpl_seq": pl.Series(cpl_seqs, dtype=pl.List(pl.Float32)),
                "time_spent_seq": pl.Series(time_seqs, dtype=pl.List(pl.Float32)),
                "seq_length": pl.Series(seq_lengths, dtype=pl.Int32),
                "has_clock": pl.Series(has_clocks, dtype=pl.Boolean),
            }
        )

        return pl.concat([meta_df, seq_df], how="horizontal")


# ╔══════════════════════════════════════════════════════════╗
# ║              MAIN                                        ║
# ╚══════════════════════════════════════════════════════════╝


def main() -> None:
    t0 = time.time()
    print("═" * 60)
    print("  TRÍCH XUẤT CPL SEQUENCE VÀ TIME SEQUENCE")
    print("  Phase V4 — Theo hướng paper arXiv:2409.11506")
    print("─" * 60)

    # Bước 1: Load sample
    print(f"\n  Load: {SAMPLE_SOURCE_FILE}")
    df = pl.read_parquet(str(SAMPLE_SOURCE_FILE))
    print(f"  Rows: {df.height}, Cols: {df.width}")

    # Kiểm tra ModelBand
    if "ModelBand" not in df.columns and "EloAvg" in df.columns:
        bins = MODEL_BINS
        df = df.with_columns(
            pl.when(pl.col("EloAvg") < bins[1])
            .then(pl.lit(0))
            .when(pl.col("EloAvg") < bins[2])
            .then(pl.lit(1))
            .when(pl.col("EloAvg") < bins[3])
            .then(pl.lit(2))
            .when(pl.col("EloAvg") < bins[4])
            .then(pl.lit(3))
            .otherwise(pl.lit(4))
            .cast(pl.Int8)
            .alias("ModelBand")
        )

    # Kiểm tra %clk trong dữ liệu
    sample_moves = df["Moves"][0]
    has_clock_data = "%clk" in (sample_moves or "")
    print(f"\n  Kiểm tra %clk trong dữ liệu: {'CÓ' if has_clock_data else 'KHÔNG'}")
    if not has_clock_data:
        print("  ⚠ Dữ liệu hiện tại không có %clk annotations.")
        print("  ⚠ time_spent_seq sẽ là empty list [].")
        print("  ⚠ Để có time sequence, cần re-preprocess từ PGN gốc.")
        print("     (Xem src/clock_parser.py và src/preprocessing_with_clocks.py)")

    # Bước 2: Khởi tạo extractor
    extractor = SequenceExtractor(
        engine_path=STOCKFISH_PATH,
        depth=STOCKFISH_DEPTH,
    )

    # Bước 3: Extract sequences
    print(f"\n  Đang extract {df.height} ván (depth={STOCKFISH_DEPTH})...")
    print(f"  Ước tính: ~{df.height * 0.05 / 60:.0f} phút")
    result = extractor.extract(df)

    # Bước 4: Thống kê
    seq_len_stats = result["seq_length"].describe()
    print(f"\n  Thống kê seq_length:")
    print(f"    Mean: {result['seq_length'].mean():.1f}")
    print(f"    Min:  {result['seq_length'].min()}")
    print(f"    Max:  {result['seq_length'].max()}")
    print(f"    P50:  {result['seq_length'].median():.0f}")

    empty_seqs = result.filter(pl.col("seq_length") == 0).height
    print(
        f"\n  Ván không có CPL sequence hợp lệ: {empty_seqs} ({empty_seqs/result.height*100:.1f}%)"
    )

    clock_count = result.filter(pl.col("has_clock")).height
    print(
        f"  Ván có %clk time data: {clock_count} ({clock_count/result.height*100:.1f}%)"
    )

    # Bước 5: Lưu
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(OUTPUT_FILE), compression="zstd")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✅ HOÀN THÀNH!")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Rows: {result.height}, Cols: {result.width}")
    print(f"  Thời gian: {elapsed/60:.1f} phút")
    print(f"\n  Schema:")
    for name, dtype in result.schema.items():
        print(f"    {name}: {dtype}")


if __name__ == "__main__":
    main()
