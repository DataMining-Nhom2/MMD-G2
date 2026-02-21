"""
convert_pgn_fast.py  —  ML Edition
====================================
Tối ưu cho: i5-14600KF (20 luồng), 31 GB RAM, file ~30 GB .pgn.zst
Mục đích output: training data cho ML model

Kiến trúc pipeline (3 tầng):
  [Reader]  →  raw_queue  →  [Worker × N]  →  json_queue  →  [Writer]
  đọc .zst    (bounded)      parse + filter   (bounded)       ghi JSONL
  tách PGN                   + orjson                         + báo lỗi

Cải tiến so với bản v1:
  ✓ orjson thay json         → serialize nhanh hơn 3–5×
  ✓ Filter data bẩn          → loại ván không đủ chất lượng cho training
  ✓ Log lỗi kèm PGN gốc     → audit data quality dễ dàng
  ✓ Field NumMoves           → filter downstream không cần re-parse
  ✓ Field EloAvg             → dễ filter theo skill level khi training
  ✓ Reader đọc theo block    → bỏ vòng for line, tăng throughput I/O
  ✓ Error writer process     → không block pipeline chính
"""

import zstandard as zstd
import chess.pgn
import io
import os
import time
import multiprocessing as mp
from multiprocessing import Queue, Process
import re
import traceback
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PGN_ZST, PGN_JSONL, PGN_ERROR_LOG, PGN_SKIP_LOG

try:
    import orjson  # pip install orjson

    def _dumps(obj: dict) -> str:
        return orjson.dumps(obj).decode("utf-8")

    USING_ORJSON = True
except ImportError:
    import json

    def _dumps(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False)

    USING_ORJSON = False


# ╔══════════════════════════════════════════════════════════╗
# ║                        CẤU HÌNH                         ║
# ╚══════════════════════════════════════════════════════════╝

# Path lấy từ src/config.py — không hardcode ở đây nữa
INPUT_FILE = PGN_ZST
OUTPUT_FILE = PGN_JSONL
ERROR_LOG = PGN_ERROR_LOG  # ván lỗi parse (PGN gốc kèm theo)
SKIPPED_LOG = PGN_SKIP_LOG  # ván bị filter (lý do + PGN gốc)

# ── CPU ──────────────────────────────────────────────────
# i5-14600KF: 20 logical cores
# Để lại 2 cho reader (main) + writer + OS
NUM_WORKERS = max(1, mp.cpu_count() - 4)  # = 16

# ── Queue / batch sizes ──────────────────────────────────
CHUNK_SIZE = 400  # ván PGN thô / chunk gửi cho worker
RAW_QUEUE_SIZE = 250  # ~250 × 400 ván × ~1.5 KB ≈ ~150 MB RAM
JSON_QUEUE_SIZE = 500  # list JSON strings chờ writer
WRITE_BATCH = 8_000  # dòng JSON ghi mỗi lần f_out.write()
ERR_QUEUE_SIZE = 2_000  # lỗi chờ error-writer

# ── I/O ──────────────────────────────────────────────────
ZST_READ_BLOCK = 1 << 19  # 512 KB text block mỗi lần đọc
FILE_BUF_SIZE = 1 << 21  # 2 MB write buffer cho JSONL

# ── Tiến độ ──────────────────────────────────────────────
LOG_INTERVAL = 100_000  # in mỗi N ván

# ╔══════════════════════════════════════════════════════════╗
# ║              FILTER DATA QUALITY (ML)                   ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Bật/tắt từng filter theo nhu cầu dataset của bạn       ║
# ╚══════════════════════════════════════════════════════════╝

FILTER = {
    # Bỏ ván không có Elo (bot, anonymous)
    "require_elo": True,
    # Elo tối thiểu cả 2 bên (0 = tắt)
    "min_elo": 0,
    # Elo tối đa cả 2 bên (0 = tắt)
    "max_elo": 0,
    # Bỏ ván dưới N nước (too short: forfeits / abandoned)
    "min_moves": 5,
    # Bỏ ván trên N nước (0 = tắt)
    "max_moves": 0,
    # Bỏ ván không có nước đi (headers-only)
    "require_moves": True,
    # Bỏ ván kết quả * (chưa kết thúc)
    "require_result": True,
}


# ══════════════════════════════════════════════════════════
# TẦNG 1: READER
# ══════════════════════════════════════════════════════════

_GAME_SEP_RE = re.compile(r"\n\n(?=\[Event )")


def reader_process(input_path: str, raw_queue: Queue) -> None:
    """
    Chạy trên process chính.
    Đọc file theo block, tách PGN thô bằng regex, gom chunk, đẩy queue.
    Không parse PGN → CPU thấp, I/O throughput cao.
    """
    try:
        with open(input_path, "rb") as f_in:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(
                f_in, read_size=1 << 24
            ) as reader:  # 16 MB zst buffer
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                leftover = ""
                chunk_buf = []

                while True:
                    block = text_stream.read(ZST_READ_BLOCK)
                    if not block:
                        break

                    leftover += block

                    if "\n\n[Event " in leftover:
                        parts = _GAME_SEP_RE.split(leftover)
                        leftover = parts[-1]  # phần cuối chưa hoàn chỉnh

                        for pgn_text in parts[:-1]:
                            pgn_text = pgn_text.strip()
                            if pgn_text:
                                chunk_buf.append(pgn_text)

                        while len(chunk_buf) >= CHUNK_SIZE:
                            raw_queue.put(chunk_buf[:CHUNK_SIZE])
                            chunk_buf = chunk_buf[CHUNK_SIZE:]

                # Flush phần còn lại
                if leftover.strip():
                    chunk_buf.append(leftover.strip())
                if chunk_buf:
                    raw_queue.put(chunk_buf)

    except Exception as e:
        print(f"\n[READER] Lỗi nghiêm trọng: {e}")
        traceback.print_exc()
    finally:
        for _ in range(NUM_WORKERS):
            raw_queue.put(None)


# ══════════════════════════════════════════════════════════
# TẦNG 2: WORKERS
# ══════════════════════════════════════════════════════════


def _safe_int(value: str) -> int | None:
    """Elo string → int, None nếu không hợp lệ."""
    try:
        v = int(value)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def parse_and_filter(pgn_text: str) -> tuple[dict | None, str | None]:
    """
    Parse một PGN thô, áp dụng filter data quality.

    Returns:
        (game_dict, None)     — ván hợp lệ, đưa vào output
        (None, skip_reason)   — bị filter (data bẩn), đưa vào skipped log
        (None, None)          — lỗi parse thực sự, đưa vào error log
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None, None

        headers = game.headers

        # Filter: kết quả
        result = headers.get("Result", "")
        if FILTER["require_result"] and result == "*":
            return None, "result=* (unfinished)"

        # Filter: nước đi — đếm một lần duy nhất, tái dùng cho num_moves
        moves_list = list(game.mainline_moves())
        num_moves = len(moves_list)

        if FILTER["require_moves"] and num_moves == 0:
            return None, "no_moves"
        if FILTER["min_moves"] > 0 and num_moves < FILTER["min_moves"]:
            return None, f"too_short({num_moves}<{FILTER['min_moves']})"
        if FILTER["max_moves"] > 0 and num_moves > FILTER["max_moves"]:
            return None, f"too_long({num_moves}>{FILTER['max_moves']})"

        # Filter: Elo
        white_elo = _safe_int(headers.get("WhiteElo", ""))
        black_elo = _safe_int(headers.get("BlackElo", ""))

        if FILTER["require_elo"] and (white_elo is None or black_elo is None):
            return (
                None,
                f"missing_elo(W={headers.get('WhiteElo')} B={headers.get('BlackElo')})",
            )
        if FILTER["min_elo"] > 0 and (
            (white_elo or 0) < FILTER["min_elo"] or (black_elo or 0) < FILTER["min_elo"]
        ):
            return None, f"elo_too_low(W={white_elo} B={black_elo})"
        if FILTER["max_elo"] > 0 and (
            (white_elo or 9999) > FILTER["max_elo"]
            or (black_elo or 9999) > FILTER["max_elo"]
        ):
            return None, f"elo_too_high(W={white_elo} B={black_elo})"

        # Build SAN string (moves_list đã có sẵn → không gọi mainline_moves() 2 lần)
        moves_san = game.board().variation_san(moves_list)

        site_url = headers.get("Site", "")
        game_id = site_url.rsplit("/", 1)[-1] if "/" in site_url else site_url
        elo_avg = (white_elo + black_elo) // 2 if (white_elo and black_elo) else None

        return {
            "GameID": game_id,
            "Event": headers.get("Event", ""),
            "Date": headers.get("UTCDate", headers.get("Date", "")),
            "Time": headers.get("UTCTime", ""),
            "White": headers.get("White", ""),
            "Black": headers.get("Black", ""),
            "Result": result,
            "WhiteElo": white_elo,  # int | None
            "BlackElo": black_elo,  # int | None
            "EloAvg": elo_avg,  # NEW: bucket filter khi training
            "NumMoves": num_moves,  # NEW: downstream filter
            "WhiteRatingDiff": headers.get("WhiteRatingDiff", ""),
            "BlackRatingDiff": headers.get("BlackRatingDiff", ""),
            "ECO": headers.get("ECO", ""),
            "Opening": headers.get("Opening", ""),
            "TimeControl": headers.get("TimeControl", ""),
            "Termination": headers.get("Termination", ""),
            "Moves": moves_san,
        }, None

    except Exception:
        return None, None  # lỗi parse — caller xử lý


def worker_process(
    raw_queue: Queue,
    json_queue: Queue,
    err_queue: Queue,
    worker_id: int,
) -> None:
    while True:
        chunk = raw_queue.get()
        if chunk is None:
            json_queue.put(None)
            break

        ok_results = []
        skip_items = []  # [(reason, pgn_text)]
        err_pgns = []  # [pgn_text] — lỗi parse thực sự

        for pgn_text in chunk:
            try:
                game_dict, skip_reason = parse_and_filter(pgn_text)
            except Exception:
                err_pgns.append(pgn_text)
                continue

            if game_dict is not None:
                ok_results.append(_dumps(game_dict))
            elif skip_reason is not None:
                skip_items.append((skip_reason, pgn_text))
            else:
                err_pgns.append(pgn_text)

        if ok_results:
            json_queue.put(ok_results)

        if err_pgns or skip_items:
            try:
                err_queue.put_nowait(
                    {
                        "worker": worker_id,
                        "errors": err_pgns,
                        "skipped": skip_items,
                    }
                )
            except Exception:
                pass  # err_queue đầy → hi sinh log entry, không block


# ══════════════════════════════════════════════════════════
# TẦNG 2b: ERROR WRITER
# ══════════════════════════════════════════════════════════


def error_writer_process(
    err_queue: Queue,
    num_workers: int,
    error_path: str,
    skipped_path: str,
) -> None:
    """
    Process độc lập, không block pipeline.
    - error_path  : PGN gốc của ván lỗi parse để debug
    - skipped_path: PGN gốc + lý do filter để audit data quality
    """
    total_errors = 0
    total_skipped = 0
    done_workers = 0
    skip_reasons: dict[str, int] = {}

    with (
        open(error_path, "w", encoding="utf-8") as f_err,
        open(skipped_path, "w", encoding="utf-8") as f_skip,
    ):
        while True:
            item = err_queue.get()

            if item is None:
                done_workers += 1
                if done_workers >= num_workers:
                    break
                continue

            for pgn_text in item.get("errors", []):
                total_errors += 1
                f_err.write(f"# ERROR #{total_errors}  worker={item['worker']}\n")
                f_err.write(pgn_text)
                f_err.write("\n\n")

            for reason, pgn_text in item.get("skipped", []):
                total_skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                f_skip.write(f"# SKIP  reason={reason}  worker={item['worker']}\n")
                f_skip.write(pgn_text)
                f_skip.write("\n\n")

        # Summary cuối file
        f_err.write(f"\n# TỔNG LỖI PARSE: {total_errors}\n")
        f_skip.write(f"\n# TỔNG BỊ FILTER: {total_skipped}\n")
        if skip_reasons:
            f_skip.write("# Phân tích theo lý do:\n")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                f_skip.write(f"#   {reason}: {count:,}\n")

    print(
        f"\n  [Log] Lỗi parse: {total_errors:,} ván → {error_path}"
        f"\n  [Log] Bị filter: {total_skipped:,} ván → {skipped_path}"
    )


# ══════════════════════════════════════════════════════════
# TẦNG 3: WRITER
# ══════════════════════════════════════════════════════════


def writer_process(
    json_queue: Queue,
    err_queue: Queue,
    output_path: str,
    num_workers: int,
) -> None:
    start_time = time.time()
    game_count = 0
    done_workers = 0
    write_buf = []

    # Khởi tạo tqdm progress bar
    pbar = tqdm(
        desc="Đang xử lý",
        unit=" ván",
        unit_scale=True,
        smoothing=0.1,
        dynamic_ncols=True,
        postfix={"tốc độ": "0 ván/s", "dung lượng": "0 GB"},
    )

    with open(output_path, "w", encoding="utf-8", buffering=FILE_BUF_SIZE) as f_out:
        while done_workers < num_workers:
            item = json_queue.get()

            if item is None:
                done_workers += 1
                continue

            write_buf.extend(item)
            batch_size = len(item)
            game_count += batch_size

            # Cập nhật progress bar
            pbar.update(batch_size)

            # Cập nhật thông tin bổ sung
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = game_count / elapsed
                gb_est = game_count * 350 / 1e9
                pbar.set_postfix(
                    {"tốc độ": f"{speed:,.0f} ván/s", "dung lượng": f"{gb_est:.1f} GB"},
                    refresh=False,
                )

            if len(write_buf) >= WRITE_BATCH:
                f_out.write("\n".join(write_buf))
                f_out.write("\n")
                write_buf.clear()

        if write_buf:
            f_out.write("\n".join(write_buf))
            f_out.write("\n")

    pbar.close()
    total_time = time.time() - start_time

    # Báo error_writer dừng
    for _ in range(num_workers):
        err_queue.put(None)

    print(f"\n\n{'═'*64}")
    print(f"  HOÀN THÀNH!")
    print(f"  Ván ghi ra JSONL : {game_count:,}")
    print(f"  Tổng thời gian   : {total_time/3600:.2f} giờ  ({total_time:.0f}s)")
    print(f"  Tốc độ TB        : {game_count/total_time:,.0f} ván/s")
    print(f"  Output           : {output_path}")
    print(f"{'═'*64}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════


def convert_zst_pgn_to_jsonl(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file '{input_path}'")
        return

    file_size_gb = os.path.getsize(input_path) / 1e9
    active_filters = {k: v for k, v in FILTER.items() if v}

    print(f"\n{'═'*64}")
    print(f"  PGN → JSONL Converter  —  ML Edition")
    print(f"{'─'*64}")
    print(f"  Input    : {input_path}  ({file_size_gb:.1f} GB compressed)")
    print(f"  Output   : {output_path}")
    print(f"  Workers  : {NUM_WORKERS} / {mp.cpu_count()} logical cores")
    print(
        f"  orjson   : {'✓' if USING_ORJSON else '✗  (pip install orjson để tăng tốc ~4×)'}"
    )
    print(f"  Filters  : {active_filters}")
    print(f"{'═'*64}\n")

    raw_queue = Queue(maxsize=RAW_QUEUE_SIZE)
    json_queue = Queue(maxsize=JSON_QUEUE_SIZE)
    err_queue = Queue(maxsize=ERR_QUEUE_SIZE)

    err_writer = Process(
        target=error_writer_process,
        args=(err_queue, NUM_WORKERS, ERROR_LOG, SKIPPED_LOG),
        daemon=False,
        name="ErrorWriter",
    )
    err_writer.start()

    writer = Process(
        target=writer_process,
        args=(json_queue, err_queue, output_path, NUM_WORKERS),
        daemon=False,
        name="Writer",
    )
    writer.start()

    workers = []
    for i in range(NUM_WORKERS):
        w = Process(
            target=worker_process,
            args=(raw_queue, json_queue, err_queue, i),
            daemon=True,
            name=f"Worker-{i}",
        )
        w.start()
        workers.append(w)

    # Reader chạy trên process chính
    reader_process(input_path, raw_queue)

    for w in workers:
        w.join()

    writer.join()
    err_writer.join()


if __name__ == "__main__":
    mp.set_start_method("spawn" if os.name == "nt" else "fork")
    # Path objects từ config → str cho các hàm open() trong subprocess
    convert_zst_pgn_to_jsonl(str(INPUT_FILE), str(OUTPUT_FILE))
