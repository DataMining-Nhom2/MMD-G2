"""Xây dựng bộ dữ liệu 600K ván cờ đầy đủ CPL và Clock Time (Multi-processing).

Luồng chạy siêu tốc 3 Tầng:
1. Reader (1 luồng): Đọc PGN.zst, parse Elo nhanh bằng Regex.
   -> Nếu Band ELO của ván này đã đủ quota thì TỪ CHỐI luôn, không gửi cho Worker.
   -> Nếu chưa đủ, gửi raw PGN vào hàng đợi Task.
2. Worker (16 luồng): Parse PGN -> Lấy Clock -> Gọi Stockfish (Depth 8) tính CPL.
   -> Gửi kết quả về hàng đợi Result.
3. Writer (1 luồng): Gom dữ liệu. Nếu rổ nào đủ 120K thì đóng.
   -> Đủ 600K ván, ghi ra sample_600k_dl.parquet.

Ước tính tiêu thụ: ~16 luồng CPU (còn trống 4 luồng cho hệ điều hành)
Thời gian ước tính: 4-6 tiếng.
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import multiprocessing.queues
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chess.engine
import chess.pgn
import polars as pl
import zstandard as zstd

from src.config import DATA_RAW, DATA_PROCESSED

# ══════════════════════════════════════════════════════════
# CẤU HÌNH HỆ THỐNG
# ══════════════════════════════════════════════════════════

NUM_WORKERS = 16          # 16 luồng cho Stockfish
STOCKFISH_DEPTH = 8       # Hạ xuống 8 để max speed mà vẫn chính xác
STOCKFISH_THREADS = 1     # 1 luồng mỗi Stockfish instance

# Stockfish binary path (hardcode)
STOCKFISH_PATH = "/home/sakana/Code/PTIT/MMDs/MMD-G2/src/.tmp/stockfish/stockfish-ubuntu-x86-64-avx2"

PGN_ZST_FILE = DATA_RAW / "lichess_db_standard_rated_2026-01.pgn.zst"
OUTPUT_FILE = DATA_PROCESSED / "sample_600k_dl.parquet"

NUM_CLASSES = 5
N_PER_CLASS = 120_000
TOTAL_SAMPLE = N_PER_CLASS * NUM_CLASSES  # 600,000
MODEL_BINS = (0, 1000, 1400, 1800, 2200)

MIN_MOVES = 5

TASK_CHUNK_SIZE = 20  # Batch 20 ván / lần gửi cho Worker

# Regex
_CLK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]')
_GAME_SEP_RE = re.compile(r'\n\n(?=\[Event )')

# Regex đọc nhanh ELO từ thô (tránh parse nguyên ván nếu rổ đã đầy)
_WHITE_ELO_RE = re.compile(r'\[WhiteElo\s+"(\d+)"\]')
_BLACK_ELO_RE = re.compile(r'\[BlackElo\s+"(\d+)"\]')

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def elo_to_band(elo: int) -> int:
    if elo < MODEL_BINS[1]: return 0
    if elo < MODEL_BINS[2]: return 1
    if elo < MODEL_BINS[3]: return 2
    if elo < MODEL_BINS[4]: return 3
    return 4

def parse_clock_from_comment(comment: str) -> float | None:
    match = _CLK_RE.search(comment)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return None

def categorize_time_control(time_control: str) -> str:
    try:
        parts = time_control.split('+')
        initial = int(parts[0])
        inc = int(parts[1]) if len(parts) > 1 else 0
        est = initial + 40 * inc
        if est < 29: return 'UltraBullet'
        elif est < 179: return 'Bullet'
        elif est < 479: return 'Blitz'
        elif est < 1499: return 'Rapid'
        else: return 'Classical'
    except:
        return 'Unknown'

def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> float | None:
    cp = score.pov(turn).score()
    return float(cp) if cp is not None else None


# ══════════════════════════════════════════════════════════
# TẦNG 2: WORKER
# ══════════════════════════════════════════════════════════

def process_game(pgn_text: str, engine: chess.engine.SimpleEngine) -> dict | None:
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None: return None

        headers = game.headers
        result = headers.get("Result", "")
        if result == "*": return None

        white_elo_str = headers.get("WhiteElo", "")
        black_elo_str = headers.get("BlackElo", "")
        if not white_elo_str.isdigit() or not black_elo_str.isdigit():
            return None
            
        white_elo = int(white_elo_str)
        black_elo = int(black_elo_str)
        elo_avg = (white_elo + black_elo) // 2

        moves_san_parts = []
        clock_seq = []
        board = game.board()
        move_nodes = list(game.mainline())

        if len(move_nodes) < MIN_MOVES:
            return None

        for node in move_nodes:
            moves_san_parts.append(board.san(node.move))
            board.push(node.move)
            clock_seconds = parse_clock_from_comment(node.comment)
            clock_seq.append(clock_seconds)

        has_clock = any(c is not None for c in clock_seq)
        if not has_clock:
            return None  # CHỈ LẤY CÁC VÁN CÓ CLOCK
            
        # Re-play board for Stockfish CPL
        cpl_seq = []
        board = chess.Board()
        limit = chess.engine.Limit(depth=STOCKFISH_DEPTH)

        for node in move_nodes:
            move = node.move
            turn = board.turn
            try:
                info_before = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
                cp_before = _score_to_cp(info_before["score"], turn)
            except: cp_before = None

            board.push(move)

            try:
                info_after = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
                cp_after = _score_to_cp(info_after["score"], turn)
            except: cp_after = None

            if cp_before is not None and cp_after is not None:
                cpl_seq.append(max(0.0, float(cp_before) - float(cp_after)))
            else:
                cpl_seq.append(float("nan"))

        # Time Spent Seq
        time_spent_seq = []
        w_clocks = [clock_seq[i] for i in range(0, len(clock_seq), 2)]
        b_clocks = [clock_seq[i] for i in range(1, len(clock_seq), 2)]

        for i, clk in enumerate(clock_seq):
            if clk is None:
                time_spent_seq.append(None)
                continue
            side_idx = i // 2
            if i % 2 == 0: prev = w_clocks[side_idx - 1] if side_idx > 0 else None
            else: prev = b_clocks[side_idx - 1] if side_idx > 0 else None
            
            if prev is not None: time_spent_seq.append(max(0.0, prev - clk))
            else: time_spent_seq.append(None)

        # SAN kết hợp
        san_numbered = []
        for i, san in enumerate(moves_san_parts):
            if i % 2 == 0: san_numbered.append(f"{i // 2 + 1}. {san}")
            else: san_numbered.append(san)
        
        tc = headers.get("TimeControl", "")
        base_time = increment = 0
        if "+" in tc:
            try:
                parts = tc.split("+")
                base_time = int(parts[0])
                increment = int(parts[1])
            except: pass
            
        site_url = headers.get("Site", "")
        game_id = site_url.rsplit("/", 1)[-1] if "/" in site_url else ""

        return {
            "GameID": game_id,
            "WhiteElo": white_elo,
            "BlackElo": black_elo,
            "EloAvg": elo_avg,
            "ModelBand": elo_to_band(elo_avg),
            "NumMoves": len(move_nodes),
            "ECO": headers.get("ECO", ""),
            "TimeControl": tc,
            "BaseTime": base_time,
            "Increment": increment,
            "GameFormat": categorize_time_control(tc),
            "Result": result,
            "Termination": headers.get("Termination", ""),
            "Moves": " ".join(san_numbered),
            "ClockSeq": json.dumps(clock_seq),
            "TimeSpentSeq": json.dumps(time_spent_seq),
            "cpl_seq": json.dumps(cpl_seq)
        }
    except Exception as e:
        return None

def worker_process(task_q: mp.Queue, res_q: mp.Queue):
    """Tiến trình phân tích sử dụng 1 Stockfish instance."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Threads": STOCKFISH_THREADS})
    except Exception as e:
        print(f"[Worker] Không khởi tạo được Stockfish (path={STOCKFISH_PATH}): {e}")
        return

    try:
        while True:
            try:
                chunk = task_q.get(timeout=10)
            except Exception:
                break
            if chunk is None:
                break
                
            results = []
            for pgn_text in chunk:
                data = process_game(pgn_text, engine)
                if data is not None:
                    results.append(data)
                    
            if results:
                try:
                    res_q.put(results)
                except (BrokenPipeError, EOFError, OSError):
                    break
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        try:
            engine.quit()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# TẦNG 1: READER
# ══════════════════════════════════════════════════════════

def reader_process(task_q: mp.Queue, shared_counts: mp.Array):
    """Tiến trình đọc Streaming file PGN, vứt bỏ ngay nếu rổ đã đầy."""
    try:
        with open(PGN_ZST_FILE, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(f_in, read_size=1 << 24) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                leftover = ''
                chunk_buf = []
                
                while True:
                    # Kiểm tra xem TẤT CẢ các rổ đã đầy chưa
                    all_full = True
                    for i in range(NUM_CLASSES):
                        if shared_counts[i] < N_PER_CLASS:
                            all_full = False
                            break
                    if all_full:
                        break  # Đọc tới đây là đủ nghỉ

                    block = text_stream.read(1 << 18)
                    if not block:
                        break

                    leftover += block
                    if '\n\n[Event ' in leftover:
                        parts = _GAME_SEP_RE.split(leftover)
                        leftover = parts[-1]

                        for pgn_text in parts[:-1]:
                            pgn_text = pgn_text.strip()
                            if not pgn_text: continue
                            
                            w_match = _WHITE_ELO_RE.search(pgn_text)
                            b_match = _BLACK_ELO_RE.search(pgn_text)
                            if w_match and b_match:
                                w_elo = int(w_match.group(1))
                                b_elo = int(b_match.group(1))
                                band = elo_to_band((w_elo + b_elo) // 2)
                                
                                # CHỈ GỬI WORKER NẾU RỔ CHƯA ĐẦY
                                if shared_counts[band] < N_PER_CLASS:
                                    chunk_buf.append(pgn_text)
                                    if len(chunk_buf) >= TASK_CHUNK_SIZE:
                                        task_q.put(chunk_buf)
                                        chunk_buf = []

                if chunk_buf:
                    task_q.put(chunk_buf)

    except Exception as e:
        print(f"[Reader] Error: {e}")
        traceback.print_exc()
    finally:
        for _ in range(NUM_WORKERS):
            task_q.put(None)


# ══════════════════════════════════════════════════════════
# TẦNG 3: WRITER / MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("═" * 64)
    print("  TRÍCH XUẤT 600K VÁN CỜ & CPL TỪ PGN.ZST (SIÊU TỐC)")
    print("─" * 64)
    if not PGN_ZST_FILE.exists():
        print(f"  ❌ Không tìm thấy PGN.zst tại: {PGN_ZST_FILE}")
        return

    # Check Engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.quit()
        print(f"  Engine       : {STOCKFISH_PATH} (Hoạt động tốt)")
    except:
        print(f"  ❌ Không chạy được Stockfish tại: {STOCKFISH_PATH}")
        return

    print(f"  Input        : {PGN_ZST_FILE}")
    print(f"  Output       : {OUTPUT_FILE}")
    print(f"  Workers      : {NUM_WORKERS} luồng")
    print(f"  Depth        : {STOCKFISH_DEPTH} (Nhanh)")
    print(f"  Mục tiêu     : {TOTAL_SAMPLE:,} ván ({N_PER_CLASS:,} ván/rổ)")
    print("═" * 64 + "\n")

    t0 = time.time()
    manager = mp.Manager()
    task_q = manager.Queue(maxsize=1000)
    res_q = manager.Queue(maxsize=1000)
    shared_counts = mp.Array('i', NUM_CLASSES)
    for i in range(NUM_CLASSES): shared_counts[i] = 0

    reader_p = mp.Process(target=reader_process, args=(task_q, shared_counts))
    reader_p.start()

    workers = []
    for _ in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(task_q, res_q))
        p.start()
        workers.append(p)

    final_data = {i: [] for i in range(NUM_CLASSES)}
    active_workers = NUM_WORKERS
    total_saved = 0
    last_val = 0

    while active_workers > 0:
        try:
            results = res_q.get(timeout=5)
            for res in results:
                b = res["ModelBand"]
                if len(final_data[b]) < N_PER_CLASS:
                    final_data[b].append(res)
                    total_saved += 1
                    shared_counts[b] = len(final_data[b])
                    
            if total_saved - last_val >= 1000:
                elapsed = time.time() - t0
                spd = total_saved / elapsed if elapsed > 0 else 0
                stats_str = ", ".join([f"B{i}:{len(final_data[i])}" for i in range(NUM_CLASSES)])
                print(f"  Đã xong: {total_saved:,}/{TOTAL_SAMPLE:,} | [{stats_str}] | Tốc độ: {spd:.1f} ván/s")
                last_val = total_saved
                
            all_full = all(len(final_data[i]) >= N_PER_CLASS for i in range(NUM_CLASSES))
            if all_full:
                print(f"\n  ✓ TẤT CẢ CÁC RỔ ĐÃ ĐẦY! Ra lệnh kết thúc...")
                break
        except Exception:
            # Check if workers are dead
            active_workers = sum(1 for w in workers if w.is_alive())

    # Graceful shutdown
    print("  Đang tắt các tiến trình...")
    reader_p.terminate()
    reader_p.join(timeout=5)
    for w in workers:
        w.terminate()
    for w in workers:
        w.join(timeout=5)

    elapsed = time.time() - t0
    print(f"\n{'─' * 64}")
    print(f"  Xử lý hoàn tất sau {elapsed/3600:.2f} giờ!")
    print("  Đang lưu vào Parquet...")

    all_rows = []
    for b in range(NUM_CLASSES):
        all_rows.extend(final_data[b])

    if all_rows:
        df = pl.DataFrame(all_rows)
        df = df.sample(fraction=1.0, seed=42) # Xáo trộn
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(OUTPUT_FILE), compression="zstd")
        print(f"  ✅ Đã lưu thành công: {OUTPUT_FILE} ({df.height:,} dòng)")
    else:
        print("  ⚠️ Không có dữ liệu để lưu!")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
