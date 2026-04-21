"""Helper: Trích xuất clock times từ PGN annotations.

Module này cung cấp các hàm để:
1. Parse %clk annotations từ PGN text (Lichess format)
2. Tính time spent mỗi nước từ clock remaining sequence
3. Dùng trong preprocessing để tạo dữ liệu có time_per_move

Tại sao cần module này:
    Dữ liệu hiện tại (sample_30k.parquet, lichess_*_ml.parquet) đã bị
    strip mất %clk annotations trong bước preprocessing (variation_san()).
    Để có per-move time data, cần re-process từ file PGN.zst gốc với
    src/preprocessing_with_clocks.py.

Format %clk trong Lichess PGN:
    1. e4 { [%clk 0:15:00] } e5 { [%clk 0:14:58] } 2. Nf3 { [%clk 0:14:55] }
    → Giây còn lại của bên vừa đi SAU khi đi nước đó
    → White ply 0: 900s → White ply 2: 895s → time_spent = 900 - 895 + increment = 5s

Ví dụ sử dụng:
    # Trong preprocessing_with_clocks.py:
    pgn_text = ... # PGN text đầy đủ với comments
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    clock_seq = extract_clock_from_game(game)
    time_spent = compute_time_spent(clock_seq, increment=0)
"""

from __future__ import annotations

import re

import chess.pgn

# Pattern tìm %clk H:MM:SS trong PGN comments
_CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
# Pattern tìm %emt (elapsed move time) — format thay thế
_EMT_PATTERN = re.compile(r"\[%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")


def parse_clk_string(h: str, m: str, s: str) -> float:
    """Chuyển H:MM:SS → giây (float)."""
    return float(h) * 3600 + float(m) * 60 + float(s)


def extract_clock_remaining_from_text(moves_with_clk: str) -> list[float]:
    """Parse %clk annotations từ chuỗi PGN text → list giây còn lại mỗi ply.

    Args:
        moves_with_clk: Chuỗi PGN có chứa { [%clk H:MM:SS] } comments

    Returns:
        Danh sách giây còn lại sau mỗi ply (theo thứ tự ply).
        [] nếu không tìm thấy %clk.
    """
    times: list[float] = []
    for match in _CLK_PATTERN.finditer(moves_with_clk):
        times.append(parse_clk_string(match.group(1), match.group(2), match.group(3)))
    return times


def extract_clock_from_game(game: chess.pgn.Game) -> list[float]:
    """Lấy clock remaining từ chess.pgn.Game object (dùng node.clock()).

    Đây là cách CHÍNH XÁC nhất để lấy clock times từ PGN đã parse.

    Args:
        game: chess.pgn.Game đã được parse với python-chess

    Returns:
        Danh sách giây còn lại (float) theo thứ tự ply.
        NaN nếu một ply không có clock annotation.
    """
    clock_seq: list[float] = []
    for node in game.mainline():
        clk = node.clock()
        clock_seq.append(float(clk) if clk is not None else float("nan"))
    return clock_seq


def extract_emt_from_game(game: chess.pgn.Game) -> list[float]:
    """Lấy elapsed move time (giây spent per ply) từ %emt annotations.

    %emt là thời gian SPENT (ngược với %clk là thời gian còn lại).
    Một số Lichess exports có %emt thay vì %clk.

    Returns:
        Danh sách giây spent per ply.
    """
    emt_seq: list[float] = []
    for node in game.mainline():
        # python-chess hỗ trợ node.emt() từ version 1.9+
        # Nếu không có, fallback về parse comment thủ công
        try:
            emt = node.emt()
            emt_seq.append(float(emt) if emt is not None else float("nan"))
        except AttributeError:
            # Fallback: parse từ comment text
            comment = node.comment or ""
            match = _EMT_PATTERN.search(comment)
            if match:
                emt_seq.append(
                    parse_clk_string(match.group(1), match.group(2), match.group(3))
                )
            else:
                emt_seq.append(float("nan"))
    return emt_seq


def compute_time_spent_from_remaining(
    clock_remaining: list[float],
    increment: float = 0.0,
) -> list[float]:
    """Tính giây SPENT mỗi ply từ danh sách giây CÒN LẠI (%clk).

    Quy tắc:
        - Trắng đi ply 0, 2, 4, ... (chỉ số chẵn)
        - Đen đi ply 1, 3, 5, ... (chỉ số lẻ)
        - time_spent[i] = clock_remaining[i-2] - clock_remaining[i] + increment
          (vì sau khi đi, clock +increment, và đây là của cùng một bên)
        - 2 ply đầu: không có ply trước đó của cùng bên → set = 0.0

    Args:
        clock_remaining: giây còn lại sau mỗi ply (từ %clk)
        increment: giây increment mỗi nước (từ cột Increment trong data)

    Returns:
        Danh sách giây spent per ply.
    """
    n = len(clock_remaining)
    spent = [0.0] * n
    for i in range(2, n):
        prev = clock_remaining[i - 2]
        curr = clock_remaining[i]
        # Bỏ qua NaN
        if prev != prev or curr != curr:  # IEEE NaN check
            spent[i] = float("nan")
        else:
            spent[i] = max(0.0, prev - curr + increment)
    return spent


def normalize_time_spent(
    time_spent: list[float],
    base_time: float,
) -> list[float]:
    """Normalize time_spent về tỷ lệ so với base_time.

    Mục đích: loại bỏ bias giữa ván Rapid (900s) và Classical (3600s).
    Theo paper arXiv:2409.11506: normalize clock-time input.

    Args:
        time_spent: giây spent mỗi ply
        base_time: tổng thời gian ban đầu mỗi bên (giây)

    Returns:
        time_spent / base_time (float ∈ [0, 1] thông thường)
    """
    if base_time <= 0:
        return [0.0] * len(time_spent)
    return [
        (t / base_time if t == t else float("nan"))  # NaN propagation
        for t in time_spent
    ]
