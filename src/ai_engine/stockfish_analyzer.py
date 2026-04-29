# src/ai_engine/stockfish_analyzer.py
# Stockfish analysis — computes CPL and Blunders (stub)

import chess.pgn
import io
import random


def analyze_game(pgn_str: str) -> dict:
    """
    Analyze a PGN game to compute CPL (Centipawn Loss) and Blunders.
    Currently stub — returns mock data with realistic values.
    AI Team will integrate actual Stockfish analysis.
    """
    total_moves = 0
    try:
        pgn_io = io.StringIO(pgn_str)
        game = chess.pgn.read_game(pgn_io)
        if game:
            total_moves = len(list(game.mainline_moves()))
    except Exception:
        pass

    if total_moves == 0:
        total_moves = random.randint(20, 80)

    # Simulate CPL based on game length
    white_avg_cpl = random.uniform(15, 60)
    black_avg_cpl = random.uniform(15, 60)
    white_blunders = random.randint(0, 3)
    black_blunders = random.randint(0, 4)

    return {
        "white_avg_cpl": white_avg_cpl,
        "black_avg_cpl": black_avg_cpl,
        "white_blunders": white_blunders,
        "black_blunders": black_blunders,
        "total_moves": total_moves,
        "per_move_cpl": [],
    }
