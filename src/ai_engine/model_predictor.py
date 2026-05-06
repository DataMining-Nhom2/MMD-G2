# src/ai_engine/model_predictor.py
# Model inference — predicts ELO from game features (stub)

import random


MODEL_LOADED = False


def load_model():
    global MODEL_LOADED
    if not MODEL_LOADED:
        # Try to load actual model if available
        try:
            import os
            model_path = os.environ.get('MODEL_PATH', 'models/rating_net_v1/model_best.pth')
            if os.path.exists(model_path):
                import torch
                # NOTE: RatingNet architecture needs to match saved weights
                # For now, this is a stub
                pass
        except Exception as e:
            print(f"[AI Engine] Could not load model: {e}")
        MODEL_LOADED = True


def predict_elo(pgn: str, clock_times: list[float], stockfish_features: dict) -> dict:
    """
    Predict ELO for white and black players.
    Currently stub — returns realistic mock ELO values.
    AI Team will implement actual model inference.
    """
    load_model()

    total_moves = stockfish_features.get('total_moves', 40)
    white_cpl = stockfish_features.get('white_avg_cpl', 30)
    black_cpl = stockfish_features.get('black_avg_cpl', 30)

    # Stub: estimate ELO based on CPL (lower CPL = higher ELO)
    # These formulas are just placeholders
    base_elo = 1500
    white_elo = max(400, min(3000, int(base_elo + (50 - white_cpl) * 10 + random.uniform(-50, 50))))
    black_elo = max(400, min(3000, int(base_elo + (50 - black_cpl) * 10 + random.uniform(-50, 50))))

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
    }
