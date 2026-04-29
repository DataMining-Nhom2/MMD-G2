# src/ai_engine/pipeline.py
# Orchestrator — runs 5-step pipeline (stub for PoC)

import asyncio


async def run_prediction_pipeline(
    pgn: str,
    clock_times: list[float],
    game_result: str,
    time_control: str,
) -> dict:
    """
    5-step pipeline stub:
    1. ECO classification
    2. Stockfish analysis (CPL + Blunders)
    3. Feature engineering
    4. Model inference (ELO)
    5. LLM explanation

    Currently returns mock data. AI Team will replace each step.
    """
    # Simulate async processing
    await asyncio.sleep(0.1)

    # Step 1: ECO classification
    from src.ai_engine.eco_classifier import classify_eco
    eco = classify_eco(pgn)

    # Step 2: Stockfish analysis
    from src.ai_engine.stockfish_analyzer import analyze_game
    analysis = analyze_game(pgn)

    # Step 3+4: Model inference
    from src.ai_engine.model_predictor import predict_elo
    elo_result = predict_elo(pgn, clock_times, analysis)

    # Step 5: LLM explanation
    from src.ai_engine.llm_explainer import generate_explanation
    explanation = await generate_explanation(eco, analysis, elo_result, game_result, time_control)

    return {
        "white_elo": elo_result["white_elo"],
        "black_elo": elo_result["black_elo"],
        "eco": eco,
        "stats": {
            "white_avg_cpl": round(analysis["white_avg_cpl"], 1),
            "black_avg_cpl": round(analysis["black_avg_cpl"], 1),
            "white_blunders": analysis["white_blunders"],
            "black_blunders": analysis["black_blunders"],
            "total_moves": analysis["total_moves"],
        },
        "explanation": explanation,
    }
