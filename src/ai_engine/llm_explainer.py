# src/ai_engine/llm_explainer.py
# LLM-based game explanation (stub)

import os
import asyncio


LLM_CLIENT = None


def get_llm_client():
    global LLM_CLIENT
    if LLM_CLIENT is None:
        # Try to initialize Gemini or OpenAI client
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if api_key:
            # NOTE: AI Team will implement actual LLM integration
            pass
    return LLM_CLIENT


async def generate_explanation(eco, analysis, elo, game_result, time_control) -> str:
    """
    Generate a natural language explanation of the game using LLM.
    Currently stub — returns a hardcoded message.
    AI Team will implement actual LLM API call.
    """
    # Try LLM if API key is available
    client = get_llm_client()
    if client:
        try:
            # AI Team: implement actual LLM call here
            pass
        except Exception as e:
            print(f"[LLM] Error: {e}")

    # Fallback: return structured explanation
    white_elo = elo.get('white_elo', '?')
    black_elo = elo.get('black_elo', '?')
    eco_name = eco.get('name', 'Unknown')
    white_cpl = analysis.get('white_avg_cpl', 0)
    black_cpl = analysis.get('black_avg_cpl', 0)
    white_blunders = analysis.get('white_blunders', 0)
    black_blunders = analysis.get('black_blunders', 0)

    result_map = {
        '1-0': 'Trắng thắng',
        '0-1': 'Đen thắng',
        '1/2-1/2': 'Hòa'
    }
    result_text = result_map.get(game_result, game_result)

    return (
        f"{result_text} với khai cuộc {eco_name}. "
        f"Trắng (ELO ~{white_elo}) có CPL trung bình {white_cpl:.1f} với {white_blunders} sai lầm nghiêm trọng. "
        f"Đen (ELO ~{black_elo}) có CPL trung bình {black_cpl:.1f} với {black_blunders} sai lầm nghiêm trọng. "
        "Phân tích chi tiết từ AI đang được cập nhật."
    )
