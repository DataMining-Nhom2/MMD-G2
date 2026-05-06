"""
AI Integration - Mock Interfaces cho AI Model và XAI Engine
ĐÂY LÀ CONTRACT - Thành viên 2 & 3 sẽ thay thế implementation
"""
import random


def predict_elo(pgn_string: str, clock_times: list[float]) -> dict:
    """
    Dự đoán ELO từ ván cờ.

    Args:
        pgn_string: Chuỗi PGN chuẩn (ví dụ: "1. e4 e5 2. Nf3 Nc6")
        clock_times: Danh sách thời gian suy nghĩ (giây), xen kẽ Trắng-Đen

    Returns:
        dict với cấu trúc:
        {
            "white_elo": int,      # ELO dự đoán cho Trắng
            "black_elo": int,      # ELO dự đoán cho Đen
            "stats": {
                "white_avg_cpl": float,
                "black_avg_cpl": float,
                "white_blunders": int,
                "black_blunders": int,
            }
        }
    """
    # MOCK - Thành viên 2 sẽ thay thế bằng model thực
    # Hiện tại trả về random values

    # Seed để deterministic cho cùng input
    seed = hash(pgn_string) % 1000000
    rng = random.Random(seed)

    # Tính toán stats giả lập dựa trên clock times
    if clock_times:
        avg_white = sum(clock_times[::2]) / max(len(clock_times[::2]), 1)
        avg_black = sum(clock_times[1::2]) / max(len(clock_times[1::2]), 1)
    else:
        avg_white = 30.0
        avg_black = 30.0

    # ELO mock - trong range 1000-2000
    white_elo = rng.randint(1000, 2000)
    black_elo = rng.randint(1000, 2000)

    # Stats mock
    white_blunders = rng.randint(0, 5)
    black_blunders = rng.randint(0, 5)

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "stats": {
            "white_avg_cpl": round(avg_white, 1),
            "black_avg_cpl": round(avg_black, 1),
            "white_blunders": white_blunders,
            "black_blunders": black_blunders,
        }
    }


def get_explanation(prediction_result: dict) -> str:
    """
    Sinh lời giải thích cho kết quả dự đoán ELO.

    Args:
        prediction_result: Output từ predict_elo()

    Returns:
        str: Lời giải thích bằng ngôn ngữ tự nhiên (tiếng Việt)
    """
    # MOCK - Thành viên 3 sẽ thay thế bằng LLM thực

    white_elo = prediction_result['white_elo']
    black_elo = prediction_result['black_elo']
    white_avg_cpl = prediction_result['stats']['white_avg_cpl']
    black_avg_cpl = prediction_result['stats']['black_avg_cpl']
    white_blunders = prediction_result['stats']['white_blunders']
    black_blunders = prediction_result['stats']['black_blunders']

    # So sánh ELO
    if white_elo > black_elo:
        diff = white_elo - black_elo
        comparison = f"Trắng được đánh giá cao hơn Đen khoảng {diff} ELO"
    elif black_elo > white_elo:
        diff = black_elo - white_elo
        comparison = f"Đen được đánh giá cao hơn Trắng khoảng {diff} ELO"
    else:
        comparison = "Hai người chơi có trình độ tương đương"

    # Chất lượng nước đi
    if white_avg_cpl < black_avg_cpl:
        white_quality = "chất lượng nước đi tốt hơn"
    elif black_avg_cpl < white_avg_cpl:
        white_quality = "chất lượng nước đi thấp hơn"
    else:
        white_quality = "chất lượng nước đi tương đương"

    # Blunders
    if white_blunders > black_blunders:
        blunder_comment = f"Trắng mắc nhiều sai lầm hơn ({white_blunders} blunders) so với Đen ({black_blunders} blunders)"
    elif black_blunders > white_blunders:
        blunder_comment = f"Đen mắc nhiều sai lầm hơn ({black_blunders} blunders) so với Trắng ({white_blunders} blunders)"
    else:
        blunder_comment = "Cả hai đều có số lượng sai lầm tương đương"

    explanation = f"""
Dựa trên phân tích ván cờ:

- {comparison}
- Trắng có CPL trung bình: {white_avg_cpl}s, Đen: {black_avg_cpl}s. Trắng có {white_quality}.
- {blunder_comment}

*Lưu ý: Đây là kết quả mock. Khi module AI thực được tích hợp, kết quả sẽ chính xác hơn.*
    """.strip()

    return explanation
