# API Contract — MMD-G2 AI Engine Server

> **Tài liệu này là bản cam kết (Contract) giữa Web Team (ChessWeb) và AI Team (MMD-G2).**
> Mọi thay đổi phải được thông báo trước cho cả hai bên.

---

## 1. Overview

AI Engine Server là một FastAPI service độc lập, chạy tại `http://<HOST>:8000`. Nó nhận dữ liệu ván cờ từ ChessWeb và trả về ELO dự đoán cùng lời giải thích từ LLM.

**Machine name (dev):** `localhost`  
**Production URL:** `http://<AI_ENGINE_HOST>:8000`

---

## 2. Endpoint Definition

### `POST /api/predict-elo`

Nhận dữ liệu ván cờ đã kết thúc, phân tích và trả về kết quả ELO + lời giải thích.

**Method:** `POST`  
**URL:** `http://<HOST>:8000/api/predict-elo`  
**Content-Type:** `application/json`

---

### 3. Request Schema

```json
{
  "pgn": "string",
  "clock_times": "number[]",
  "result": "string",
  "time_control": "string (optional, default: '5+0')"
}
```

| Field | Type | Required | Description |
|:------|:-----|:---------|:-----------|
| `pgn` | `string` | **YES** | Chuỗi PGN chuẩn quốc tế, chỉ chứa nước đi (không chứa header). Ví dụ: `"1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"` |
| `clock_times` | `number[]` | **YES** | Mảng số thực (giây), xen kẽ Trắng → Đen → Trắng → Đen... Độ dài = tổng số nước đi. |
| `result` | `string` | **YES** | Kết quả ván: `"1-0"` (Trắng thắng) \| `"0-1"` (Đen thắng) \| `"1/2-1/2"` (Hòa) |
| `time_control` | `string` | NO | Thể thức thời gian. Ví dụ: `"5+0"`, `"10+5"`. Mặc định: `"5+0"` |

**Example Request:**

```json
{
  "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O",
  "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3, 3.0, 7.2, 8.0],
  "result": "1-0",
  "time_control": "5+0"
}
```

---

### 4. Response Schema — Success

```json
{
  "success": true,
  "data": {
    "white_elo": 1540,
    "black_elo": 1200,
    "eco": {
      "code": "B50",
      "name": "Sicilian Defense"
    },
    "stats": {
      "white_avg_cpl": 25.3,
      "black_avg_cpl": 78.1,
      "white_blunders": 0,
      "black_blunders": 3,
      "total_moves": 42
    },
    "explanation": "Trang khai cuoc Phong thu Sicilian bien the B50 cuc ky bai ban. Den mac 3 sai lam nghiem trong tu nuoc 15-22, dac biet nuoc Hau d5 o nuoc 18 khiên mat kiem soat hoan toan trung tam..."
  }
}
```

| Field | Type | Description |
|:------|:-----|:-----------|
| `success` | `boolean` | Luôn là `true` khi xử lý thành công |
| `data.white_elo` | `number` | ELO dự đoán cho phe Trắng (range: 400–3000) |
| `data.black_elo` | `number` | ELO dự đoán cho phe Đen (range: 400–3000) |
| `data.eco.code` | `string` | Mã ECO khai cuộc. Ví dụ: `"B50"`, `"C42"`, `"A00"` |
| `data.eco.name` | `string` | Tên khai cuộc bằng tiếng Anh. Ví dụ: `"Sicilian Defense"` |
| `data.stats.white_avg_cpl` | `number` | Centipawn Loss trung bình của Trắng (số thực) |
| `data.stats.black_avg_cpl` | `number` | Centipawn Loss trung bình của Đen (số thực) |
| `data.stats.white_blunders` | `number` | Số nước đi tệ (CPL > 200) của Trắng |
| `data.stats.black_blunders` | `number` | Số nước đi tệ (CPL > 200) của Đen |
| `data.stats.total_moves` | `number` | Tổng số nước đi trong ván |
| `data.explanation` | `string` | Lời giải thích bằng ngôn ngữ tự nhiên (tiếng Việt), do LLM sinh ra |

---

### 5. Response Schema — Error

```json
{
  "success": false,
  "error": "Error description string"
}
```

| Field | Type | Description |
|:------|:-----|:-----------|
| `success` | `boolean` | Luôn là `false` khi có lỗi |
| `error` | `string` | Mô tả lỗi bằng tiếng Anh |

**AI Team must handle:**
- Invalid PGN format → return `success: false`
- Empty `clock_times` → return `success: false`
- Stockfish crash → return `success: false`, still populate `error`
- LLM API failure → **vẫn trả về `success: true`** với `explanation` là fallback string

**Web Team handles gracefully:**
- Nếu AI Engine không phản hồi trong 30s → ChessWeb hiện error message
- Nếu AI Engine trả `success: false` → ChessWeb hiện error message
- Dù có lỗi gì, ChessWeb vẫn hiển thị kết quả cơ bản (Thắng/Thua/Hòa)

---

### 6. Pipeline Processing (5 Bước)

```
Input: PGN + Clock Times
  │
  ├─> Step 1: ECO Classification
  │      File: src/ai_engine/eco_classifier.py
  │      Output: { code: "B50", name: "Sicilian Defense" }
  │
  ├─> Step 2: Stockfish Analysis
  │      File: src/ai_engine/stockfish_analyzer.py
  │      Output: { white_avg_cpl, black_avg_cpl, white_blunders, black_blunders }
  │
  ├─> Step 3: Feature Engineering
  │      File: src/features/feature_engineering.py
  │      Output: Tensor / feature vector
  │
  ├─> Step 4: Model Inference
  │      File: src/ai_engine/model_predictor.py
  │      Output: { white_elo, black_elo }
  │
  └─> Step 5: LLM Explanation
         File: src/ai_engine/llm_explainer.py
         Output: explanation string (tiếng Việt)

Output: JSON response
```

---

### 7. Behavior Notes

1. **Startup:** Model weights chỉ load **1 lần duy nhất** khi server khởi động. Không được load lại mỗi request.
2. **CORS:** AI Engine `allow_origins=["*"]` — không cần authentication cho PoC.
3. **Stockfish depth:** Khuyến nghị depth 15–20 cho PoC. Giảm xuống 12 nếu response time > 30s.
4. **ECO fallback:** Nếu không khớp ECO → trả về `{"code": "A00", "name": "Uncommon Opening"}`.
5. **LLM fallback:** Nếu không có API key hoặc LLM lỗi → trả về chuỗi mặc định, **không** return error.
6. **Timeout:** Web Server timeout sau 30s. AI Team cố gắng giữ pipeline < 20s.

---

### 8. Environment Variables

```env
PORT=8000
AI_ENGINE_HOST=0.0.0.0
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
MODEL_PATH=models/rating_net_v1/model_best.pth
STOCKFISH_PATH=/usr/local/bin/stockfish
STOCKFISH_DEPTH=15
```

---

### 9. Verification Checklist (AI Team)

- [ ] Server khởi động thành công tại `http://localhost:8000`
- [ ] Truy cập `http://localhost:8000/docs` → Swagger UI hiển thị
- [ ] Gửi POST `/api/predict-elo` với PGN hợp lệ → nhận đúng JSON format
- [ ] ECO trả về đúng mã (test: `"1. e4 e5"` → C20)
- [ ] Stockfish chạy được, CPL và Blunders có giá trị hợp lệ
- [ ] Model inference trả ELO trong khoảng 400–3000
- [ ] LLM trả về tiếng Việt, đề cập đúng khai cuộc và số liệu
- [ ] Gửi PGN sai format → `success: false`, không crash server
- [ ] Response time toàn pipeline < 30 giây
- [ ] Model chỉ load 1 lần khi startup
- [ ] Health endpoint: `GET /health` → `{"status": "ok"}`

---

### 10. Testing with curl

```bash
curl -X POST http://localhost:8000/api/predict-elo \
  -H "Content-Type: application/json" \
  -d '{
    "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
    "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3, 3.0, 7.2],
    "result": "1-0",
    "time_control": "5+0"
  }'
```

---

*Document version: 1.0 — Created 2026-04-28*
