# src/ai_engine/main.py
# FastAPI entry point — single endpoint /api/predict-elo

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="MMD-G2 AI Engine",
    description="Chess ELO Prediction API — PoC Stub",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    pgn: str
    clock_times: list[float]
    result: str = "1-0"
    time_control: str = "5+0"


class PredictResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


# ─── Endpoint ────────────────────────────────────────────────────────────────

@app.post("/api/predict-elo", response_model=PredictResponse)
async def predict_elo_endpoint(request: PredictRequest):
    try:
        from src.ai_engine.pipeline import run_prediction_pipeline
        result = await run_prediction_pipeline(
            pgn=request.pgn,
            clock_times=request.clock_times,
            game_result=request.result,
            time_control=request.time_control,
        )
        return PredictResponse(success=True, data=result)
    except Exception as e:
        return PredictResponse(success=False, error=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-engine"}
