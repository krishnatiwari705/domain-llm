"""FastAPI service for Banking77 predictions."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import IntentPredictor


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
class PredictResponse(BaseModel):
    intent: str
    raw_output: str


def create_app() -> FastAPI:
    app = FastAPI(title="DomainLLM", version="0.1.0")

    @lru_cache
    def predictor() -> IntentPredictor:
        return IntentPredictor(os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"), os.getenv("LABELS_PATH", "data/processed/labels.json"), os.getenv("ADAPTER_PATH") or None, os.getenv("DEVICE_MAP", "auto"))
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            intent, raw = predictor().predict(request.text)
            return PredictResponse(intent=intent, raw_output=raw)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model is unavailable") from exc

    return app


app = create_app()
