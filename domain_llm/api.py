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
