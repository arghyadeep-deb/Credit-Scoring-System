from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import HybridCreditScorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

scorer = HybridCreditScorer.load(ARTIFACTS_DIR)

app = FastAPI(title="Credit Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Encoded model feature map. Missing features default to 0.0."
    )


class PredictResponse(BaseModel):
    decision: str
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    top_drivers: list[str]
    reasoning: str
    rejection_label: str


class PredictRawRequest(BaseModel):
    raw_fields: dict[str, Any] = Field(
        ..., description="Raw applicant fields. Numeric + categorical fields from /raw-schema."
    )


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    input_type: Literal["encoded", "raw"] = "encoded"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "feature_columns": scorer.feature_columns,
        "classes": [str(c) for c in scorer.label_encoder.classes_],
        "rejection_label": str(scorer.rejection_label),
        "blend_weights": {
            "ensemble": float(scorer.blend_weights[0]),
            "nn": float(scorer.blend_weights[1]),
        },
    }


@app.get("/raw-schema")
def raw_schema() -> dict[str, Any]:
    return scorer.get_raw_schema()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = scorer.predict_one(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(**result)


@app.post("/predict-raw", response_model=PredictResponse)
def predict_raw(payload: PredictRawRequest) -> PredictResponse:
    try:
        result = scorer.predict_one_raw(payload.raw_fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(**result)


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictRequest) -> dict[str, Any]:
    try:
        results = scorer.predict_batch(payload.records, input_type=payload.input_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}") from exc

    return {
        "count": len(results),
        "input_type": payload.input_type,
        "results": results,
    }
