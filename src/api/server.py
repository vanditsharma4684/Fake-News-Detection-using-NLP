#!/usr/bin/env python3
"""
server.py
---------
FastAPI backend for the Fake News Verification System.

Endpoints
---------
POST /verify      → Full multi-source verification
POST /predict     → ML-only quick prediction (no API calls)
GET  /health      → Health check

Run:
    uvicorn src.api.server:app --reload --port 8000
"""
from __future__ import annotations
import os
import sys
import logging
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.verification.fact_verifier import FactVerifier, VerificationResult
from src.models.fake_news_classifier import FakeNewsClassifier
from src.preprocessing.text_clean import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fake News Verification API",
    description=(
        "Hybrid system: ML classifier + multi-source fact verification "
        "using NewsAPI and semantic similarity."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
_PIPELINE = MODELS_DIR / "pipeline.joblib"
_MODEL    = MODELS_DIR / "fake_news_model.joblib"
_VEC      = MODELS_DIR / "vectorizer.joblib"

_classifier = FakeNewsClassifier(
    pipeline_path=_PIPELINE,
    model_path=_MODEL,
    vectorizer_path=_VEC,
)

_verifier = FactVerifier(
    pipeline_path=_PIPELINE,
    model_path=_MODEL,
    vectorizer_path=_VEC,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Headline or article body")
    newsapi_key: str | None = Field(None, description="Optional NewsAPI key override")
    threshold: float = Field(0.45, ge=0.0, le=1.0, description="Credibility threshold")

class QuickPredictRequest(BaseModel):
    text: str = Field(..., min_length=5)
    threshold: float = Field(0.50, ge=0.0, le=1.0)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_model_loaded": _classifier.loaded,
    }


@app.post("/predict", summary="ML-only quick prediction (no web calls)")
def quick_predict(req: QuickPredictRequest):
    """Run the ML classifier only — fast, no external API needed."""
    cleaned = clean_text(req.text)
    label, prob = _classifier.predict(cleaned, threshold=req.threshold)
    return {
        "label": label,
        "fake_probability": round(prob, 4),
        "threshold": req.threshold,
    }


@app.post("/verify", summary="Full multi-source fact verification")
def verify(req: ArticleRequest):
    """
    Full pipeline:
      clean → extract claims → search news → semantic similarity
      → source credibility → ML classifier → final score.
    """
    try:
        _verifier.threshold = req.threshold
        result: VerificationResult = _verifier.verify(
            req.text,
            newsapi_key=req.newsapi_key,
        )
        return result.to_dict()
    except Exception as exc:
        logger.exception("Verification error")
        raise HTTPException(status_code=500, detail=str(exc))
