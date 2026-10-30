"""FastAPI surface for scoring reviews and clustering complaints.

The model is loaded lazily so tests and the /themes endpoint do not need
torch or a trained checkpoint.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .labels import LABELS, clean_text
from .themes import cluster_complaints

app = FastAPI(title="ComplaintGraph", version="0.1.0")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/bert-tiny-app-review-sentiment"))


class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=500)


class Theme(BaseModel):
    cluster: int
    size: int
    top_terms: list[str]
    examples: list[str]


class ThemesResponse(BaseModel):
    themes: list[Theme]


class Score(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class ScoreResponse(BaseModel):
    scores: list[Score]


@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"model not found at {MODEL_DIR}; run `python -m complaintgraph.train`")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL_DIR.exists()}


@app.post("/themes", response_model=ThemesResponse)
def themes(req: ReviewRequest):
    cleaned = [clean_text(r) for r in req.reviews if clean_text(r)]
    return {"themes": cluster_complaints(cleaned)}


@app.post("/score", response_model=ScoreResponse)
def score(req: ReviewRequest):
    try:
        tokenizer, model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    import torch

    scores = []
    for text in req.reviews:
        enc = tokenizer(clean_text(text), truncation=True, padding=True, max_length=96, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
        best = int(max(range(len(probs)), key=probs.__getitem__))
        scores.append({
            "label": LABELS[best],
            "confidence": float(probs[best]),
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        })
    return {"scores": scores}
