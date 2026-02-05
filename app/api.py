from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Body, FastAPI

from tabular_ml_lab.lab.serving import load_active, predict_single


@lru_cache(maxsize=1)
def _artifacts():
    return load_active("models/active")


app = FastAPI(title="Tabular ML Lab API", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    artifacts = _artifacts()
    return artifacts.get("metadata", {})


@app.post("/predict")
def predict(payload: dict[str, Any] = Body(...)):
    artifacts = _artifacts()
    return predict_single(payload, artifacts)
