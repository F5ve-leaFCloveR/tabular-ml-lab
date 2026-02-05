from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from tabular_ml_lab.model import MLPClassifier
from tabular_ml_lab.preprocess import preprocess_transform
from tabular_ml_lab.utils import read_json


def load_active(model_dir: str | Path = "models/active") -> dict[str, Any]:
    model_dir = Path(model_dir)
    metadata = read_json(model_dir / "metadata.json")
    model_type = metadata.get("model_type")

    if model_type == "sklearn":
        model = joblib.load(model_dir / "model.joblib")
        return {"model_type": model_type, "model": model, "metadata": metadata}

    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    return {
        "model_type": "torch",
        "model": model,
        "preprocessor": preprocessor,
        "metadata": {**metadata, **checkpoint},
    }


def predict_single(payload: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    model_type = artifacts["model_type"]
    metadata = artifacts["metadata"]

    if "features" in payload:
        record = payload["features"]
    else:
        record = payload

    df = pd.DataFrame([record])

    if model_type == "sklearn":
        model = artifacts["model"]
        pred = model.predict(df)
        response: dict[str, Any] = {"prediction": pred[0]}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            if proba.shape[1] == 2:
                response["probability"] = float(proba[0][1])
            else:
                response["probability"] = proba[0].tolist()
        return response

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    X = preprocess_transform(preprocessor, df)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        probs = torch.sigmoid(logits).numpy()
    return {"prediction": int(probs[0] >= 0.5), "probability": float(probs[0])}
