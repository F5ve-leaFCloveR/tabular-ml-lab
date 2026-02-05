from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from tabular_ml_lab.preprocess import preprocess_transform
from tabular_ml_lab.utils import read_json


def load_artifacts(model_dir: str | Path):
    model_dir = Path(model_dir)
    metadata = read_json(model_dir / "metadata.json")
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    from tabular_ml_lab.model import MLPClassifier

    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    return model, preprocessor, metadata


def predict_proba(
    records: list[dict[str, Any]],
    model,
    preprocessor,
) -> list[float]:
    df = pd.DataFrame(records)
    X = preprocess_transform(preprocessor, df)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        probs = torch.sigmoid(logits).numpy()
    return probs.tolist()
