from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabular_ml_lab.model import MLPClassifier
from tabular_ml_lab.preprocess import build_preprocessor, preprocess_fit_transform, preprocess_transform
from tabular_ml_lab.lab.metrics import classification_metrics


@dataclass
class TorchResult:
    model: torch.nn.Module
    metrics: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    checkpoint: Dict[str, Any]
    preprocessor: Any


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
) -> tuple[torch.nn.Module, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(device)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = _build_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, batch_size, shuffle=False)

    best_val = float("inf")
    patience = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_loss": best_val, "device": str(device)}


def train_and_eval_binary(
    df,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    params: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> TorchResult:
    X = df.drop(columns=[target])
    y = df[target]

    if y.nunique() > 2:
        raise ValueError("PyTorch MLP supports only binary classification in this UI.")

    classes = list(y.unique())
    y_encoded = (y == classes[1]).astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=params.get("val_split", 0.2),
        random_state=random_state,
        stratify=y_train,
    )

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scale_numeric=scale_numeric,
    )

    X_train_np, preprocessor = preprocess_fit_transform(preprocessor, X_train)
    X_val_np = preprocess_transform(preprocessor, X_val)
    X_test_np = preprocess_transform(preprocessor, X_test)

    model, meta = _train_binary_classifier(
        X_train_np,
        y_train,
        X_val_np,
        y_val,
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        early_stopping_patience=params["early_stopping_patience"],
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_np).float())
        probs = torch.sigmoid(logits).numpy()

    preds = (probs >= 0.5).astype(int)
    metrics = classification_metrics(y_test, preds, probs)
    metrics.update(meta)

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": X_train_np.shape[1],
        "hidden_layers": params["hidden_layers"],
        "dropout": params["dropout"],
        "positive_label": classes[1],
        "negative_label": classes[0],
    }

    return TorchResult(
        model=model,
        metrics=metrics,
        y_true=y_test,
        y_pred=preds,
        y_proba=probs,
        checkpoint=checkpoint,
        preprocessor=preprocessor,
    )
