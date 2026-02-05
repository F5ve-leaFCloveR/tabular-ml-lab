from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabular_ml_lab.config import Config
from tabular_ml_lab.data import load_dataset
from tabular_ml_lab.metrics import classification_metrics
from tabular_ml_lab.model import MLPClassifier
from tabular_ml_lab.preprocess import (
    build_preprocessor,
    preprocess_fit_transform,
    preprocess_transform,
)
from tabular_ml_lab.utils import ensure_dir, save_json, set_seed


LABEL_POSITIVE = ">50K"


def encode_target(series) -> np.ndarray:
    return (series == LABEL_POSITIVE).astype(int).to_numpy()


def build_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(y).float()
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(cfg: Config) -> dict:
    set_seed(cfg.project.seed)

    train_df, val_df, _ = load_dataset(cfg)
    preprocessor = build_preprocessor(cfg.dataset.categorical_features, cfg.dataset.numeric_features)

    X_train, preprocessor = preprocess_fit_transform(preprocessor, train_df)
    X_val = preprocess_transform(preprocessor, val_df)

    y_train = encode_target(train_df[cfg.dataset.target])
    y_val = encode_target(val_df[cfg.dataset.target])

    train_loader = build_dataloader(X_train, y_train, cfg.train.batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, cfg.train.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_layers=cfg.train.hidden_layers,
        dropout=cfg.train.dropout,
    ).to(device)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_val_loss = float("inf")
    patience = 0
    history_rows: list[dict] = []
    best_state = None

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                val_probs.append(probs)
                val_targets.append(yb.detach().cpu().numpy())

        val_probs_np = np.concatenate(val_probs)
        val_targets_np = np.concatenate(val_targets)

        metrics = classification_metrics(val_targets_np, val_probs_np, cfg.metrics.threshold)
        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **metrics,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                break

    model_dir = ensure_dir(cfg.paths.model_dir)
    reports_dir = ensure_dir(cfg.paths.reports_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = model_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden_layers": cfg.train.hidden_layers,
            "dropout": cfg.train.dropout,
            "label_positive": LABEL_POSITIVE,
        },
        model_path,
    )

    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    history_path = reports_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)

    metrics_path = reports_dir / "val_metrics.json"
    save_json(metrics_path, history_rows[-1])

    metadata = {
        "run_name": cfg.project.run_name,
        "seed": cfg.project.seed,
        "device": str(device),
        "features": {
            "categorical": cfg.dataset.categorical_features,
            "numeric": cfg.dataset.numeric_features,
        },
        "target": cfg.dataset.target,
        "label_positive": LABEL_POSITIVE,
        "model_path": model_path.as_posix(),
        "preprocessor_path": preprocessor_path.as_posix(),
    }
    save_json(model_dir / "metadata.json", metadata)

    return {
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "val_metrics": history_rows[-1],
        "history_path": history_path,
    }
