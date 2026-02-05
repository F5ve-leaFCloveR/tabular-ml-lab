from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, precision_recall_curve

from tabular_ml_lab.config import Config
from tabular_ml_lab.data import load_dataset
from tabular_ml_lab.metrics import classification_metrics
from tabular_ml_lab.preprocess import preprocess_transform
from tabular_ml_lab.train import LABEL_POSITIVE, encode_target
from tabular_ml_lab.utils import ensure_dir, save_json


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    from tabular_ml_lab.model import MLPClassifier

    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate(cfg: Config) -> dict:
    _, _, test_df = load_dataset(cfg)

    model_path = Path(cfg.paths.model_dir) / "model.pt"
    preprocessor_path = Path(cfg.paths.model_dir) / "preprocessor.joblib"

    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_test = preprocess_transform(preprocessor, test_df)
    y_test = encode_target(test_df[cfg.dataset.target])

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        probs = torch.sigmoid(logits).numpy()

    metrics = classification_metrics(y_test, probs, cfg.metrics.threshold)

    reports_dir = ensure_dir(cfg.paths.reports_dir)
    save_json(reports_dir / "test_metrics.json", metrics)

    report_txt = classification_report(y_test, (probs >= cfg.metrics.threshold).astype(int), digits=4)
    (reports_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        (probs >= cfg.metrics.threshold).astype(int),
        display_labels=["<=50K", LABEL_POSITIVE],
        cmap="Blues",
    )
    disp.figure_.tight_layout()
    disp.figure_.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    plt.close(disp.figure_)

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=120)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=120)
    plt.close()

    return metrics
