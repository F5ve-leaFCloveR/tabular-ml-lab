import numpy as np

from tabular_ml_lab.metrics import classification_metrics


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.6, 0.2])
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] >= 0.5
    assert metrics["roc_auc"] >= 0.5
