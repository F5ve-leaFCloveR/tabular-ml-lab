from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from tabular_ml_lab.lab.metrics import classification_metrics, regression_metrics
from tabular_ml_lab.preprocess import build_preprocessor


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    feature_names: list[str]
    best_params: Dict[str, Any] | None


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def _split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    task: str,
):
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if task == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def get_sklearn_model(task: str, name: str, params: Dict[str, Any]):
    if task == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(**params)
        if name == "Random Forest":
            return RandomForestClassifier(**params)
        if name == "Gradient Boosting":
            return GradientBoostingClassifier(**params)
    else:
        if name == "Linear Regression":
            return LinearRegression(**params)
        if name == "Random Forest":
            return RandomForestRegressor(**params)
        if name == "Gradient Boosting":
            return GradientBoostingRegressor(**params)

    raise ValueError(f"Unknown model: {name}")


def train_sklearn(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    task: str,
    model_name: str,
    params: Dict[str, Any],
    grid: Dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> TrainResult:
    X_train, X_test, y_train, y_test = _split_data(
        df, target, test_size=test_size, random_state=random_state, task=task
    )

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scale_numeric=scale_numeric,
    )
    estimator = get_sklearn_model(task, model_name, params)

    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    if task == "classification":
        target_type = type_of_target(y_train)
        if target_type in {"binary", "multiclass"}:
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

    best_params = None
    if grid:
        prefixed = {f"model__{k}": v for k, v in grid.items()}
        search = GridSearchCV(
            pipe,
            prefixed,
            cv=3,
            n_jobs=-1,
            scoring="accuracy" if task == "classification" else "r2",
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = pipe.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = None
    if task == "classification":
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = scores if scores.ndim == 1 else None

        metrics = classification_metrics(y_test, y_pred, y_proba)
    else:
        metrics = regression_metrics(y_test, y_pred)

    feature_names = _get_feature_names(preprocessor)

    return TrainResult(
        model=model,
        metrics=metrics,
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        y_proba=y_proba if y_proba is None else np.array(y_proba),
        feature_names=feature_names,
        best_params=best_params,
    )
