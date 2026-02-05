from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool = True,
    encode_categorical: bool = True,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)

    if encode_categorical:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        cat_transformer = categorical_pipe
    else:
        cat_transformer = "drop"

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", cat_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def preprocess_fit_transform(
    preprocessor: ColumnTransformer, X
) -> Tuple[np.ndarray, ColumnTransformer]:
    X_transformed = preprocessor.fit_transform(X)
    return to_dense(X_transformed), preprocessor


def preprocess_transform(preprocessor: ColumnTransformer, X) -> np.ndarray:
    X_transformed = preprocessor.transform(X)
    return to_dense(X_transformed)
