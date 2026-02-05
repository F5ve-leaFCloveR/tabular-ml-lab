from __future__ import annotations

from typing import Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype


def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col in df.columns:
        if col == target:
            continue
        if is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features
