import pandas as pd

from tabular_ml_lab.preprocess import build_preprocessor, preprocess_fit_transform


def test_preprocessor_shapes():
    df = pd.DataFrame(
        {
            "age": [25, 45],
            "hours-per-week": [40, 60],
            "workclass": ["Private", "Self-emp"],
            "education": ["Bachelors", "HS-grad"],
        }
    )
    pre = build_preprocessor(
        categorical_features=["workclass", "education"],
        numeric_features=["age", "hours-per-week"],
    )
    X, _ = preprocess_fit_transform(pre, df)
    assert X.shape[0] == 2
