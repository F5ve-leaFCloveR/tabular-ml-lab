from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest.as_posix())


def load_adult(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, str]:
    data_dir = Path(data_dir)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    _download(ADULT_TRAIN_URL, train_path)
    _download(ADULT_TEST_URL, test_path)

    train_df = pd.read_csv(
        train_path,
        names=ADULT_COLUMNS,
        sep=",",
        na_values="?",
        skipinitialspace=True,
    )
    test_df = pd.read_csv(
        test_path,
        names=ADULT_COLUMNS,
        sep=",",
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    return df, "income"


def load_breast_cancer_dataset() -> tuple[pd.DataFrame, str]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    target = data.target.name or "target"
    df[target] = data.target
    return df, target


def load_iris_dataset() -> tuple[pd.DataFrame, str]:
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    target = data.target.name or "target"
    df[target] = data.target
    return df, target


def load_wine_dataset() -> tuple[pd.DataFrame, str]:
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    target = data.target.name or "target"
    df[target] = data.target
    return df, target


BUILTIN_DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, str]]] = {
    "Adult (UCI)": load_adult,
    "Breast Cancer (Sklearn)": load_breast_cancer_dataset,
    "Iris (Sklearn)": load_iris_dataset,
    "Wine (Sklearn)": load_wine_dataset,
}


def load_builtin(name: str) -> tuple[pd.DataFrame, str]:
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return BUILTIN_DATASETS[name]()


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
