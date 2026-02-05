from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tabular_ml_lab.config import Config
from tabular_ml_lab.utils import ensure_dir

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


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest.as_posix())


def download_adult_dataset(cfg: Config) -> tuple[Path, Path]:
    raw_dir = ensure_dir(cfg.paths.data_raw_dir)
    train_path = raw_dir / "adult.data"
    test_path = raw_dir / "adult.test"
    download_file(cfg.dataset.train_url, train_path)
    download_file(cfg.dataset.test_url, test_path)
    return train_path, test_path


def _load_adult(path: Path, is_test: bool) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        names=ADULT_COLUMNS,
        sep=",",
        na_values="?",
        skipinitialspace=True,
        skiprows=1 if is_test else 0,
    )
    df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
    return df


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, test_path = download_adult_dataset(cfg)
    full_train = _load_adult(train_path, is_test=False)
    test_df = _load_adult(test_path, is_test=True)

    train_df, val_df = train_test_split(
        full_train,
        test_size=cfg.train.val_split,
        random_state=cfg.project.seed,
        stratify=full_train[cfg.dataset.target],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
