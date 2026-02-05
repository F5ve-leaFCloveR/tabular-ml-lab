from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    seed: int = 42
    run_name: str = "baseline"


class PathsConfig(BaseModel):
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    model_dir: str = "models"
    reports_dir: str = "reports"


class DatasetConfig(BaseModel):
    name: str
    train_url: str
    test_url: str
    target: str
    categorical_features: list[str]
    numeric_features: list[str]


class TrainConfig(BaseModel):
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_layers: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    val_split: float = 0.15
    early_stopping_patience: int = 4


class MetricsConfig(BaseModel):
    threshold: float = 0.5


class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    dataset: DatasetConfig
    train: TrainConfig
    metrics: MetricsConfig


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return Config.model_validate(payload)
