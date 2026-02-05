from __future__ import annotations

from pathlib import Path

import typer

from tabular_ml_lab.config import load_config
from tabular_ml_lab.data import download_adult_dataset
from tabular_ml_lab.eval import evaluate
from tabular_ml_lab.train import train_model

app = typer.Typer(help="Tabular ML Lab CLI")


@app.command()
def download(config: str = "configs/default.yaml") -> None:
    cfg = load_config(config)
    train_path, test_path = download_adult_dataset(cfg)
    typer.echo(f"Downloaded: {train_path} and {test_path}")


@app.command()
def train(config: str = "configs/default.yaml") -> None:
    cfg = load_config(config)
    result = train_model(cfg)
    typer.echo(f"Saved model: {result['model_path']}")


@app.command()
def evaluate_cmd(config: str = "configs/default.yaml") -> None:
    cfg = load_config(config)
    metrics = evaluate(cfg)
    typer.echo(f"Test metrics: {metrics}")


@app.command()
def pipeline(config: str = "configs/default.yaml") -> None:
    cfg = load_config(config)
    train_model(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    app()
