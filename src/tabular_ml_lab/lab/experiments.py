from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import torch


@dataclass
class RunArtifacts:
    model_type: str
    model_path: Path
    preprocessor_path: Path | None = None


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run(
    base_dir: str | Path,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    model,
    model_type: str,
    preprocessor=None,
) -> tuple[Path, RunArtifacts]:
    base_dir = Path(base_dir)
    run_id = _now_id()
    run_dir = base_dir / "reports" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "metrics.json", metrics)

    if model_type == "sklearn":
        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)
        artifacts = RunArtifacts(model_type=model_type, model_path=model_path)
    else:
        model_path = run_dir / "model.pt"
        torch.save(model, model_path)
        preprocessor_path = run_dir / "preprocessor.joblib"
        if preprocessor is not None:
            joblib.dump(preprocessor, preprocessor_path)
        artifacts = RunArtifacts(
            model_type=model_type,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
        )

    summary_path = base_dir / "reports" / "experiments.csv"
    row = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **{k: metadata.get(k) for k in ["dataset", "model", "task"]},
        **metrics,
    }

    if not summary_path.exists():
        summary_path.write_text(",".join(row.keys()) + "\n", encoding="utf-8")

    with summary_path.open("a", encoding="utf-8") as f:
        f.write(",".join(str(row.get(k, "")) for k in row.keys()) + "\n")

    return run_dir, artifacts


def export_active(
    base_dir: str | Path,
    metadata: Dict[str, Any],
    artifacts: RunArtifacts,
) -> Path:
    base_dir = Path(base_dir)
    active_dir = base_dir / "models" / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    for item in active_dir.iterdir():
        if item.is_file():
            item.unlink()

    _write_json(active_dir / "metadata.json", metadata)

    if artifacts.model_type == "sklearn":
        shutil.copy2(artifacts.model_path, active_dir / "model.joblib")
    else:
        shutil.copy2(artifacts.model_path, active_dir / "model.pt")
        if artifacts.preprocessor_path is not None:
            shutil.copy2(artifacts.preprocessor_path, active_dir / "preprocessor.joblib")

    return active_dir
