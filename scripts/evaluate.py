from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from tabular_ml_lab.config import load_config
from tabular_ml_lab.eval import evaluate


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    metrics = evaluate(cfg)
    print("Test metrics:", metrics)
