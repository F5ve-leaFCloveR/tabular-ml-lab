from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from tabular_ml_lab.config import load_config
from tabular_ml_lab.eda import run_eda


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    path = run_eda(cfg)
    print("EDA report saved:", path)
