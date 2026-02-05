from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tabular_ml_lab.config import Config
from tabular_ml_lab.data import load_dataset
from tabular_ml_lab.utils import ensure_dir


def run_eda(cfg: Config) -> Path:
    train_df, val_df, test_df = load_dataset(cfg)
    reports_dir = ensure_dir(cfg.paths.reports_dir)

    combined = pd.concat([train_df, val_df], ignore_index=True)
    target_col = cfg.dataset.target

    missing = combined.isna().mean().sort_values(ascending=False)
    summary = combined.describe(include="all").transpose()
    target_dist = combined[target_col].value_counts(normalize=True)

    plt.figure()
    target_dist.plot(kind="bar", color=["#4C78A8", "#F58518"])
    plt.title("Target Distribution")
    plt.ylabel("Share")
    plt.tight_layout()
    plot_path = reports_dir / "target_distribution.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    eda_path = reports_dir / "eda.md"
    eda_path.write_text(
        "\n".join(
            [
                "# EDA Summary",
                "",
                "## Dataset sizes",
                f"- Train: {len(train_df)}",
                f"- Validation: {len(val_df)}",
                f"- Test: {len(test_df)}",
                "",
                "## Missing value ratio (top 10)",
                missing.head(10).to_string(),
                "",
                "## Target distribution",
                target_dist.to_string(),
                "",
                "## Numeric summary (head)",
                summary.head(10).to_string(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return eda_path
