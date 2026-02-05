from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd


def run_pandas_script(df: pd.DataFrame, script: str) -> pd.DataFrame:
    local_vars: dict[str, Any] = {"df": df.copy(), "pd": pd, "np": np}
    script = textwrap.dedent(script)
    exec(script, {}, local_vars)
    if "df" not in local_vars:
        raise ValueError("Script must define 'df'.")
    if not isinstance(local_vars["df"], pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")
    return local_vars["df"]


def parse_list(value: str, cast):
    if not value:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in items]
