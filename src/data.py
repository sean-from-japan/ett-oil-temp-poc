"""Loading and exploratory data analysis for the ETT datasets.

All functions here are read-only with respect to modeling: they describe
the data (frequency, missingness, duplicates, drift, autocorrelation,
seasonality, load/OT relationships) but do not build model features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, LOAD_COLUMNS, TARGET_COLUMN

EXPECTED_FREQ = {
    "ETTh1": pd.Timedelta(hours=1),
    "ETTh2": pd.Timedelta(hours=1),
    "ETTm1": pd.Timedelta(minutes=15),
    "ETTm2": pd.Timedelta(minutes=15),
}


def load_raw(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one dataset CSV, parse timestamps, sort, and set as index.

    Does not impute, scale, or otherwise transform values - this is the
    raw series as released, just type-cast and indexed by time.
    """
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download` first (official source only)."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.name = "timestamp"
    return df


def basic_integrity_report(df: pd.DataFrame, name: str) -> dict:
    """Timestamp/frequency sanity checks + missing value / duplicate audit."""
    idx = df.index
    diffs = idx.to_series().diff().dropna()
    expected = EXPECTED_FREQ.get(name)
    irregular_mask = diffs != expected if expected is not None else pd.Series(dtype=bool)
    report = {
        "name": name,
        "n_rows": int(len(df)),
        "start": str(idx.min()),
        "end": str(idx.max()),
        "expected_step": str(expected) if expected is not None else None,
        "n_duplicate_timestamps": int(idx.duplicated().sum()),
        "n_irregular_steps": int(irregular_mask.sum()) if expected is not None else None,
        "n_missing_values_total": int(df.isnull().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isnull().sum().items()},
        "n_missing_grid_steps": None,
    }
    if expected is not None and len(diffs) > 0:
        full_range = pd.date_range(idx.min(), idx.max(), freq=expected)
        report["n_missing_grid_steps"] = int(len(full_range) - len(idx))
    return report


def describe_columns(df: pd.DataFrame) -> dict:
    return json.loads(df.describe().to_json())


def monthly_drift(df: pd.DataFrame, column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Month-by-month mean/std to visualize distribution drift over time."""
    g = df[column].groupby(df.index.to_period("M"))
    out = g.agg(["mean", "std", "min", "max"])
    out.index = out.index.astype(str)
    return out


def autocorrelation(series: pd.Series, max_lag: int) -> pd.Series:
    """Simple ACF computed directly (no third-party ACF dependency)."""
    s = series.dropna()
    n = len(s)
    mean = s.mean()
    var = ((s - mean) ** 2).sum()
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
            continue
        cov = ((s.iloc[: n - lag].values - mean) * (s.iloc[lag:].values - mean)).sum()
        acf_vals.append(cov / var if var > 0 else np.nan)
    return pd.Series(acf_vals, index=range(max_lag + 1), name=f"acf_{series.name}")


def load_ot_correlation(df: pd.DataFrame) -> pd.Series:
    """Contemporaneous Pearson correlation of each load column with OT.

    This is purely descriptive EDA (same-time correlation structure); it is
    NOT the nowcasting model and must not be read as a forecast skill number.
    """
    return df[LOAD_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)


def metadata_only_summary(name: str, data_dir: Path = DATA_DIR) -> dict:
    """Shape/frequency/missingness summary for datasets we do NOT model.

    ETTm1/ETTm2 are inspected at metadata level only;
    no features or models are built from them.
    """
    df = load_raw(name, data_dir=data_dir)
    report = basic_integrity_report(df, name)
    report["columns"] = list(df.columns)
    report["note"] = "metadata-only inspection; not used for modeling in this benchmark"
    return report
