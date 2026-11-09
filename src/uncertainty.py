"""Uncertainty of test-set comparisons, computed from the frozen prediction
exports only (reporting, never used for model choice).

Hourly errors are strongly autocorrelated, so an i.i.d. bootstrap would
overstate precision. We use a moving-block bootstrap with 1-week blocks on
the paired per-sample difference |e_model| - |e_persistence|, and also
report per-calendar-month MAE to show how unstable a single 4-month test
window is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED

BLOCK_HOURS = 168
N_BOOT = 1000


def block_bootstrap_mean_ci(values: np.ndarray, block: int = BLOCK_HOURS, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_predictions(pred_dir=None, out_dir=None) -> pd.DataFrame:
    pred_dir = pred_dir or RESULTS_DIR / "predictions"
    out_dir = out_dir or RESULTS_DIR
    rows, monthly = [], []
    for path in sorted(pred_dir.glob("*_h*.csv")):
        dataset, h = path.stem.split("_h")
        df = pd.read_csv(path, parse_dates=["origin_timestamp", "target_timestamp"])
        base = np.abs(df["pred_persistence"] - df["y_true"]).to_numpy()
        for col in [c for c in df.columns if c.startswith("pred_")]:
            err = np.abs(df[col] - df["y_true"]).to_numpy()
            diff = err - base
            lo, hi = block_bootstrap_mean_ci(diff) if col != "pred_persistence" else (0.0, 0.0)
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": int(h),
                    "model": col.removeprefix("pred_"),
                    "test_mae": float(err.mean()),
                    "mae_minus_persistence": float(diff.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )
            m = pd.DataFrame({"month": df["target_timestamp"].dt.to_period("M").astype(str), "err": err})
            for month, g in m.groupby("month"):
                monthly.append({"dataset": dataset, "horizon": int(h), "model": col.removeprefix("pred_"), "target_month": month, "n": len(g), "mae": float(g["err"].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "test_uncertainty_block_bootstrap.csv", index=False, float_format="%.6f")
    pd.DataFrame(monthly).to_csv(out_dir / "test_mae_by_month.csv", index=False, float_format="%.6f")
    return out
