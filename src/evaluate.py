"""Metrics and diagnostics, computed on the original (unscaled) OT scale."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def skill_vs_persistence(model_mae: float, persistence_mae: float) -> float:
    """Percent MAE improvement over persistence; positive = better than persistence."""
    if persistence_mae == 0:
        return float("nan")
    return float(100.0 * (persistence_mae - model_mae) / persistence_mae)


def warm_threshold_from_train(train_target: np.ndarray, quantile: float) -> float:
    """Train-only quantile defining the 'warm / high-OT' evaluation subset.

    This is purely a descriptive percentile of observed oil temperature in
    the training period - it is NOT an engineering safety threshold, failure
    criterion, or maintenance trigger, and must not be presented as one.
    """
    return float(np.quantile(train_target, quantile))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    warm_threshold: float | None = None,
) -> dict:
    out = {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
    if warm_threshold is not None:
        warm_mask = y_true > warm_threshold
        out["warm_subset"] = {
            "threshold": warm_threshold,
            "n_warm": int(warm_mask.sum()),
            "mae": mae(y_true[warm_mask], y_pred[warm_mask]) if warm_mask.sum() > 0 else None,
            "rmse": rmse(y_true[warm_mask], y_pred[warm_mask]) if warm_mask.sum() > 0 else None,
        }
    residuals = y_pred - y_true
    by_hour = (
        pd.DataFrame({"hour": timestamps.hour, "residual": residuals})
        .groupby("hour")["residual"]
        .agg(["mean", "std", "count"])
    )
    out["residual_by_hour_of_day"] = {int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in by_hour.to_dict(orient="index").items()}
    return out


def worst_examples(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex, k: int = 10, horizon: int | None = None
) -> list[dict]:
    """Representative failing examples: the k largest-absolute-error rows.

    `timestamps` are origins for forecasting (pass `horizon` to also report
    the target time) and the shared origin/target time for nowcasting.
    """
    abs_err = np.abs(y_pred - y_true)
    order = np.argsort(-abs_err)[:k]
    out = []
    for i in order:
        row = {"y_true": float(y_true[i]), "y_pred": float(y_pred[i]), "abs_error": float(abs_err[i])}
        if horizon is None:
            row = {"timestamp": str(timestamps[i]), **row}
        else:
            row = {"origin_timestamp": str(timestamps[i]), "target_timestamp": str(timestamps[i] + pd.Timedelta(hours=horizon)), **row}
        out.append(row)
    return out
