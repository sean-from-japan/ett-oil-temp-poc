"""Baselines and models for the bounded forecasting task.

All models here operate on a feature matrix / target pair already built by
`src.features` (strictly causal). This module adds no new information
sources; it only fits/predicts and reports timing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN
from src.splits import expanding_window_cv_splits


@dataclass
class FitResult:
    name: str
    y_pred: np.ndarray
    fit_time_s: float
    predict_time_s: float
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Baselines (no training; deterministic transforms of past observations)
# ---------------------------------------------------------------------------

def persistence_predict(df: pd.DataFrame, horizon: int, target_column: str = TARGET_COLUMN) -> pd.Series:
    """y_hat(t+h) = OT(t). Always valid: OT(t) is observed at the origin."""
    t0 = time.perf_counter()
    pred = df[target_column].copy()
    dt = time.perf_counter() - t0
    pred.name = "persistence"
    return pred, dt


def seasonal_naive_predict(
    df: pd.DataFrame, horizon: int, period: int, target_column: str = TARGET_COLUMN
) -> tuple[pd.Series, float]:
    """y_hat(t+h) = OT(t+h-period), requires period >= horizon so the
    reference timestamp t+h-period <= t is actually observed at the origin.

    Raises if period < horizon rather than silently returning a forecast
    that would secretly depend on an unobserved future value.
    """
    if period < horizon:
        raise ValueError(
            f"seasonal_naive with period={period} is invalid for horizon={horizon}: "
            f"the reference timestamp would be {horizon - period}h in the future of the origin."
        )
    t0 = time.perf_counter()
    pred = df[target_column].shift(period - horizon)
    dt = time.perf_counter() - t0
    pred.name = f"seasonal_naive_p{period}"
    return pred, dt


# ---------------------------------------------------------------------------
# Hyperparameter search (train-only expanding-window CV)
# ---------------------------------------------------------------------------

def _cv_score_ridge(X: np.ndarray, y: np.ndarray, folds: list, alpha: float, valid_mask: np.ndarray) -> float:
    maes = []
    for train_mask, val_mask in folds:
        train_mask = train_mask & valid_mask
        val_mask = val_mask & valid_mask
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xva = scaler.transform(X[val_mask])
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(Xtr, y[train_mask])
        pred = model.predict(Xva)
        maes.append(np.mean(np.abs(pred - y[val_mask])))
    return float(np.mean(maes))


def _cv_score_rf(X: np.ndarray, y: np.ndarray, folds: list, max_depth, n_estimators: int, seed: int, valid_mask: np.ndarray) -> float:
    maes = []
    for train_mask, val_mask in folds:
        train_mask = train_mask & valid_mask
        val_mask = val_mask & valid_mask
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[val_mask])
        maes.append(np.mean(np.abs(pred - y[val_mask])))
    return float(np.mean(maes))


def select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    train_start_idx: int,
    train_end_idx: int,
    horizon: int,
    n_splits: int,
    min_train_hours: int,
    alpha_grid: list[float] | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict:
    alpha_grid = alpha_grid or [0.1, 1.0, 10.0, 100.0]
    valid_mask = valid_mask if valid_mask is not None else np.ones(len(X), dtype=bool)
    folds = expanding_window_cv_splits(
        train_start_idx, train_end_idx, horizon, n_splits, min_train_hours, n_rows=len(X)
    )
    scores = {a: _cv_score_ridge(X, y, folds, a, valid_mask) for a in alpha_grid}
    best = min(scores, key=scores.get)
    return {"best_alpha": best, "cv_mae_by_alpha": scores, "n_folds": len(folds)}


def select_rf_params(
    X: np.ndarray,
    y: np.ndarray,
    train_start_idx: int,
    train_end_idx: int,
    horizon: int,
    n_splits: int,
    min_train_hours: int,
    seed: int,
    param_grid: list[dict] | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict:
    param_grid = param_grid or [
        {"max_depth": 5, "n_estimators": 100},
        {"max_depth": 8, "n_estimators": 100},
        {"max_depth": None, "n_estimators": 100},
    ]
    valid_mask = valid_mask if valid_mask is not None else np.ones(len(X), dtype=bool)
    folds = expanding_window_cv_splits(
        train_start_idx, train_end_idx, horizon, n_splits, min_train_hours, n_rows=len(X)
    )
    scores = {}
    for i, params in enumerate(param_grid):
        scores[i] = _cv_score_rf(X, y, folds, params["max_depth"], params["n_estimators"], seed, valid_mask)
    best_i = min(scores, key=scores.get)
    return {
        "best_params": param_grid[best_i],
        "cv_mae_by_candidate": {str(param_grid[i]): v for i, v in scores.items()},
        "n_folds": len(folds),
    }


# ---------------------------------------------------------------------------
# Fit/predict wrappers used by the pipeline
# ---------------------------------------------------------------------------

def fit_predict_ridge(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, alpha: float, seed: int
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(Xtr, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    Xev = scaler.transform(X_eval)
    pred = model.predict(Xev)
    predict_time = time.perf_counter() - t0
    return pred, fit_time, predict_time, model, scaler


def fit_predict_rf(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, max_depth, n_estimators: int, seed: int
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_eval)
    predict_time = time.perf_counter() - t0
    return pred, fit_time, predict_time, model
