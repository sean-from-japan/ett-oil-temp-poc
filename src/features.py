"""Feature engineering for the forecasting and nowcasting tasks.

Strict rule enforced throughout this module: every feature for an origin
timestamp t uses only observations with timestamp <= t. The observation at
the origin itself (lag 0) is included, because OT(t) and load(t) are
available by t - the same information persistence uses. Rolling windows are
trailing and end at t inclusive; lags use non-negative `.shift(lag)` (never a
negative/future shift).

One documented exception: the assignment states that when predicting OT at
time T, features observed at T may be used. With `target_load_known=True`
the six load readings at the target time T = t+h (and calendar features of
T) are added. OT after t is never used in either scenario. Apart from that
option, the only negative shift is in `forecast_target`, which builds the
*label* OT(t+h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LAG_HOURS, LOAD_COLUMNS, ROLLING_WINDOWS, TARGET_COLUMN


def causal_lag_features(series: pd.Series, lags: list[int], prefix: str) -> pd.DataFrame:
    out = {}
    for lag in lags:
        if lag < 0:
            raise ValueError("Negative lags would read the future; lags must be >= 0.")
        out[f"{prefix}_lag{lag}"] = series.shift(lag)
    return pd.DataFrame(out, index=series.index)


def causal_rolling_features(series: pd.Series, windows: list[int], prefix: str) -> pd.DataFrame:
    """Trailing rolling mean/std over the `w` observations t-w+1 .. t."""
    out = {}
    for w in windows:
        if w <= 0:
            raise ValueError("Rolling windows must be positive.")
        roll = series.rolling(window=w, min_periods=w)
        out[f"{prefix}_rollmean{w}"] = roll.mean()
        out[f"{prefix}_rollstd{w}"] = roll.std()
    return pd.DataFrame(out, index=series.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features of the origin timestamp t.

    These are known arbitrarily far in advance (a clock/calendar function),
    so using them carries no leakage risk regardless of horizon.
    """
    hour = index.hour.values.astype(float)
    dow = index.day_of_week.values.astype(float)
    month = index.month.values.astype(float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_weekend": (dow >= 5).astype(float),
        },
        index=index,
    )


def forecast_target(df: pd.DataFrame, horizon: int, target_column: str = TARGET_COLUMN) -> pd.Series:
    """Label OT(t+h) aligned at row t. This is the ONLY future-looking shift
    in the codebase, and it produces the label, never a feature column."""
    return df[target_column].shift(-horizon).rename(f"target_h{horizon}")


def build_forecast_features(
    df: pd.DataFrame,
    horizon: int,
    lag_hours: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    target_column: str = TARGET_COLUMN,
    load_columns: list[str] | None = None,
    target_load_known: bool = False,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build the feature matrix X, label y, and a validity mask for the
    forecasting task: predict OT(t+h) from information available by t.

    target_load_known=True adds the load readings at the target time t+h,
    their change since t, and calendar features of t+h (scenario
    "load at T known", allowed by the assignment). OT stays origin-only.
    """
    lag_hours = lag_hours or LAG_HOURS
    rolling_windows = rolling_windows or ROLLING_WINDOWS
    load_columns = load_columns if load_columns is not None else LOAD_COLUMNS

    blocks = [
        causal_lag_features(df[target_column], lag_hours, prefix="OT"),
        causal_rolling_features(df[target_column], rolling_windows, prefix="OT"),
        calendar_features(df.index),
    ]
    for col in load_columns:
        blocks.append(causal_lag_features(df[col], [0, 1, 24, 168], prefix=col))
        blocks.append(causal_rolling_features(df[col], rolling_windows, prefix=col))

    if target_load_known:
        at_target = {}
        for col in load_columns:
            at_target[f"{col}_atT"] = df[col].shift(-horizon)
            at_target[f"{col}_deltaT"] = df[col].shift(-horizon) - df[col]
        blocks.append(pd.DataFrame(at_target, index=df.index))
        cal_t = calendar_features(df.index + pd.Timedelta(hours=horizon))
        cal_t.index = df.index
        blocks.append(cal_t.add_suffix("_atT"))

    X = pd.concat(blocks, axis=1)
    y = forecast_target(df, horizon, target_column=target_column)

    valid_features = ~X.isnull().any(axis=1)
    valid_target = ~y.isnull()
    valid_mask = (valid_features & valid_target).to_numpy()
    return X, y, valid_mask


def build_nowcast_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    load_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Contemporaneous regression: predict OT(t) from non-OT load readings
    at the SAME timestamp t. This is a different, easier information set
    than true forecasting (future loads would be unknown at deployment
    time) and must be reported separately, never as an apples-to-apples
    comparison with the forecasting task's operational performance.
    """
    load_columns = load_columns if load_columns is not None else LOAD_COLUMNS
    X = df[load_columns].copy()
    X.columns = [f"{c}_now" for c in X.columns]
    y = df[target_column].rename("target_now")
    assert target_column not in X.columns and f"{target_column}_now" not in X.columns
    valid_mask = (~X.isnull().any(axis=1) & ~y.isnull()).to_numpy()
    return X, y, valid_mask
