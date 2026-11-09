"""Baseline model correctness, including the seasonal-naive validity guard
(every reference value must exist at the origin)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import warm_threshold_from_train
from src.models import persistence_predict, seasonal_naive_predict


def test_persistence_equals_origin_value(synthetic_df):
    pred, _ = persistence_predict(synthetic_df, horizon=24)
    assert np.allclose(pred.to_numpy(), synthetic_df["OT"].to_numpy())


def test_seasonal_naive_rejects_period_shorter_than_horizon(synthetic_df):
    with pytest.raises(ValueError):
        seasonal_naive_predict(synthetic_df, horizon=96, period=24)


def test_seasonal_naive_daily_valid_for_horizon_equal_to_period(synthetic_df):
    pred, _ = seasonal_naive_predict(synthetic_df, horizon=24, period=24)
    # period == horizon means the reference timestamp IS the origin itself.
    assert np.allclose(pred.to_numpy(), synthetic_df["OT"].to_numpy())


def test_seasonal_naive_weekly_references_only_past_values(synthetic_df):
    horizon = 96
    period = 168
    pred, _ = seasonal_naive_predict(synthetic_df, horizon, period)
    ot = synthetic_df["OT"].to_numpy()
    i = 300
    # reference position = i + horizon - period = i - 72, strictly before i.
    expected = ot[i + horizon - period]
    assert np.isclose(pred.to_numpy()[i], expected)


def test_warm_threshold_uses_only_train_quantile_not_full_series(synthetic_df):
    train_target = synthetic_df["OT"].to_numpy()[:1000]
    full_target = synthetic_df["OT"].to_numpy()
    t_train = warm_threshold_from_train(train_target, 0.75)
    t_full = warm_threshold_from_train(full_target, 0.75)
    assert np.isclose(t_train, np.quantile(train_target, 0.75))
    # The synthetic series has a deliberate upward trend, so the train-only
    # quantile must differ from the full-series quantile - demonstrating the
    # function is sensitive to *which* slice is passed in, not silently
    # falling back to the whole series.
    assert not np.isclose(t_train, t_full)
