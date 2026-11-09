"""Explicit target/feature separation checks between the two differently
labeled tasks: true future forecasting vs. contemporaneous nowcasting.
"""
from __future__ import annotations

import numpy as np

from src.config import LOAD_COLUMNS, TARGET_COLUMN
from src.features import build_forecast_features, build_nowcast_features


def test_forecast_feature_columns_never_use_raw_current_or_future_load_names(synthetic_df):
    """Forecast features must only contain *_lag*/_roll*/calendar columns -
    never a bare current-timestamp load/OT column (which would implicitly
    mean "available at t" being confused with "available for t as if it
    were t+h", a common nowcasting/forecasting mix-up)."""
    X, y, mask = build_forecast_features(synthetic_df, horizon=24)
    for col in LOAD_COLUMNS + [TARGET_COLUMN]:
        assert col not in X.columns
    for col in X.columns:
        assert ("_lag" in col) or ("_roll" in col) or col in {
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"
        }


def test_nowcast_feature_columns_are_contemporaneous_load_only(synthetic_df):
    X, y, mask = build_nowcast_features(synthetic_df)
    assert set(X.columns) == {f"{c}_now" for c in LOAD_COLUMNS}
    assert TARGET_COLUMN not in X.columns


def test_forecast_and_nowcast_targets_differ_in_definition(synthetic_df):
    horizon = 24
    _, y_forecast, mask_f = build_forecast_features(synthetic_df, horizon)
    _, y_now, mask_n = build_nowcast_features(synthetic_df)
    # forecast target at row i is OT at i+horizon; nowcast target at row i is OT at i.
    ot = synthetic_df[TARGET_COLUMN].to_numpy()
    i = 500
    assert np.isclose(y_now.to_numpy()[i], ot[i])
    assert np.isclose(y_forecast.to_numpy()[i], ot[i + horizon])
    assert y_now.to_numpy()[i] != y_forecast.to_numpy()[i]
