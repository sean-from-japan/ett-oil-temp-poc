"""Feature-construction correctness, verified by independent manual
recomputation rather than by re-calling the implementation under test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_forecast_features, calendar_features, causal_lag_features, causal_rolling_features


def test_lag_feature_matches_manual_indexing(synthetic_df):
    lag = 5
    feats = causal_lag_features(synthetic_df["OT"], [lag], prefix="OT")
    col = feats[f"OT_lag{lag}"].to_numpy()
    ot = synthetic_df["OT"].to_numpy()
    for i in [0, 1, lag - 1, lag, lag + 1, 100, len(ot) - 1]:
        expected = ot[i - lag] if i - lag >= 0 else np.nan
        got = col[i]
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == expected, f"row {i}: expected {expected}, got {got}"


def test_rolling_feature_matches_manual_window_mean(synthetic_df):
    window = 6
    feats = causal_rolling_features(synthetic_df["OT"], [window], prefix="OT")
    col = feats[f"OT_rollmean{window}"].to_numpy()
    ot = synthetic_df["OT"].to_numpy()
    # The window at origin row i covers the `window` observations ending at i.
    for i in [window - 1, window, 50, 500, len(ot) - 1]:
        manual_mean = np.mean(ot[i - window + 1 : i + 1])
        assert np.isclose(col[i], manual_mean), f"row {i}: expected {manual_mean}, got {col[i]}"
    # Rows before a full window is available must be NaN (not silently
    # computed from a short/partial window).
    for i in range(window - 1):
        assert np.isnan(col[i])


def test_origin_observation_is_used_but_next_hour_is_not(synthetic_df):
    """Exact boundary: features at origin t must change when OT(t) changes
    (the origin value is legitimately available) and must NOT change when
    only OT(t+1) changes."""
    horizon = 1
    t = 900
    X0, _, _ = build_forecast_features(synthetic_df, horizon)

    bumped_origin = synthetic_df.copy()
    bumped_origin.iloc[t, bumped_origin.columns.get_loc("OT")] += 50.0
    X1, _, _ = build_forecast_features(bumped_origin, horizon)
    assert X1.iloc[t]["OT_lag0"] == X0.iloc[t]["OT_lag0"] + 50.0

    bumped_next = synthetic_df.copy()
    bumped_next.iloc[t + 1, bumped_next.columns.get_loc("OT")] += 50.0
    X2, y2, _ = build_forecast_features(bumped_next, horizon)
    pd.testing.assert_series_equal(X2.iloc[t], X0.iloc[t])
    # ...while the label for origin t (OT(t+1)) is exactly what changed.
    assert np.isclose(y2.iloc[t], synthetic_df["OT"].iloc[t + 1] + 50.0)


def test_calendar_features_are_bounded_and_deterministic(synthetic_df):
    feats = calendar_features(synthetic_df.index)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert feats[c].between(-1.0001, 1.0001).all()
    expected_weekend = (synthetic_df.index.day_of_week >= 5).astype(float)
    assert np.allclose(feats["is_weekend"].to_numpy(), np.asarray(expected_weekend))


def test_forecast_target_is_the_only_future_shift(synthetic_df):
    horizon = 24
    X, y, valid_mask = build_forecast_features(synthetic_df, horizon)
    ot = synthetic_df["OT"].to_numpy()
    y_arr = y.to_numpy()
    # target at row i must equal OT at row i+horizon (or NaN past the end)
    for i in [0, 100, len(ot) - horizon - 1]:
        assert np.isclose(y_arr[i], ot[i + horizon])
    for i in range(len(ot) - horizon, len(ot)):
        assert np.isnan(y_arr[i])


def test_availability_mask_flags_burn_in_and_tail(synthetic_df):
    horizon = 24
    X, y, valid_mask = build_forecast_features(synthetic_df, horizon)
    max_lag = max(int(c.split("lag")[1]) for c in X.columns if "lag" in c)
    # The very first rows cannot have a full max-lag/rolling history available.
    assert not valid_mask[: max_lag].any()
    assert valid_mask[max_lag]
    # The last `horizon` rows have no observed target yet.
    assert valid_mask[-1] == False  # noqa: E712
    # Comfortably past burn-in, with target ahead of the series end, should be valid.
    mid = len(synthetic_df) // 2
    assert valid_mask[mid] == True  # noqa: E712
