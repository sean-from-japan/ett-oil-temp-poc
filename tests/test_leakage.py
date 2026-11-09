"""Leakage detection: future-target/feature separation, and a perturbation
check that future observations cannot influence earlier-origin features or
predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_forecast_features, build_nowcast_features
from src.models import fit_predict_ridge


def test_nowcast_features_never_equal_target_column(synthetic_df):
    X, y, valid_mask = build_nowcast_features(synthetic_df)
    assert "OT" not in X.columns
    y_arr = y.to_numpy()
    for col in X.columns:
        assert not np.allclose(X[col].to_numpy(), y_arr), f"{col} is suspiciously identical to the OT target"


def test_nowcast_origin_equals_target_timestamp(synthetic_df):
    # Nowcasting has no horizon: target OT(t) is read at the same row as the
    # features X(t), never shifted.
    X, y, valid_mask = build_nowcast_features(synthetic_df)
    assert np.allclose(y.to_numpy(), synthetic_df["OT"].to_numpy())


def test_forecast_features_exclude_future_rows_by_perturbation(synthetic_df):
    """Perturb all observations strictly after a cutoff timestamp, rebuild
    features, and verify every earlier-origin feature row is byte-identical.
    This is the concrete "detect using values after the forecast origin"
    check.
    """
    horizon = 24
    cutoff_pos = 1000
    cutoff_ts = synthetic_df.index[cutoff_pos]

    X_orig, y_orig, mask_orig = build_forecast_features(synthetic_df, horizon)

    perturbed = synthetic_df.copy()
    future_rows = perturbed.index > cutoff_ts
    rng = np.random.default_rng(0)
    for col in perturbed.columns:
        perturbed.loc[future_rows, col] = rng.normal(1e6, 1e3, int(future_rows.sum()))

    X_pert, y_pert, mask_pert = build_forecast_features(perturbed, horizon)

    earlier_origin = synthetic_df.index <= cutoff_ts
    # Features for origins at/before the cutoff must be unaffected by
    # perturbing anything after the cutoff.
    pd.testing.assert_frame_equal(X_orig.loc[earlier_origin], X_pert.loc[earlier_origin])

    # A deterministic model trained on the (unperturbed) data would
    # therefore also predict identically for these origins.
    Xv_orig = X_orig.to_numpy()
    Xv_pert = X_pert.to_numpy()
    valid_and_early = mask_orig & np.asarray(earlier_origin)
    assert valid_and_early.sum() > 50
    pred_orig, _, _, _, _ = fit_predict_ridge(
        Xv_orig[mask_orig], y_orig.to_numpy()[mask_orig], Xv_orig[valid_and_early], alpha=1.0, seed=0
    )
    pred_pert, _, _, _, _ = fit_predict_ridge(
        Xv_orig[mask_orig], y_orig.to_numpy()[mask_orig], Xv_pert[valid_and_early], alpha=1.0, seed=0
    )
    assert np.allclose(pred_orig, pred_pert)


def test_perturbation_check_catches_an_injected_lookahead_bug(synthetic_df):
    """Sanity-check the detector itself: deliberately build a *leaky*
    feature (today's feature peeking 1 hour into the future) and confirm the
    same perturb-and-compare method used above correctly flags it. This
    demonstrates the test suite would catch a real lookahead bug, not just
    pass vacuously on correct code.
    """
    horizon = 24
    cutoff_pos = 1000
    cutoff_ts = synthetic_df.index[cutoff_pos]

    def build_leaky(df):
        X, y, mask = build_forecast_features(df, horizon)
        X = X.copy()
        X["LEAKY_future_OT"] = df["OT"].shift(-1)  # uses t+1: a future value
        return X, y, mask & ~X["LEAKY_future_OT"].isnull().to_numpy()

    X_orig, _, _ = build_leaky(synthetic_df)

    perturbed = synthetic_df.copy()
    future_rows = perturbed.index > cutoff_ts
    perturbed.loc[future_rows, "OT"] = 1e6
    X_pert, _, _ = build_leaky(perturbed)

    earlier_origin = synthetic_df.index <= cutoff_ts
    # The row immediately at the cutoff reads OT at cutoff+1h, which was
    # perturbed, so it MUST differ - proving the check has detection power.
    row = synthetic_df.index.get_loc(cutoff_ts)
    assert X_orig["LEAKY_future_OT"].iloc[row] != X_pert["LEAKY_future_OT"].iloc[row]
    with np.testing.assert_raises(AssertionError):
        pd.testing.assert_frame_equal(X_orig.loc[earlier_origin], X_pert.loc[earlier_origin])


def test_load_at_T_scenario_uses_load_at_target_but_never_future_ot(synthetic_df):
    """Scenario allowed by the assignment: loads at the target time T=t+h may
    be features. Perturbing OT after the origin must leave every feature row
    unchanged, while perturbing a load at T must change that origin's row."""
    horizon = 24
    cutoff_pos = 1000
    cutoff_ts = synthetic_df.index[cutoff_pos]
    X_orig, _, _ = build_forecast_features(synthetic_df, horizon, target_load_known=True)

    pert_ot = synthetic_df.copy()
    pert_ot.loc[pert_ot.index > cutoff_ts, "OT"] = 1e6
    X_ot, _, _ = build_forecast_features(pert_ot, horizon, target_load_known=True)
    earlier_origin = synthetic_df.index <= cutoff_ts
    pd.testing.assert_frame_equal(X_orig.loc[earlier_origin], X_ot.loc[earlier_origin])

    pert_load = synthetic_df.copy()
    target_ts = synthetic_df.index[cutoff_pos + horizon]
    pert_load.loc[target_ts, "HUFL"] = 1e6
    X_load, _, _ = build_forecast_features(pert_load, horizon, target_load_known=True)
    assert X_load.loc[cutoff_ts, "HUFL_atT"] == 1e6
    # Loads strictly between the origin and T are not used.
    pert_mid = synthetic_df.copy()
    pert_mid.loc[synthetic_df.index[cutoff_pos + 1 : cutoff_pos + horizon], "HUFL"] = 1e6
    X_mid, _, _ = build_forecast_features(pert_mid, horizon, target_load_known=True)
    pd.testing.assert_series_equal(X_orig.loc[cutoff_ts], X_mid.loc[cutoff_ts])
