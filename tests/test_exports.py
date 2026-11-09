"""Checks on the frozen prediction exports written by the full run: every
row's y_true must be the raw OT at its target timestamp, origin/target must
differ by exactly the horizon, and the export must lie inside the test split.
Skipped when the full run has not been executed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_DIR, RESULTS_DIR
from src.splits import build_split_ranges

PRED_DIR = RESULTS_DIR / "predictions"
CASES = [(d, h) for d in ("ETTh1", "ETTh2") for h in (1, 24, 96)]


@pytest.mark.parametrize("dataset,horizon", CASES)
def test_forecast_export_is_timestamp_aligned_with_raw_data(dataset, horizon):
    path = PRED_DIR / f"{dataset}_h{horizon}.csv"
    raw_path = DATA_DIR / f"{dataset}.csv"
    if not path.exists() or not raw_path.exists():
        pytest.skip("full run outputs or raw data not present")
    preds = pd.read_csv(path, parse_dates=["origin_timestamp", "target_timestamp"])
    raw = pd.read_csv(raw_path, parse_dates=["date"]).set_index("date")["OT"]

    assert (preds["target_timestamp"] - preds["origin_timestamp"] == pd.Timedelta(hours=horizon)).all()
    assert np.allclose(preds["y_true"].to_numpy(), raw.loc[preds["target_timestamp"]].to_numpy(), atol=1e-5)
    assert np.allclose(preds["pred_persistence"].to_numpy(), raw.loc[preds["origin_timestamp"]].to_numpy(), atol=1e-5)

    test = build_split_ranges()["test"]
    assert preds["origin_timestamp"].min() >= test.start_ts
    assert preds["target_timestamp"].max() < test.end_ts
    for col in [c for c in preds.columns if c.startswith("pred_")]:
        assert np.isfinite(preds[col]).all(), col


@pytest.mark.parametrize("dataset", ["ETTh1", "ETTh2"])
def test_nowcast_export_target_is_same_timestamp_ot(dataset):
    path = PRED_DIR / f"{dataset}_nowcast.csv"
    raw_path = DATA_DIR / f"{dataset}.csv"
    if not path.exists() or not raw_path.exists():
        pytest.skip("full run outputs or raw data not present")
    preds = pd.read_csv(path, parse_dates=["timestamp"])
    raw = pd.read_csv(raw_path, parse_dates=["date"]).set_index("date")["OT"]
    assert np.allclose(preds["y_true"].to_numpy(), raw.loc[preds["timestamp"]].to_numpy(), atol=1e-5)
