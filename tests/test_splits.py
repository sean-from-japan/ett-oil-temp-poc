"""Split-boundary, purge, and expanding-window CV correctness."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_DIR
from src.splits import (
    HOURS_PER_MONTH,
    assert_regular_hourly,
    build_split_ranges,
    expanding_window_cv_splits,
    get_forecast_split_masks,
    origin_mask_for_split,
)


def test_30day_month_boundaries_are_contiguous_and_correctly_sized():
    ranges = build_split_ranges()
    assert ranges["train"].start_idx == 0
    assert ranges["train"].end_idx == 12 * HOURS_PER_MONTH
    assert ranges["val"].start_idx == ranges["train"].end_idx
    assert ranges["val"].end_idx == ranges["val"].start_idx + 4 * HOURS_PER_MONTH
    assert ranges["test"].start_idx == ranges["val"].end_idx
    assert ranges["test"].end_idx == ranges["test"].start_idx + 4 * HOURS_PER_MONTH
    # exactly 20 (30-day) months total
    assert ranges["test"].end_idx == 20 * HOURS_PER_MONTH


def test_origin_mask_purges_targets_crossing_the_split_boundary():
    from src.splits import SplitRange

    split = SplitRange(name="x", start_idx=0, end_idx=50, start_ts=pd.Timestamp("2020-01-01"), end_ts=pd.Timestamp("2020-01-01") + pd.Timedelta(hours=50))
    horizon = 10
    mask = origin_mask_for_split(n_rows=100, split=split, horizon=horizon)
    valid_positions = np.nonzero(mask)[0]
    # Only origins 0..39 have a target (origin+10) still inside [0,50).
    assert valid_positions.max() == 39
    assert valid_positions.min() == 0
    assert mask.sum() == 40


def test_forecast_split_masks_are_mutually_disjoint_and_never_touch_unused_tail():
    n_rows = 17420  # ETTh1 length
    for horizon in [1, 24, 96]:
        masks = get_forecast_split_masks(n_rows, horizon)
        train, val, test = masks["train"], masks["val"], masks["test"]
        assert not np.any(train & val)
        assert not np.any(val & test)
        assert not np.any(train & test)
        # Nothing beyond the 20-month (14400h) boundary is ever selected -
        # the remaining ~3.8 months of data are not silently mixed in.
        tail = np.zeros(n_rows, dtype=bool)
        tail[14400:] = True
        assert not np.any((train | val | test) & tail)


def test_expanding_window_cv_folds_leave_a_horizon_sized_gap():
    horizon = 24
    folds = expanding_window_cv_splits(
        train_start_idx=0, train_end_idx=8640, horizon=horizon, n_splits=4, min_train_hours=2880, n_rows=8640
    )
    assert len(folds) > 0
    for train_mask, val_mask in folds:
        train_positions = np.nonzero(train_mask)[0]
        val_positions = np.nonzero(val_mask)[0]
        assert train_positions.max() < val_positions.min()
        gap = val_positions.min() - train_positions.max() - 1
        assert gap >= horizon - 1, f"gap={gap} smaller than horizon-1={horizon - 1}"
        # expanding: every fold's training set starts at the same origin
        assert train_positions.min() == 0


def test_expanding_window_cv_folds_grow_across_iterations():
    folds = expanding_window_cv_splits(
        train_start_idx=0, train_end_idx=8640, horizon=24, n_splits=4, min_train_hours=2880, n_rows=8640
    )
    sizes = [tm.sum() for tm, _ in folds]
    assert sizes == sorted(sizes), "expanding window training set must not shrink across folds"


@pytest.mark.skipif(not (DATA_DIR / "ETTh1.csv").exists(), reason="requires downloaded ETTh1.csv")
def test_real_etth1_is_a_regular_hourly_grid_matching_split_assumptions():
    from src.data import load_raw

    df = load_raw("ETTh1")
    assert_regular_hourly(df.index)  # raises if not perfectly regular
