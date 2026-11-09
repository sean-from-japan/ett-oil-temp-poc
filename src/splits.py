"""Chronological split logic for the ETT hourly datasets.

Design (documented in README.md / REPORT.md):

* We follow the 30-day-month convention used by the original ETT / Informer
  benchmark code (not real calendar months): train = 12*30*24 hours,
  val = 4*30*24 hours, test = 4*30*24 hours, i.e. hour-offset boundaries
  [0, 8640), [8640, 11520), [11520, 14400) from the dataset start. This
  covers the first 20*30 = 600 days of the series; remaining rows (the
  series runs ~725 days) are *not* silently mixed into any split - they are
  simply unused, and that is reported explicitly in the run manifest.
* Splits are defined on **origin timestamp** ranges, not target timestamp
  ranges. For a forecasting sample with origin t and horizon h, the sample
  is only assigned to a split if its target timestamp t+h also falls
  strictly inside that split's range. This purges the last h origins of
  every split so a training/validation/test label never crosses into the
  next split - "accounting for target timestamps when splitting windows"
  rather than silently mixing adjacent data.
* Internal model selection uses expanding-window cross-validation *within
  the train split only*, with a gap of exactly `horizon` hours between the
  end of each CV training prefix and the start of its validation block, so
  that no CV training label is drawn from inside the corresponding CV
  validation window either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATASET_START, TEST_MONTHS, TRAIN_MONTHS, VAL_MONTHS

HOURS_PER_MONTH = 30 * 24  # the ETT/Informer 30-day-month convention


@dataclass(frozen=True)
class SplitRange:
    name: str
    start_idx: int  # inclusive hour offset from dataset start
    end_idx: int  # exclusive hour offset from dataset start
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp  # exclusive


def hour_offset_boundaries(
    train_months: int = TRAIN_MONTHS, val_months: int = VAL_MONTHS, test_months: int = TEST_MONTHS
) -> dict:
    train_end = train_months * HOURS_PER_MONTH
    val_end = train_end + val_months * HOURS_PER_MONTH
    test_end = val_end + test_months * HOURS_PER_MONTH
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def build_split_ranges(dataset_start: str = DATASET_START) -> dict:
    """Return {split_name: SplitRange} with both hour-offset and timestamp boundaries."""
    start_ts = pd.Timestamp(dataset_start)
    offsets = hour_offset_boundaries()
    ranges = {}
    for name, (s, e) in offsets.items():
        ranges[name] = SplitRange(
            name=name,
            start_idx=s,
            end_idx=e,
            start_ts=start_ts + pd.Timedelta(hours=s),
            end_ts=start_ts + pd.Timedelta(hours=e),
        )
    return ranges


def assert_regular_hourly(index: pd.DatetimeIndex, dataset_start: str = DATASET_START) -> None:
    """Fail loudly if the series is not a perfectly regular hourly grid.

    The position-as-hour-offset arithmetic used throughout this module
    depends on that regularity; ETTh1/ETTh2 satisfy it exactly (verified in
    EDA), but we guard here rather than assume it silently.
    """
    expected = pd.date_range(start=dataset_start, periods=len(index), freq="h")
    if not index.equals(expected):
        raise ValueError(
            "Index is not a perfectly regular hourly grid starting at dataset_start; "
            "position-based split arithmetic is invalid for this series."
        )


def origin_mask_for_split(n_rows: int, split: SplitRange, horizon: int) -> np.ndarray:
    """Boolean mask over row positions [0, n_rows) selecting valid *origin* rows.

    A row position p is a valid origin for this split iff:
      - p is within [split.start_idx, split.end_idx)  (origin belongs to split)
      - p + horizon is within [split.start_idx, split.end_idx) (target does too)
    This purges boundary rows whose target would otherwise leak into the
    next split.
    """
    positions = np.arange(n_rows)
    origin_in_split = (positions >= split.start_idx) & (positions < split.end_idx)
    target_in_split = (positions + horizon >= split.start_idx) & (positions + horizon < split.end_idx)
    return origin_in_split & target_in_split


def get_forecast_split_masks(n_rows: int, horizon: int, ranges: dict | None = None) -> dict:
    ranges = ranges or build_split_ranges()
    return {name: origin_mask_for_split(n_rows, r, horizon) for name, r in ranges.items()}


def get_nowcast_split_masks(n_rows: int, ranges: dict | None = None) -> dict:
    """Nowcasting has horizon 0 (origin == target), so no purge is needed."""
    return get_forecast_split_masks(n_rows, horizon=0, ranges=ranges)


def expanding_window_cv_splits(
    train_start_idx: int,
    train_end_idx: int,
    horizon: int,
    n_splits: int,
    min_train_hours: int,
    n_rows: int | None = None,
) -> list:
    """Expanding-window CV folds within [train_start_idx, train_end_idx).

    Each fold i uses an expanding training prefix, a gap of `horizon` hours,
    then a validation block. Returns a list of (train_mask, val_mask) pairs
    as boolean arrays of length `n_rows` (defaults to train_end_idx), so
    they can be used directly to index any full-length feature array.
    """
    total = train_end_idx - train_start_idx
    usable = total - min_train_hours
    if usable <= 0:
        raise ValueError("min_train_hours exceeds the available train range.")
    val_block = max(usable // (n_splits + 1), horizon + 1)
    mask_len = n_rows if n_rows is not None else train_end_idx
    folds = []
    for i in range(n_splits):
        val_start = train_start_idx + min_train_hours + i * val_block
        val_end = val_start + val_block
        cv_train_end = val_start - horizon
        val_end = min(val_end, train_end_idx)
        if cv_train_end <= train_start_idx or val_start >= val_end:
            continue
        train_mask = np.zeros(mask_len, dtype=bool)
        val_mask = np.zeros(mask_len, dtype=bool)
        train_mask[train_start_idx:cv_train_end] = True
        val_mask[val_start:val_end] = True
        folds.append((train_mask, val_mask))
    if not folds:
        raise ValueError("No valid CV folds could be constructed with the given sizes.")
    return folds
