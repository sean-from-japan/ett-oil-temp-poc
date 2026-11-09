"""Shared fixtures: a small synthetic but realistic hourly series.

Synthetic (not downloaded) data is used for unit-level correctness tests so
they run offline and deterministically; a few integration-level tests in
test_splits.py additionally exercise the real downloaded ETTh1 series when
present (skipped otherwise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n = 2000
    idx = pd.date_range("2016-07-01 00:00:00", periods=n, freq="h")
    hours = idx.hour.values
    t = np.arange(n)
    ot = 20 + 5 * np.sin(2 * np.pi * hours / 24) + 0.01 * t + rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {
            "HUFL": rng.normal(5, 1, n),
            "HULL": rng.normal(2, 0.5, n),
            "MUFL": rng.normal(1.5, 0.3, n),
            "MULL": rng.normal(0.5, 0.2, n),
            "LUFL": rng.normal(4, 0.8, n),
            "LULL": rng.normal(1.3, 0.3, n),
            "OT": ot,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
