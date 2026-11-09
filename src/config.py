"""Central configuration for the ETT benchmark.

All paths, URLs, split boundaries, and modeling knobs live here so the
pipeline, tests, and CLI share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# Official public source: https://github.com/zhouhaoyi/ETDataset
# Raw files served from the `main` branch of that repository.
RAW_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

# Modeled in full (hourly resolution): ETTh1, ETTh2.
# Inspected at metadata level only: ETTm1, ETTm2.
MODELED_DATASETS = ["ETTh1", "ETTh2"]
METADATA_ONLY_DATASETS = ["ETTm1", "ETTm2"]
ALL_DATASETS = MODELED_DATASETS + METADATA_ONLY_DATASETS

LOAD_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COLUMN = "OT"

SEED = 20240521

# Forecast horizons in hours (the task's primary bounded objective).
HORIZONS = [1, 24, 96]
SMOKE_HORIZONS = [24]

# Hourly data starts 2016-07-01 00:00:00 (ETTh1/ETTh2, confirmed on download).
# Standard 12/4/4-month contiguous split over the first 20 months, using the
# conventional 30-day-month approximation. This
# mirrors the split used in the original ETT / Informer benchmark.
DATASET_START = "2016-07-01 00:00:00"
TRAIN_MONTHS = 12
VAL_MONTHS = 4
TEST_MONTHS = 4

# Causal feature-engineering window sizes (hours), all strictly backward-looking.
LAG_HOURS = [0, 1, 2, 3, 6, 12, 24, 48, 168]  # lag 0 = observation at the origin t
ROLLING_WINDOWS = [6, 24, 168]

# Expanding-window internal CV (within train only) for hyperparameter choice.
CV_N_SPLITS = 4
CV_MIN_TRAIN_HOURS = 24 * 30 * 4  # at least 4 months before first CV fold

# Train-only quantile defining the "warm / high-OT" evaluation subset.
WARM_QUANTILE = 0.75

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class RunConfig:
    """Knobs that vary between a smoke run and the full run."""

    smoke: bool = False
    datasets: list = field(default_factory=lambda: list(MODELED_DATASETS))
    horizons: list = field(default_factory=lambda: list(HORIZONS))
    cv_n_splits: int = CV_N_SPLITS
    seed: int = SEED

    @classmethod
    def for_mode(cls, smoke: bool) -> "RunConfig":
        if smoke:
            return cls(
                smoke=True,
                datasets=["ETTh1"],
                horizons=list(SMOKE_HORIZONS),
                cv_n_splits=2,
                seed=SEED,
            )
        return cls(smoke=False, datasets=list(MODELED_DATASETS), horizons=list(HORIZONS), cv_n_splits=CV_N_SPLITS, seed=SEED)
