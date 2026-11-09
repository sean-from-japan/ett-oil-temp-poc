"""Contemporaneous regression ("nowcasting"): predict OT(t) from the six
non-OT load readings observed at the SAME timestamp t.

This is intentionally kept separate from the forecasting task. Its
information set - same-instant load readings - is not available in a true
forecasting deployment, where only information up to the *origin* is known
and the target is in the *future*. Results are labeled
`task_type = "nowcasting_contemporaneous"` end-to-end, compared against a
train-mean reference (not persistence, which would require OT itself), and
kept in a separate results section.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import CV_MIN_TRAIN_HOURS, LOAD_COLUMNS, SEED, TARGET_COLUMN, WARM_QUANTILE
from src.evaluate import evaluate_predictions, mae, skill_vs_persistence, warm_threshold_from_train, worst_examples
from src.features import build_nowcast_features
from src.models import fit_predict_ridge, select_ridge_alpha
from src.splits import build_split_ranges, get_nowcast_split_masks


def run_nowcast_task(df: pd.DataFrame, dataset_name: str, cv_n_splits: int = 4, seed: int = SEED) -> dict:
    X, y, valid_mask = build_nowcast_features(df)
    ranges = build_split_ranges()
    split_masks = get_nowcast_split_masks(len(df), ranges)
    masks = {s: split_masks[s] & valid_mask for s in ("train", "val", "test")}

    Xv = X.to_numpy()
    yv = y.to_numpy()

    search = select_ridge_alpha(
        Xv, yv, ranges["train"].start_idx, ranges["train"].end_idx,
        horizon=0, n_splits=cv_n_splits, min_train_hours=CV_MIN_TRAIN_HOURS, valid_mask=masks["train"],
    )
    pred_val, fit_time, pt_val, model, scaler = fit_predict_ridge(
        Xv[masks["train"]], yv[masks["train"]], Xv[masks["val"]], search["best_alpha"], seed
    )
    t0 = time.perf_counter()
    pred_test = model.predict(scaler.transform(Xv[masks["test"]]))
    pt_test = time.perf_counter() - t0
    pred_train = model.predict(scaler.transform(Xv[masks["train"]]))

    train_target = yv[masks["train"]]
    train_mean = float(train_target.mean())
    warm_threshold = warm_threshold_from_train(train_target, WARM_QUANTILE)

    results = {
        "task_type": "nowcasting_contemporaneous",
        "dataset": dataset_name,
        "information_set": "HUFL,HULL,MUFL,MULL,LUFL,LULL at the same timestamp t as the OT(t) target; OT itself never used",
        "reference_model": "train_mean (constant train-split OT mean)",
        "hyperparam_search": search,
        "sample_boundaries": {
            s: {"n": int(m.sum()), "first": str(df.index[m].min()), "last": str(df.index[m].max())} for s, m in masks.items()
        },
        "ridge_coefficients_standardized": dict(zip(X.columns, [float(c) for c in model.coef_])),
        "models": {},
    }

    preds = {
        "train_mean": {"val": np.full(masks["val"].sum(), train_mean), "test": np.full(masks["test"].sum(), train_mean)},
        "ridge_nowcast": {"val": pred_val, "test": pred_test},
    }
    times = {"train_mean": (0.0, 0.0, 0.0), "ridge_nowcast": (fit_time, pt_val, pt_test)}
    for name, by_split in preds.items():
        entry = {}
        for split_name in ("val", "test"):
            m_mask = masks[split_name]
            m = evaluate_predictions(yv[m_mask], by_split[split_name], df.index[m_mask], warm_threshold=warm_threshold)
            m["fit_time_s"] = times[name][0]
            m["predict_time_s"] = times[name][1] if split_name == "val" else times[name][2]
            m["skill_vs_train_mean_pct"] = skill_vs_persistence(m["mae"], mae(yv[m_mask], preds["train_mean"][split_name]))
            entry[split_name] = m
        results["models"][name] = entry

    val_mae = {name: results["models"][name]["val"]["mae"] for name in preds}
    results["selected_model_by_validation"] = min(val_mae, key=val_mae.get)

    # Diagnostics used to check (not assume) the explanation for weak
    # out-of-period nowcasting: error decomposition into bias vs. spread,
    # per-split load/OT correlation, and per-split OT/load levels.
    diag = {}
    for split_name in ("train", "val", "test"):
        m_mask = masks[split_name]
        p = pred_train if split_name == "train" else preds["ridge_nowcast"][split_name]
        res = p - yv[m_mask]
        part = df.loc[m_mask]
        diag[split_name] = {
            "ridge_mae": float(np.mean(np.abs(res))),
            "ridge_residual_mean_bias": float(res.mean()),
            "ridge_residual_std": float(res.std()),
            "ridge_mae_after_removing_split_bias": float(np.mean(np.abs(res - res.mean()))),
            "OT_mean": float(part[TARGET_COLUMN].mean()),
            "load_means": part[LOAD_COLUMNS].mean().round(3).to_dict(),
            "load_ot_correlation": part[LOAD_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).round(3).to_dict(),
        }
        # Diagnostic only (NOT a valid model, it is fitted on the very split
        # it is scored on): how well could a linear map from same-time loads
        # explain OT *within* this period at all?
        oracle = Ridge(alpha=1.0).fit(Xv[m_mask], yv[m_mask])
        diag[split_name]["diagnostic_only_in_split_refit_mae"] = float(np.mean(np.abs(oracle.predict(Xv[m_mask]) - yv[m_mask])))
    results["drift_diagnostics"] = diag

    tm = masks["test"]
    results["worst_examples"] = worst_examples(yv[tm], pred_test, df.index[tm], k=10)
    results["frozen_predictions"] = pd.DataFrame(
        {
            "timestamp": df.index[tm],
            "y_true": yv[tm],
            "pred_ridge_nowcast": pred_test,
            "pred_train_mean": preds["train_mean"]["test"],
        }
    )
    return results
