"""End-to-end orchestration: EDA, forecasting benchmark, nowcasting,
figures, and result export. Called by `src.cli`.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from src import config as cfg_module
from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    LOAD_COLUMNS,
    METADATA_ONLY_DATASETS,
    REPO_ROOT,
    RESULTS_DIR,
    TARGET_COLUMN,
    WARM_QUANTILE,
)
from src.data import (
    autocorrelation,
    basic_integrity_report,
    load_ot_correlation,
    load_raw,
    metadata_only_summary,
    monthly_drift,
)
from src.evaluate import evaluate_predictions, mae, skill_vs_persistence, warm_threshold_from_train, worst_examples
from src.features import build_forecast_features
from src.models import fit_predict_rf, fit_predict_ridge, persistence_predict, seasonal_naive_predict, select_ridge_alpha, select_rf_params
from src.nowcast import run_nowcast_task
from src.splits import build_split_ranges, get_forecast_split_masks
from src.uncertainty import summarize_predictions

CV_MIN_TRAIN_HOURS = cfg_module.CV_MIN_TRAIN_HOURS
_FIG = [FIGURES_DIR]


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def run_eda(dataset_name: str) -> dict:
    df = load_raw(dataset_name)
    report = basic_integrity_report(df, dataset_name)
    report["load_ot_correlation_full_series"] = load_ot_correlation(df).to_dict()
    acf = autocorrelation(df[TARGET_COLUMN], 168)
    report["acf_OT"] = {int(k): float(acf.loc[k]) for k in [1, 24, 48, 96, 168]}
    report["monthly_mean_OT"] = monthly_drift(df)["mean"].round(3).to_dict()

    ranges = build_split_ranges()
    per_split = {}
    for name, r in ranges.items():
        part = df.iloc[r.start_idx : r.end_idx]
        per_split[name] = {
            "OT_mean": float(part[TARGET_COLUMN].mean()),
            "OT_std": float(part[TARGET_COLUMN].std()),
            "OT_min": float(part[TARGET_COLUMN].min()),
            "OT_max": float(part[TARGET_COLUMN].max()),
            "load_means": part[LOAD_COLUMNS].mean().round(3).to_dict(),
            "load_ot_correlation": part[LOAD_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).round(3).to_dict(),
        }
    report["per_split_distribution"] = per_split
    hourly_profile = df.iloc[: ranges["train"].end_idx].groupby(df.index[: ranges["train"].end_idx].hour)[TARGET_COLUMN].mean()
    report["train_hour_of_day_OT_mean_range"] = float(hourly_profile.max() - hourly_profile.min())

    drift = monthly_drift(df)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(df.index, df[TARGET_COLUMN], linewidth=0.5)
    for name, r in ranges.items():
        axes[0].axvline(r.start_ts, color="k", linestyle="--", linewidth=0.8)
        axes[0].text(r.start_ts, axes[0].get_ylim()[1] * 0.92, f" {name}", fontsize=8)
    axes[0].axvline(ranges["test"].end_ts, color="k", linestyle="--", linewidth=0.8)
    axes[0].text(ranges["test"].end_ts, axes[0].get_ylim()[1] * 0.92, " unused", fontsize=8)
    axes[0].set_title(f"{dataset_name}: OT over time with 12/4/4 (30-day month) split boundaries")
    axes[0].set_ylabel("OT")

    axes[1].stem(acf.index, acf.values, markerfmt=" ", basefmt=" ")
    axes[1].set_title(f"{dataset_name}: OT autocorrelation (lags 0-168 h, full series, descriptive)")
    axes[1].set_xlabel("lag (hours)")

    drift["mean"].plot(ax=axes[2], marker="o")
    axes[2].set_title(f"{dataset_name}: monthly mean OT (distribution drift)")
    axes[2].tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(_FIG[0] / f"eda_{dataset_name}.png", dpi=110)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    corr = df.iloc[: ranges["train"].end_idx][LOAD_COLUMNS + [TARGET_COLUMN]].corr()
    im = ax2.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_yticklabels(corr.columns)
    ax2.set_title(f"{dataset_name}: load/OT correlation (train split)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(_FIG[0] / f"corr_{dataset_name}.png", dpi=110)
    plt.close(fig2)

    return report


# ---------------------------------------------------------------------------
# Forecasting benchmark for one (dataset, horizon)
# ---------------------------------------------------------------------------

TASK_ORIGIN_ONLY = "forecasting"
TASK_LOAD_AT_T = "forecasting_load_at_T_known"


def run_forecast_task(df: pd.DataFrame, dataset_name: str, horizon: int, cv_n_splits: int, seed: int, target_load_known: bool = False) -> dict:
    X, y, valid_mask = build_forecast_features(df, horizon, target_load_known=target_load_known)
    ranges = build_split_ranges()
    n_rows = len(df)
    split_masks = get_forecast_split_masks(n_rows, horizon, ranges)

    train_mask = split_masks["train"] & valid_mask
    val_mask = split_masks["val"] & valid_mask
    test_mask = split_masks["test"] & valid_mask

    def bounds(mask):
        ts = df.index[mask]
        return {
            "n": int(mask.sum()),
            "first_origin": str(ts.min()),
            "last_origin": str(ts.max()),
            "first_target": str(ts.min() + pd.Timedelta(hours=horizon)),
            "last_target": str(ts.max() + pd.Timedelta(hours=horizon)),
        }

    out = {
        "dataset": dataset_name,
        "task_type": TASK_LOAD_AT_T if target_load_known else TASK_ORIGIN_ONLY,
        "horizon_hours": horizon,
        "split_boundaries": {
            name: {"start": str(r.start_ts), "end_exclusive": str(r.end_ts), "start_idx": r.start_idx, "end_idx": r.end_idx}
            for name, r in ranges.items()
        },
        "sample_boundaries": {"train": bounds(train_mask), "val": bounds(val_mask), "test": bounds(test_mask)},
        "n_feature_columns": int(X.shape[1]),
        "models": {},
    }

    y_full = y.to_numpy()
    warm_threshold = warm_threshold_from_train(y_full[train_mask], WARM_QUANTILE)
    out["warm_threshold_train_quantile"] = {"quantile": WARM_QUANTILE, "value": warm_threshold}

    persist_mae = {}
    test_preds = {}

    def record(model_name, pred_val, pred_test, fit_time, pt_val, pt_test):
        entry = {}
        for split_name, mask, pred, pt in (("val", val_mask, pred_val, pt_val), ("test", test_mask, pred_test, pt_test)):
            m = evaluate_predictions(y_full[mask], pred, df.index[mask], warm_threshold=warm_threshold)
            m["fit_time_s"] = fit_time
            m["predict_time_s"] = pt
            m["skill_vs_persistence_pct"] = skill_vs_persistence(m["mae"], persist_mae[split_name])
            entry[split_name] = m
        out["models"][model_name] = entry
        test_preds[model_name] = pred_test

    def timed_index(arr, mask):
        t0 = time.perf_counter()
        sel = arr[mask]
        return sel, time.perf_counter() - t0

    # --- Baselines: deterministic functions of data observed by the origin ---
    persist_series, dt = persistence_predict(df, horizon)
    p_arr = persist_series.to_numpy()
    persist_mae["val"] = mae(y_full[val_mask], p_arr[val_mask])
    persist_mae["test"] = mae(y_full[test_mask], p_arr[test_mask])
    pv, tv = timed_index(p_arr, val_mask)
    pt, tt = timed_index(p_arr, test_mask)
    record("persistence", pv, pt, 0.0, dt + tv, tt)

    # Daily seasonal naive only where the reference t+h-24 <= t exists at the
    # origin (h <= 24); weekly (168 h) is valid for every horizon used here.
    for label, period in (("seasonal_naive_daily", 24), ("seasonal_naive_weekly", 168)):
        if period < horizon:
            out.setdefault("skipped_models", {})[label] = f"period {period}h < horizon {horizon}h: reference value not observed at origin"
            continue
        s_series, dt = seasonal_naive_predict(df, horizon, period=period)
        s_arr = s_series.to_numpy()
        assert not np.isnan(s_arr[val_mask]).any() and not np.isnan(s_arr[test_mask]).any()
        sv, tv = timed_index(s_arr, val_mask)
        st, tt = timed_index(s_arr, test_mask)
        record(label, sv, st, 0.0, dt + tv, tt)

    # Hyperparameter search: expanding-window CV restricted to the purged
    # train split, so no CV label can come from the val/test periods.
    Xv = X.to_numpy()
    cv_mask = train_mask
    ridge_search = select_ridge_alpha(
        Xv, y_full, ranges["train"].start_idx, ranges["train"].end_idx, horizon, cv_n_splits, CV_MIN_TRAIN_HOURS, valid_mask=cv_mask
    )
    pred_val, fit_t, pt_val, ridge_model, scaler = fit_predict_ridge(
        Xv[train_mask], y_full[train_mask], Xv[val_mask], ridge_search["best_alpha"], seed
    )
    t0 = time.perf_counter()
    pred_test = ridge_model.predict(scaler.transform(Xv[test_mask]))
    pt_test = time.perf_counter() - t0
    record("ridge", pred_val, pred_test, fit_t, pt_val, pt_test)
    out["ridge_hyperparam_search"] = ridge_search
    out["ridge_coefficients_top10_abs_standardized"] = [
        [c, float(v)] for c, v in sorted(zip(X.columns, ridge_model.coef_.tolist()), key=lambda kv: -abs(kv[1]))[:10]
    ]

    rf_search = select_rf_params(
        Xv, y_full, ranges["train"].start_idx, ranges["train"].end_idx, horizon, cv_n_splits, CV_MIN_TRAIN_HOURS, seed, valid_mask=cv_mask
    )
    bp = rf_search["best_params"]
    pred_val, fit_t, pt_val, rf_model = fit_predict_rf(
        Xv[train_mask], y_full[train_mask], Xv[val_mask], bp["max_depth"], bp["n_estimators"], seed
    )
    t0 = time.perf_counter()
    pred_test = rf_model.predict(Xv[test_mask])
    pt_test = time.perf_counter() - t0
    record("random_forest", pred_val, pred_test, fit_t, pt_val, pt_test)
    out["rf_hyperparam_search"] = rf_search
    out["rf_feature_importance_top10"] = [
        [c, float(v)] for c, v in sorted(zip(X.columns, rf_model.feature_importances_.tolist()), key=lambda kv: -kv[1])[:10]
    ]

    # --- Validation-based model choice. Test metrics are computed above for
    # reporting, but nothing below reads them to make a decision. ---
    val_mae_by_model = {name: m["val"]["mae"] for name, m in out["models"].items()}
    selected = min(val_mae_by_model, key=val_mae_by_model.get)
    out["selected_model_by_validation"] = selected
    out["val_mae_by_model"] = val_mae_by_model

    timestamps_test = df.index[test_mask]
    frame = {
        "origin_timestamp": timestamps_test,
        "target_timestamp": timestamps_test + pd.Timedelta(hours=horizon),
        "y_true": y_full[test_mask],
    }
    for name, pred in test_preds.items():
        frame[f"pred_{name}"] = pred
    out["frozen_predictions"] = pd.DataFrame(frame)
    out["worst_examples_selected_model"] = worst_examples(y_full[test_mask], test_preds[selected], timestamps_test, k=10, horizon=horizon)
    sel_res = test_preds[selected] - y_full[test_mask]
    out["selected_model_test_residual"] = {"mean": float(sel_res.mean()), "std": float(sel_res.std())}
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat_row(dataset, horizon, task_type, model_name, split_name, m, selected):
    return {
        "dataset": dataset,
        "task_type": task_type,
        "horizon": horizon,
        "model": model_name,
        "split": split_name,
        "selected_by_validation": model_name == selected,
        "n": m["n"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "skill_vs_persistence_pct": m.get("skill_vs_persistence_pct"),
        "skill_vs_train_mean_pct": m.get("skill_vs_train_mean_pct"),
        "fit_time_s": m.get("fit_time_s"),
        "predict_time_s": m.get("predict_time_s"),
        "warm_threshold": m.get("warm_subset", {}).get("threshold"),
        "warm_subset_n": m.get("warm_subset", {}).get("n_warm"),
        "warm_subset_mae": m.get("warm_subset", {}).get("mae"),
    }


def run_pipeline(run_config) -> dict:
    t_start = time.perf_counter()
    _FIG[0] = FIGURES_DIR / "smoke" if run_config.smoke else FIGURES_DIR
    _FIG[0].mkdir(parents=True, exist_ok=True)
    results = {
        "run_config": {"smoke": run_config.smoke, "datasets": run_config.datasets, "horizons": run_config.horizons, "seed": run_config.seed, "cv_n_splits": run_config.cv_n_splits},
        "eda": {},
        "metadata_only": {},
        "forecasting": [],
        "nowcasting": [],
    }
    scenarios = ((False, TASK_ORIGIN_ONLY, ""), (True, TASK_LOAD_AT_T, "_loadT"))

    for name in run_config.datasets:
        results["eda"][name] = run_eda(name)
    if not run_config.smoke:
        for name in METADATA_ONLY_DATASETS:
            results["metadata_only"][name] = metadata_only_summary(name)

    flat_rows = []
    all_predictions = {}
    for name in run_config.datasets:
        df = load_raw(name)
        for horizon in run_config.horizons:
            for load_known, task_type, tag in scenarios:
                task_out = run_forecast_task(df, name, horizon, run_config.cv_n_splits, run_config.seed, target_load_known=load_known)
                key = f"{name}{tag}_h{horizon}"
                all_predictions[key] = task_out.pop("frozen_predictions")
                results["forecasting"].append(task_out)
                for model_name, split_metrics in task_out["models"].items():
                    for split_name, m in split_metrics.items():
                        flat_rows.append(_flat_row(name, horizon, task_type, model_name, split_name, m, task_out["selected_model_by_validation"]))
                _residual_figure(f"{name}{tag}", horizon, all_predictions[key], task_out["selected_model_by_validation"])

        now_out = run_nowcast_task(df, name, run_config.cv_n_splits, run_config.seed)
        all_predictions[f"{name}_nowcast"] = now_out.pop("frozen_predictions")
        results["nowcasting"].append(now_out)
        for model_name, split_metrics in now_out["models"].items():
            for split_name, m in split_metrics.items():
                flat_rows.append(_flat_row(name, 0, "nowcasting_contemporaneous", model_name, split_name, m, now_out["selected_model_by_validation"]))

    results_df = pd.DataFrame(flat_rows)
    numeric = results_df[["mae", "rmse"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite MAE/RMSE produced; refusing to write results.")

    suffix = "_smoke" if run_config.smoke else ""
    results_df.to_csv(RESULTS_DIR / f"benchmark_results{suffix}.csv", index=False, float_format="%.6f")
    with open(RESULTS_DIR / f"benchmark_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    pred_dir = RESULTS_DIR / ("predictions_smoke" if run_config.smoke else "predictions")
    pred_dir.mkdir(exist_ok=True)
    for key, preds in all_predictions.items():
        preds.to_csv(pred_dir / f"{key}.csv", index=False, float_format="%.6f")

    if not run_config.smoke:
        _make_comparison_figures(results_df, run_config)
        summarize_predictions(pred_dir, RESULTS_DIR)
    _write_run_manifest(run_config, results, pred_dir, suffix, time.perf_counter() - t_start)
    return results


def _write_run_manifest(run_config, results, pred_dir, suffix, elapsed_s):
    config_values = {
        k: getattr(cfg_module, k)
        for k in [
            "RAW_BASE_URL", "MODELED_DATASETS", "METADATA_ONLY_DATASETS", "LOAD_COLUMNS", "TARGET_COLUMN", "SEED", "HORIZONS",
            "DATASET_START", "TRAIN_MONTHS", "VAL_MONTHS", "TEST_MONTHS", "LAG_HOURS", "ROLLING_WINDOWS",
            "CV_N_SPLITS", "CV_MIN_TRAIN_HOURS", "WARM_QUANTILE",
        ]
    }
    config_json = json.dumps(config_values, sort_keys=True)
    source_manifest_path = DATA_DIR / "source_manifest.json"
    source = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
    for name, info in source.items():
        local = REPO_ROOT / info["local_path"]
        info["sha256_verified_at_run"] = _sha256_file(local) == info["sha256"] if local.exists() else False

    n_rows = {name: int(rep["n_rows"]) for name, rep in results["eda"].items()}
    manifest = {
        "run_mode": "smoke" if run_config.smoke else "full",
        "run_config": results["run_config"],
        "config_values": config_values,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "code_sha256": {str(p.relative_to(REPO_ROOT)): _sha256_file(p) for p in sorted((REPO_ROOT / "src").glob("*.py"))},
        "source_data": source,
        "split_policy": "30-day-month hour offsets 12/4/4 from 2016-07-01 00:00; origin and target must both fall in the split",
        "rows_after_20_months_unused": {name: n - 20 * 30 * 24 for name, n in n_rows.items()},
        "sample_boundaries": {
            f"{r['dataset']}_{r['task_type']}_h{r['horizon_hours']}": r["sample_boundaries"] for r in results["forecasting"]
        } | {f"{r['dataset']}_nowcast": r["sample_boundaries"] for r in results["nowcasting"]},
        "selected_model_by_validation": {
            f"{r['dataset']}_{r['task_type']}_h{r['horizon_hours']}": r["selected_model_by_validation"] for r in results["forecasting"]
        },
        "prediction_exports": {
            str(p.relative_to(REPO_ROOT)): _sha256_file(p) for p in sorted(pred_dir.glob("*.csv"))
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "elapsed_s": round(elapsed_s, 1),
        "note": "Timings are wall-clock on this machine and vary between runs; metrics are deterministic given seed, data hashes and package versions.",
    }
    (RESULTS_DIR / f"run_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")


def _residual_figure(name, horizon, preds: pd.DataFrame, selected: str):
    res = preds[f"pred_{selected}"] - preds["y_true"]
    res_p = preds["pred_persistence"] - preds["y_true"]
    tgt = pd.to_datetime(preds["target_timestamp"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(tgt, res_p, linewidth=0.4, alpha=0.6, label="persistence")
    if selected != "persistence":
        axes[0].plot(tgt, res, linewidth=0.4, alpha=0.8, label=f"{selected} (val-selected)")
    axes[0].axhline(0, color="k", linewidth=0.6)
    axes[0].set_title(f"{name} h={horizon}: test residual (pred - true) by target time")
    axes[0].legend(fontsize=7)
    by_hour = pd.DataFrame({"h": tgt.dt.hour, "sel": res.abs(), "per": res_p.abs()}).groupby("h").mean()
    axes[1].plot(by_hour.index, by_hour["per"], marker="o", label="persistence")
    if selected != "persistence":
        axes[1].plot(by_hour.index, by_hour["sel"], marker="o", label=selected)
    axes[1].set_title("test MAE by target hour of day")
    axes[1].set_xlabel("hour")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_FIG[0] / f"residuals_{name}_h{horizon}.png", dpi=110)
    plt.close(fig)


def _make_comparison_figures(results_df: pd.DataFrame, run_config) -> None:
    for task_type, fname in ((TASK_ORIGIN_ONLY, "compare_test_mae.png"), (TASK_LOAD_AT_T, "compare_test_mae_loadT.png")):
        test_df = results_df[(results_df["split"] == "test") & (results_df["task_type"] == task_type)]
        fig, axes = plt.subplots(1, len(run_config.datasets), figsize=(6 * len(run_config.datasets), 4.5), squeeze=False)
        for ax, name in zip(axes[0], run_config.datasets):
            sub = test_df[test_df["dataset"] == name]
            pivot = sub.pivot(index="horizon", columns="model", values="mae")
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{name}: test MAE ({task_type})")
            ax.set_xlabel("horizon (hours)")
            ax.set_ylabel("MAE")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(_FIG[0] / fname, dpi=110)
        plt.close(fig)
