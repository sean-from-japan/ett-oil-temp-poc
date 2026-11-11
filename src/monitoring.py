"""Monitoring-band summary for the 1-hour-ahead test predictions.

For a residual monitor that alerts when |observed - predicted| exceeds a band,
the 99th percentile of the absolute error is the band that keeps routine
false alerts at about 1%. A narrower band means smaller deviations can be
flagged at the same false-alert rate. Measured on the test period itself.

Usage: python -m src.monitoring   (reads results/predictions/, writes results/monitoring_band.csv)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = []
    for name in ["ETTh1", "ETTh2"]:
        path = ROOT / "results" / "predictions" / f"{name}_loadT_h1.csv"
        if not path.exists():  # --smoke runs only ETTh1, h=24
            continue
        d = pd.read_csv(path)
        for model in ["persistence", "ridge"]:
            err = (d[f"pred_{model}"] - d["y_true"]).abs()
            rows.append({"dataset": name, "horizon": 1, "model": model,
                         "abs_err_q99": round(float(err.quantile(0.99)), 3),
                         "hours_abs_err_over_2": int((err > 2).sum()), "n": len(d)})
    out = ROOT / "results" / "monitoring_band.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
