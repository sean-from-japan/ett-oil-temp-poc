"""CLI entrypoint.

Usage:
    python -m src.cli download            # fetch data from the official source
    python -m src.cli run --smoke         # fast reduced run (1 dataset, 1 horizon)
    python -m src.cli run --full          # complete benchmark (default)
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import RunConfig
from src.download import download_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Public ETT forecasting benchmark CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download ETTh1/ETTh2/ETTm1/ETTm2 from the official source.")

    run_parser = sub.add_parser("run", help="Run the benchmark pipeline.")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Reduced smoke run: ETTh1 only, horizon=24, 2 CV folds.")
    mode.add_argument("--full", action="store_true", help="Full run: ETTh1+ETTh2, horizons 1/24/96 (default).")

    args = parser.parse_args()

    if args.command == "download":
        download_all()
        return

    if args.command == "run":
        smoke = bool(args.smoke)
        cfg = RunConfig.for_mode(smoke=smoke)
        print(f"[cli] starting {'SMOKE' if smoke else 'FULL'} run: datasets={cfg.datasets} horizons={cfg.horizons}", file=sys.stderr)
        t0 = time.perf_counter()
        from src.pipeline import run_pipeline  # deferred import: matplotlib/sklearn only needed here

        run_pipeline(cfg)
        print(f"[cli] done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
