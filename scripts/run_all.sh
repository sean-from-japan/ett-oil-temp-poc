#!/usr/bin/env bash
# One-command setup + reproduction: install pinned deps, download public data
# with SHA256 reporting, run tests, run the full benchmark.
# Usage: bash scripts/run_all.sh            (full, about 3-4 min on CPU)
#        bash scripts/run_all.sh --smoke    (reduced, ~20 s)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m src.cli download
python -m src.cli run "$MODE"
python -m pytest tests/ -q
