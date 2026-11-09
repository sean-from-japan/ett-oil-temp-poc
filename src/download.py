"""Deterministic downloader for the public ETDataset CSVs.

Fetches directly from the official repository
(https://github.com/zhouhaoyi/ETDataset) raw file host, verifies the
response, writes the file, and reports a SHA256 hash so results are
reproducible and the untracked `data/` folder can be re-verified by anyone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.config import ALL_DATASETS, DATA_DIR, RAW_BASE_URL

TIMEOUT_SECONDS = 60


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, dest_dir: Path = DATA_DIR, force: bool = False) -> dict:
    """Download one <name>.csv from the official source into dest_dir.

    Returns a metadata dict with url, local path, byte size and sha256.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE_URL}/{name}.csv"
    dest = dest_dir / f"{name}.csv"

    if dest.exists() and not force:
        pass
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "ett-benchmark/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Unexpected HTTP status {resp.status} for {url}")
                data = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        dest.write_bytes(data)

    size = dest.stat().st_size
    digest = sha256_of_file(dest)
    return {
        "name": name,
        "source_url": url,
        "local_path": str(dest.relative_to(dest_dir.parent)),
        "byte_count": size,
        "sha256": digest,
    }


def download_all(dest_dir: Path = DATA_DIR, force: bool = False, datasets=None) -> dict:
    datasets = datasets or ALL_DATASETS
    manifest = {}
    for name in datasets:
        print(f"[download] fetching {name} from official source ...", file=sys.stderr)
        info = download_one(name, dest_dir=dest_dir, force=force)
        manifest[name] = info
        print(f"[download] {name}: {info['byte_count']} bytes, sha256={info['sha256']}", file=sys.stderr)
    manifest_path = dest_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[download] wrote {manifest_path}", file=sys.stderr)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public ETDataset CSVs with SHA256 reporting.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a local file already exists.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset names to fetch (default: all of ETTh1/ETTh2/ETTm1/ETTm2).",
    )
    args = parser.parse_args()
    download_all(force=args.force, datasets=args.datasets)


if __name__ == "__main__":
    main()
