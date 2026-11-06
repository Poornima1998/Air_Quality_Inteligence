#!/usr/bin/env python3
"""Download and verify the official UCI Air Quality dataset."""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official UCI dataset from: {URL}")
    response = requests.get(URL, timeout=args.timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        candidates = [n for n in archive.namelist() if Path(n).name.lower() == "airqualityuci.csv"]
        if not candidates:
            candidates = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError(f"No CSV found in UCI archive. Archive members: {archive.namelist()}")
        csv_bytes = archive.read(candidates[0])
    out = args.output_dir / "AirQualityUCI.csv"
    out.write_bytes(csv_bytes)
    digest = sha256(out)
    manifest = {
        "dataset": "UCI Air Quality (id=360)",
        "source_url": URL,
        "landing_page": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "file": out.name,
        "bytes": out.stat().st_size,
        "sha256": digest,
        "retrieved_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "citation": "Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F.",
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {out} ({out.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest}")
    print("Next: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
