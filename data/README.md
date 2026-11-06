# Dataset directory

Run `python scripts/download_data.py` from the project root. It downloads the official UCI Air Quality archive and writes:

- `raw/AirQualityUCI.csv`
- `raw/dataset_manifest.json` (source URL, retrieval timestamp, byte size and SHA-256)

The actual dataset is intentionally not replaced by a synthetic file. The initial ZIP was assembled without a UCI CSV because the archive download endpoint could not be reached from the build environment. Download it before the full training run and retain the file in the final submission archive.

`demo/input_template.csv` is a 24-hour sample of observed UCI sensor readings for testing the web app. It includes optional `C6H6(GT)` history for chart display only; that target column is not used as a model input. The sample is not a substitute for the full UCI dataset.
