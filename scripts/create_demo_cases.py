#!/usr/bin/env python3
"""Create Streamlit demo input CSVs from the project's held-out test predictions.

Run from the Air Quality Intelligence project root, or place this file in the
project's scripts/ directory and run:
    python scripts/create_demo_cases.py

The generated *_input.csv files contain only historical sensor/weather inputs.
Actual benzene values and recorded ANN predictions are kept in separate
reference files and must NEVER be uploaded to the Streamlit app.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def is_project_root(path: Path) -> bool:
    return (
        (path / "src" / "airquality" / "config.py").is_file()
        and (path / "scripts").is_dir()
        and (path / "data").is_dir()
        and (path / "outputs").is_dir()
    )


def find_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not is_project_root(root):
            raise FileNotFoundError(
                f"The supplied project folder does not look like the project root: {root}\n"
                "Expected src/airquality/config.py, scripts/, data/ and outputs/."
            )
        return root

    candidates: list[Path] = []
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend([start, *start.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and is_project_root(candidate):
            return candidate
        seen.add(candidate)

    raise FileNotFoundError(
        "Could not locate the Air Quality Intelligence project folder.\n"
        "Open Anaconda Prompt, change directory to the folder containing "
        "README.md, and run:\n"
        "    python scripts/create_demo_cases.py\n"
        "Alternatively, run this downloaded file with --project-root \"FULL\\PATH\\TO\\air-quality-intelligence\"."
    )


def to_naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare typical and largest-error examples for the Streamlit demonstration."
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Project root (folder containing README.md, src/, scripts/, data/ and outputs/)."
    )
    args = parser.parse_args()

    try:
        project_root = find_project_root(args.project_root)

        # Import project configuration/data parser only after discovering root.
        sys.path.insert(0, str(project_root / "src"))
        from airquality.config import BASE_FEATURES, TARGET, WINDOW_SIZE  # noqa: WPS433
        from airquality.data import load_uci_csv  # noqa: WPS433

        raw_path = project_root / "data" / "raw" / "AirQualityUCI.csv"
        predictions_path = project_root / "outputs" / "tables" / "test_predictions.csv"
        metadata_path = project_root / "models" / "model_metadata.json"
        output_dir = project_root / "data" / "demo" / "cases"

        missing_files = [p for p in (raw_path, predictions_path) if not p.is_file()]
        if missing_files:
            missing_text = "\n".join(f"  - {p}" for p in missing_files)
            raise FileNotFoundError(
                "Required project output(s) were not found:\n"
                f"{missing_text}\n\n"
                "First confirm that these commands completed successfully from the project root:\n"
                "  python scripts/download_data.py\n"
                "  python scripts/run_pipeline.py"
            )

        lookback = int(WINDOW_SIZE)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            lookback = int(metadata.get("window_size", WINDOW_SIZE))
        if lookback < 1:
            raise ValueError(f"Invalid model lookback length in metadata: {lookback}")

        predictions = pd.read_csv(predictions_path)
        required_prediction_columns = {"timestamp", "actual", "ann_prediction"}
        missing_columns = required_prediction_columns - set(predictions.columns)
        if missing_columns:
            raise ValueError(
                "The test prediction file does not have the expected project columns.\n"
                f"Missing: {sorted(missing_columns)}\n"
                f"Found: {predictions.columns.tolist()}\n"
                "Please confirm you are using outputs/tables/test_predictions.csv from the completed project pipeline."
            )

        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        predictions["actual"] = pd.to_numeric(predictions["actual"], errors="coerce")
        predictions["ann_prediction"] = pd.to_numeric(predictions["ann_prediction"], errors="coerce")
        predictions = predictions.dropna(subset=["timestamp", "actual", "ann_prediction"]).copy()
        if predictions.empty:
            raise ValueError("No usable held-out test predictions were found in test_predictions.csv.")

        predictions["absolute_error"] = (
            predictions["actual"] - predictions["ann_prediction"]
        ).abs()
        # Typical example = held-out point nearest the median absolute error.
        median_error = float(predictions["absolute_error"].median())
        typical_index = (predictions["absolute_error"] - median_error).abs().idxmin()
        # Incorrect example = genuinely largest absolute error in the held-out test set.
        worst_index = predictions["absolute_error"].idxmax()

        frame = load_uci_csv(raw_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        references: list[dict[str, Any]] = []

        chosen_cases = [
            ("typical_case", predictions.loc[typical_index]),
            ("largest_error_case", predictions.loc[worst_index]),
        ]

        for label, prediction_row in chosen_cases:
            target_timestamp = to_naive_timestamp(prediction_row["timestamp"])
            if target_timestamp not in frame.index:
                closest = frame.index.get_indexer([target_timestamp], method="nearest")[0]
                closest_ts = frame.index[closest] if closest >= 0 else "no nearby timestamp"
                raise ValueError(
                    f"Prediction timestamp {target_timestamp} is not present in the UCI source data.\n"
                    f"Nearest source timestamp: {closest_ts}. Check that the test_predictions.csv and "
                    "AirQualityUCI.csv belong to the same pipeline run."
                )

            target_position = int(frame.index.get_loc(target_timestamp))
            history = frame.iloc[target_position - lookback:target_position]
            if len(history) != lookback:
                raise ValueError(
                    f"Could not extract {lookback} historical rows before {target_timestamp}; "
                    f"only found {len(history)}."
                )
            time_differences = history.index.to_series().diff().dropna()
            if not time_differences.eq(pd.Timedelta(hours=1)).all():
                raise ValueError(
                    f"The historical readings before {target_timestamp} are not consecutive hourly rows."
                )

            # Exactly the same permitted feature columns that the app expects.
            app_input = history.loc[:, BASE_FEATURES].copy()
            app_input.insert(0, "timestamp", history.index.strftime("%Y-%m-%d %H:%M:%S"))
            input_path = output_dir / f"{label}_input.csv"
            app_input.to_csv(input_path, index=False, na_rep="")

            actual = float(prediction_row["actual"])
            predicted = float(prediction_row["ann_prediction"])
            residual = actual - predicted
            references.append({
                "case": label,
                "input_file": input_path.name,
                "forecast_target_timestamp": target_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "last_input_timestamp": history.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "lookback_rows": lookback,
                "actual_benzene_ug_m3": actual,
                "recorded_ann_prediction_ug_m3": predicted,
                "absolute_error_ug_m3": abs(residual),
                "residual_actual_minus_prediction": residual,
                "interpretation": (
                    "Underprediction: actual is higher than forecast."
                    if residual > 0 else
                    "Overprediction: forecast is higher than actual."
                    if residual < 0 else "Exact prediction."
                ),
                "source": "Original chronological held-out test prediction; actual value is reference-only and excluded from app input.",
            })

        # Guardrail: the app input files must not contain the target or other
        # reference-analyser ground-truth pollutant columns.
        forbidden = {"CO(GT)", "NMHC(GT)", TARGET, "NOx(GT)", "NO2(GT)"}
        for label, _ in chosen_cases:
            check = pd.read_csv(output_dir / f"{label}_input.csv", nrows=0)
            leaked = forbidden.intersection(check.columns)
            if leaked:
                raise RuntimeError(f"Forbidden reference-analyser columns found in app input: {sorted(leaked)}")

        references_json = output_dir / "case_references.json"
        references_csv = output_dir / "case_references.csv"
        references_json.write_text(json.dumps(references, indent=2), encoding="utf-8")
        pd.DataFrame(references).to_csv(references_csv, index=False)

        readme = output_dir / "README.txt"
        readme.write_text(
            "DEMONSTRATION CASES\n"
            "===================\n\n"
            "Upload ONLY these files to Streamlit:\n"
            "  - typical_case_input.csv\n"
            "  - largest_error_case_input.csv\n\n"
            "Do NOT upload case_references.csv or case_references.json. They contain\n"
            "the actual benzene target and the recorded ANN prediction for comparison\n"
            "after you have generated the forecast in the web app.\n\n"
            "For the incorrect-prediction demonstration, upload largest_error_case_input.csv,\n"
            "generate the forecast, then compare it against largest_error_case in\n"
            "case_references.csv.\n",
            encoding="utf-8",
        )

        print("\nDEMONSTRATION CASES CREATED SUCCESSFULLY")
        print(f"Project folder: {project_root}")
        print(f"Output folder:  {output_dir}")
        for item in references:
            print(f"\n{item['case']}")
            print(f"  Upload file: {item['input_file']}")
            print(f"  Forecast target time: {item['forecast_target_timestamp']}")
            print(f"  Actual: {item['actual_benzene_ug_m3']:.3f} µg/m³")
            print(f"  ANN prediction: {item['recorded_ann_prediction_ug_m3']:.3f} µg/m³")
            print(f"  Absolute error: {item['absolute_error_ug_m3']:.3f} µg/m³")
            print(f"  {item['interpretation']}")
        print("\nIMPORTANT: Upload only the *_input.csv files to Streamlit.")
        print(f"Reference comparison file: {references_csv}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
