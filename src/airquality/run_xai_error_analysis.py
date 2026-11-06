#!/usr/bin/env python3
"""
Reproduce Integrated Gradients explanations for the three largest ANN errors.

Run from the project root:
    python scripts/run_xai_error_analysis.py

This script does NOT retrain the ANN or alter the train/validation/test split.
It loads the saved model artefacts and rebuilds the existing chronological test
windows from the original UCI dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from airquality.config import DATA_RAW, TABLES_DIR, FIGURES_DIR
from airquality.data import build_data_bundle
from airquality.inference import load_ann_artifacts
from airquality.models import predict_ann
from airquality.xai import integrated_gradients_full


def find_dataset() -> Path:
    candidates = [
        DATA_RAW / "AirQualityUCI.csv",
        DATA_RAW / "airqualityuci.csv",
        DATA_RAW / "AirQualityUCI.txt",
        DATA_RAW / "AirQualityUCI",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    csvs = sorted(DATA_RAW.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        f"Could not find the UCI dataset in {DATA_RAW}. "
        "Place AirQualityUCI.csv in data/raw/ and run again."
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    model, preprocessor, target_scaler, metadata = load_ann_artifacts(ROOT)
    bundle = build_data_bundle(dataset_path, window_size=int(metadata["window_size"]))

    # Use the archived held-out predictions to identify the three largest
    # errors, then align each timestamp to its exact held-out input window.
    prediction_path = TABLES_DIR / "test_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing {prediction_path}. Run the existing pipeline first."
        )
    saved = pd.read_csv(prediction_path, parse_dates=["timestamp"])
    required = {"timestamp", "actual", "ann_prediction"}
    if not required.issubset(saved.columns):
        raise ValueError(f"{prediction_path} must contain {sorted(required)}")

    saved["absolute_error"] = (
        pd.to_numeric(saved["actual"]) - pd.to_numeric(saved["ann_prediction"])
    ).abs()
    cases = saved.nlargest(3, "absolute_error").copy()

    # Timestamp-to-test-window lookup is safer than relying on row ordering.
    test_lookup = {
        pd.Timestamp(bundle.timestamps[int(target_idx)]): i
        for i, target_idx in enumerate(bundle.test_target_index)
    }

    rows = []
    feature_names = list(bundle.feature_names)
    for rank, (_, record) in enumerate(cases.iterrows(), start=1):
        timestamp = pd.Timestamp(record["timestamp"])
        if timestamp not in test_lookup:
            raise ValueError(
                f"{timestamp} from test_predictions.csv was not found in the "
                "rebuilt held-out test set. Check dataset and pipeline versions."
            )
        local_test_index = test_lookup[timestamp]
        x_one = bundle.x_test[local_test_index]

        # Verify that the saved prediction and current artefacts correspond.
        pred_scaled = predict_ann(model, x_one[None, ...], device="cpu")
        pred = float(target_scaler.inverse_transform(
            np.asarray(pred_scaled).reshape(-1, 1)
        ).ravel()[0])
        saved_pred = float(record["ann_prediction"])
        if not np.isclose(pred, saved_pred, atol=0.03, rtol=0):
            raise ValueError(
                f"Model/prediction mismatch for {timestamp}: "
                f"recomputed={pred:.5f}, saved={saved_pred:.5f}. "
                "Do not mix model artefacts and output files from different runs."
            )

        # (24, 22) signed IG values, summed over the 24 time steps.
        attribution = integrated_gradients_full(
            model, x_one, steps=64, device="cpu"
        )
        signed = attribution.sum(axis=0)
        absolute = np.abs(attribution).sum(axis=0)
        order = np.argsort(absolute)[::-1]
        top_n = min(8, len(feature_names))

        for feature_index, feature in enumerate(feature_names):
            rows.append({
                "error_rank": rank,
                "test_index": int(record.get("Unnamed: 0", local_test_index)),
                "timestamp": timestamp.isoformat(sep=" "),
                "actual_ug_m3": float(record["actual"]),
                "prediction_ug_m3": pred,
                "residual_actual_minus_predicted": float(record["actual"]) - pred,
                "absolute_error_ug_m3": abs(float(record["actual"]) - pred),
                "feature": feature,
                "signed_ig_standardised_output": float(signed[feature_index]),
                "absolute_ig_standardised_output": float(absolute[feature_index]),
            })

        # Individual plot for each error case, ranked by absolute attribution.
        selected = order[:top_n][::-1]
        labels = [feature_names[i] for i in selected]
        values = [float(signed[i]) for i in selected]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels, values)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Integrated Gradients – Error Case {rank}: "
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Actual {float(record['actual']):.2f} | "
            f"Predicted {pred:.2f} | "
            f"Absolute error {abs(float(record['actual']) - pred):.2f} µg/m³"
        )
        ax.set_xlabel("Signed attribution (standardised model output)")
        ax.set_ylabel("Input feature (aggregated across 24 hours)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"xai_case_{rank}.png", dpi=180)
        plt.close(fig)

        print(
            f"Case {rank}: {timestamp} | actual={float(record['actual']):.3f} | "
            f"predicted={pred:.3f} | abs_error={abs(float(record['actual'])-pred):.3f}"
        )
        print("  Top features:", ", ".join(
            f"{feature_names[i]} ({signed[i]:+.3f})" for i in order[:5]
        ))

    result = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "three_error_cases_integrated_gradients.csv"
    result.to_csv(csv_path, index=False)

    summary = cases[["timestamp", "actual", "ann_prediction", "absolute_error"]].copy()
    summary.insert(0, "error_rank", range(1, len(summary) + 1))
    summary_path = TABLES_DIR / "three_error_cases_summary.csv"
    summary.to_csv(summary_path, index=False)

    log_path = ROOT / "experiments" / "xai_error_analysis_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame([{
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "method": "Integrated Gradients",
        "steps": 64,
        "baseline": "all-zero standardised input",
        "cases_analysed": len(cases),
        "source": "saved held-out test_predictions.csv",
        "test_set_used_for_model_selection": False,
        "status": "completed",
    }])
    log.to_csv(log_path, index=False)

    print("\nGenerated files:")
    for path in [
        FIGURES_DIR / "xai_case_1.png",
        FIGURES_DIR / "xai_case_2.png",
        FIGURES_DIR / "xai_case_3.png",
        csv_path,
        summary_path,
        log_path,
    ]:
        print(f" - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
