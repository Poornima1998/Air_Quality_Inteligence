#!/usr/bin/env python3
"""Run a five-case LLM shift-handover evaluation for Air Quality Intelligence.

Run from the project root:
  python scripts/evaluate_genai.py --n 5

Only ANN predictions and the latest available sensor context are sent to the LLM.
The held-out actual target, residuals, and comparator predictions are never sent.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airquality.genai import generate_handover_summary

SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]


def spaced_indices(length: int, n: int) -> list[int]:
    if length < 1:
        raise ValueError("Prediction table is empty")
    n = min(max(1, n), length)
    if n == 1:
        return [length // 2]
    # Deterministic evenly spaced sample including first and last test examples.
    return sorted(set(round(x) for x in __import__('numpy').linspace(0, length - 1, n)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of held-out examples (default: 5)")
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Create a local .env file; never paste the key into chat or source code.")
    if not os.getenv("GENAI_MODEL"):
        raise SystemExit("GENAI_MODEL is not set. Set a currently supported model identifier in .env.")

    pred_path = ROOT / "outputs/tables/test_predictions.csv"
    meta_path = ROOT / "models/model_metadata.json"
    if not pred_path.exists() or not meta_path.exists():
        raise SystemExit("Required project outputs are missing. Run the ANN pipeline first.")
    frame = pd.read_csv(pred_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validation_mae = float(meta["validation_ann"]["MAE"])
    indices = spaced_indices(len(frame), args.n)
    results: list[dict] = []

    for case_number, idx in enumerate(indices, start=1):
        row = frame.iloc[idx]
        recent = {}
        for feature in SENSORS:
            value = row.get(f"input_last__{feature}")
            recent[feature] = float(value) if pd.notna(value) else None
        # Explicit allow-list: do not pass row.to_dict(), actual, residuals, or baseline values.
        context = {
            "test_case": case_number,
            "timestamp_of_forecast_origin": str(row["timestamp"]),
            "recent_sensor_readings": recent,
            "ann_forecast": round(float(row["ann_prediction"]), 6),
            "ann_validation_mae": round(validation_mae, 6),
            "missing_input_fraction": float(row["missing_input_fraction"]),
        }
        note, details = generate_handover_summary(
            prediction=float(row["ann_prediction"]), unit="µg/m³",
            validation_mae=validation_mae, recent_context=context,
        )
        results.append({
            "test_case": case_number,
            "test_row_index": int(idx),
            "timestamp": str(row["timestamp"]),
            "ann_forecast_ug_m3": round(float(row["ann_prediction"]), 6),
            "generated_note": note,
            **details,
            "actual_target_sent_to_llm": False,
            "manual_quality_review": "PENDING",
        })
        print(f"Completed GenAI case {case_number}/{len(indices)}")

    out_dir = ROOT / "outputs/genai"
    table_dir = ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "task": "LLM-generated operational shift-handover note",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(results),
        "model_requested": os.getenv("GENAI_MODEL"),
        "prompt_version": "shift_handover_v1",
        "prediction_source": "outputs/tables/test_predictions.csv",
        "leakage_control": "Actual future target, residuals and baseline predictions were excluded from LLM input.",
        "results": results,
    }
    json_path = out_dir / "genai_handover_evaluation.json"
    json_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = [{k: v for k, v in r.items() if k != "generated_note"} | {"generated_note": r["generated_note"]} for r in results]
    pd.DataFrame(flat_rows).to_csv(table_dir / "genai_evaluation_log.csv", index=False)
    review_rows = [{
        "test_case": r["test_case"], "timestamp": r["timestamp"], "model": r.get("model"),
        "factuality_1_to_5": "", "completeness_1_to_5": "", "clarity_1_to_5": "",
        "unsupported_claims_present_yes_no": "", "reviewer_notes": "",
    } for r in results]
    pd.DataFrame(review_rows).to_csv(table_dir / "genai_human_review.csv", index=False)
    cost_rows = [{
        "test_case": r["test_case"], "model": r.get("model"),
        "latency_seconds": r.get("latency_seconds"),
        "input_tokens": r.get("prompt_tokens"), "output_tokens": r.get("completion_tokens"),
        "total_tokens": r.get("total_tokens"), "estimated_cost_usd": r.get("estimated_cost_usd"),
        "pricing_rates_configured": r.get("pricing_rates_configured"),
    } for r in results]
    pd.DataFrame(cost_rows).to_csv(table_dir / "genai_cost_summary.csv", index=False)
    print(f"Saved: {json_path}")
    print(f"Saved: {table_dir / 'genai_evaluation_log.csv'}")
    print(f"Saved: {table_dir / 'genai_human_review.csv'}")
    print(f"Saved: {table_dir / 'genai_cost_summary.csv'}")
    print("Human review remains pending; complete it after reviewing each generated note.")

if __name__ == "__main__":
    main()
