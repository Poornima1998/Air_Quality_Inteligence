from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .models import TemporalANN, predict_ann


def load_ann_artifacts(project_root: str | Path):
    root = Path(project_root)
    models = root / "models"
    required = [models / "model_metadata.json", models / "preprocessor.joblib",
                models / "target_scaler.joblib", models / "ann_state_dict.pt"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Trained model files are missing: " + ", ".join(missing) + ". Run scripts/run_pipeline.py first.")
    meta = json.loads((models / "model_metadata.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(models / "preprocessor.joblib")
    scaler = joblib.load(models / "target_scaler.joblib")
    model = TemporalANN(meta["ann_input_size"], meta["ann_hidden_size"], meta["ann_num_layers"], meta["ann_dropout"])
    state = torch.load(models / "ann_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, preprocessor, scaler, meta


def predict_from_sensor_history(sensor_history, model, preprocessor, target_scaler, metadata):
    """sensor_history: DataFrame indexed by hourly timestamps, with the 8 sensor/weather columns."""
    from .config import BASE_FEATURES
    if len(sensor_history) < int(metadata["window_size"]):
        raise ValueError(f"Upload at least {metadata['window_size']} hourly rows.")
    history = sensor_history.sort_index().tail(int(metadata["window_size"])).copy()
    if not all(c in history.columns for c in BASE_FEATURES):
        raise ValueError(f"Input file must contain these columns: {BASE_FEATURES}")
    if history.index.has_duplicates:
        raise ValueError("Input timestamps must be unique.")
    diffs = history.index.to_series().diff().dropna()
    if not (diffs == __import__("pandas").Timedelta(hours=1)).all():
        raise ValueError("The most recent input rows must be consecutive hourly readings.")
    rows = preprocessor.transform(history)
    x = rows[None, :, :].astype(np.float32)
    scaled = predict_ann(model, x, device="cpu")
    predicted = float(target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()[0])
    return predicted, x


def check_training_range(sensor_history, preprocessor) -> dict:
    """Flag, without rejecting, numeric inputs outside training-period 1st–99th percentiles."""
    from .config import BASE_FEATURES
    result = {}
    out_total = 0
    for col in BASE_FEATURES:
        series = sensor_history[col].replace(-200, np.nan)
        low = preprocessor.lower_bounds_.get(col, np.nan)
        high = preprocessor.upper_bounds_.get(col, np.nan)
        mask = series.notna() & ((series < low) | (series > high))
        count = int(mask.sum())
        if count:
            result[col] = {"outside_rows": count, "lower_1pct_training": float(low), "upper_99pct_training": float(high)}
            out_total += count
    return {"outside_training_range_count": out_total, "features": result, "range_definition": "training-period 1st–99th percentile"}
