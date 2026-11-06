from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import BASE_FEATURES, TARGET, WINDOW_SIZE, TRAIN_FRACTION, VALIDATION_FRACTION


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train_scaled: np.ndarray
    y_train_raw: np.ndarray
    x_val: np.ndarray
    y_val_scaled: np.ndarray
    y_val_raw: np.ndarray
    x_test: np.ndarray
    y_test_scaled: np.ndarray
    y_test_raw: np.ndarray
    train_target_index: np.ndarray
    val_target_index: np.ndarray
    test_target_index: np.ndarray
    preprocessor: "FeaturePreprocessor"
    target_scaler: StandardScaler
    timestamps: pd.DatetimeIndex
    feature_names: list[str]
    split_indices: dict[str, int]
    raw_frame: pd.DataFrame


class FeaturePreprocessor:
    """Median imputation and scaling fitted on the training period only."""
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None
        self.lower_bounds_: pd.Series | None = None
        self.upper_bounds_: pd.Series | None = None

    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = frame[BASE_FEATURES].apply(pd.to_numeric, errors="coerce").replace(-200, np.nan)
        missing = base.isna().astype(float)
        missing.columns = [f"{c}__missing" for c in BASE_FEATURES]
        idx = pd.DatetimeIndex(frame.index)
        hour = idx.hour.to_numpy()
        dow = idx.dayofweek.to_numpy()
        doy = idx.dayofyear.to_numpy()
        calendar = pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 366),
            "doy_cos": np.cos(2 * np.pi * doy / 366),
        }, index=frame.index)
        return pd.concat([base, missing, calendar], axis=1)

    def fit(self, frame: pd.DataFrame, train_end: int) -> "FeaturePreprocessor":
        engineered = self._engineer(frame)
        train = engineered.iloc[:train_end]
        self.medians_ = train[BASE_FEATURES].median()
        self.lower_bounds_ = train[BASE_FEATURES].quantile(0.01)
        self.upper_bounds_ = train[BASE_FEATURES].quantile(0.99)
        # An all-missing training feature is a data-quality failure, not something to hide.
        if self.medians_.isna().any():
            bad = self.medians_[self.medians_.isna()].index.tolist()
            raise ValueError(f"These input features are entirely missing in training data: {bad}")
        filled = engineered.copy()
        filled.loc[:, BASE_FEATURES] = filled[BASE_FEATURES].fillna(self.medians_)
        filled = filled.reindex(columns=self.feature_names)
        self.scaler_ = StandardScaler().fit(filled.iloc[:train_end])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.scaler_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform().")
        engineered = self._engineer(frame)
        engineered.loc[:, BASE_FEATURES] = engineered[BASE_FEATURES].fillna(self.medians_)
        engineered = engineered.reindex(columns=self.feature_names)
        return self.scaler_.transform(engineered).astype(np.float32)


def load_uci_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse the official UCI semicolon-delimited, decimal-comma CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run: python scripts/download_data.py")
    frame = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8", low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.dropna(axis=1, how="all")
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
    frame = frame.dropna(axis=0, how="all")
    required = {"Date", "Time", TARGET, *BASE_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing expected UCI columns: {missing}. Found: {frame.columns.tolist()}")
    date_text = frame["Date"].astype(str).str.strip()
    time_text = frame["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
    timestamp = pd.to_datetime(date_text + " " + time_text, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if timestamp.isna().any():
        bad = int(timestamp.isna().sum())
        raise ValueError(f"Could not parse {bad} Date/Time values from the UCI file.")
    frame = frame.drop(columns=["Date", "Time"])
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(-200, np.nan)
    frame.index = pd.DatetimeIndex(timestamp, name="timestamp")
    frame = frame.sort_index()
    duplicate_count = int(frame.index.duplicated().sum())
    if duplicate_count:
        frame = frame[~frame.index.duplicated(keep="first")]
    # Insert missing clock hours. A target row is then always exactly one hour after its input window.
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="timestamp")
    frame = frame.reindex(full_index)
    frame.attrs["duplicate_timestamps_removed"] = duplicate_count
    frame.attrs["rows_inserted_for_missing_hours"] = int(len(full_index) - len(timestamp) + duplicate_count)
    return frame


def make_feature_names() -> list[str]:
    return BASE_FEATURES + [f"{c}__missing" for c in BASE_FEATURES] + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"
    ]


def build_data_bundle(csv_path: str | Path, window_size: int = WINDOW_SIZE) -> DataBundle:
    frame = load_uci_csv(csv_path)
    n = len(frame)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    feature_names = make_feature_names()
    target = pd.to_numeric(frame[TARGET], errors="coerce").replace(-200, np.nan).to_numpy(dtype=float)

    # Target at row j is forecast from rows [j-window_size, ..., j-1].
    sample_idx = np.arange(window_size, n, dtype=int)
    valid = np.isfinite(target[sample_idx])
    sample_idx = sample_idx[valid]
    train_mask = sample_idx < train_end
    val_mask = (sample_idx >= train_end) & (sample_idx < val_end)
    test_mask = sample_idx >= val_end
    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("At least one chronological split has no valid target windows. Check the target missingness.")

    # Fit preprocessing only through the last row actually used as a training
    # input (the row immediately before the last observed training target).
    # This excludes the first validation-context row from fitted statistics.
    preprocessor_fit_rows = int(sample_idx[train_mask].max())
    preprocessor = FeaturePreprocessor(feature_names).fit(frame, preprocessor_fit_rows)
    x_rows = preprocessor.transform(frame)
    x_all = np.stack([x_rows[j-window_size:j] for j in sample_idx]).astype(np.float32)
    y_all = target[sample_idx].astype(np.float32)

    y_scaler = StandardScaler().fit(y_all[train_mask].reshape(-1, 1))
    y_scaled = y_scaler.transform(y_all.reshape(-1, 1)).ravel().astype(np.float32)
    return DataBundle(
        x_train=x_all[train_mask], y_train_scaled=y_scaled[train_mask], y_train_raw=y_all[train_mask],
        x_val=x_all[val_mask], y_val_scaled=y_scaled[val_mask], y_val_raw=y_all[val_mask],
        x_test=x_all[test_mask], y_test_scaled=y_scaled[test_mask], y_test_raw=y_all[test_mask],
        train_target_index=sample_idx[train_mask], val_target_index=sample_idx[val_mask], test_target_index=sample_idx[test_mask],
        preprocessor=preprocessor, target_scaler=y_scaler, timestamps=frame.index,
        feature_names=feature_names, split_indices={"train_end": train_end, "validation_end": val_end, "n_rows": n,
                                                "preprocessor_fit_rows": preprocessor_fit_rows},
        raw_frame=frame,
    )


def leakage_audit(bundle: DataBundle) -> dict:
    cols = set(bundle.feature_names)
    target_leaks = [c for c in cols if c in {"CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"}]
    # For every sample, last input row is target index - 1 by construction.
    max_train_feature_idx = int(bundle.train_target_index.max() - 1)
    min_val_target_idx = int(bundle.val_target_index.min())
    max_val_feature_idx = int(bundle.val_target_index.max() - 1)
    min_test_target_idx = int(bundle.test_target_index.min())
    return {
        "reference_analyser_pollutant_inputs": target_leaks,
        "target_row_excluded_from_input_window": True,
        "forecast_horizon_hours": 1,
        "lookback_hours": int(bundle.x_train.shape[1]),
        "chronological_split": True,
        "preprocessor_fit_rows": int(bundle.split_indices["preprocessor_fit_rows"]),
        "preprocessor_max_fit_row_index": int(bundle.split_indices["preprocessor_fit_rows"] - 1),
        "preprocessor_uses_only_training_input_rows": bool(
            bundle.split_indices["preprocessor_fit_rows"] - 1 == max_train_feature_idx),
        "test_used_for_model_selection": False,
        "train_max_feature_row_index": max_train_feature_idx,
        "validation_min_target_row_index": min_val_target_idx,
        "validation_max_feature_row_index": max_val_feature_idx,
        "test_min_target_row_index": min_test_target_idx,
        "pass": (not target_leaks
                and max_train_feature_idx < bundle.split_indices["train_end"]
                and bundle.split_indices["preprocessor_fit_rows"] - 1 == max_train_feature_idx),
    }


def save_data_audit(bundle: DataBundle, destination: str | Path) -> None:
    frame = bundle.raw_frame
    numeric = frame.select_dtypes(include=[np.number])
    missing = numeric.isna().sum().sort_values(ascending=False)
    target = frame[TARGET]
    summary = {
        "rows_after_hourly_reindex": len(frame),
        "columns": frame.shape[1],
        "date_start": str(frame.index.min()),
        "date_end": str(frame.index.max()),
        "target": TARGET,
        "target_observed": int(target.notna().sum()),
        "target_missing": int(target.isna().sum()),
        "target_missing_pct": float(target.isna().mean() * 100),
        "target_min": float(target.min()),
        "target_median": float(target.median()),
        "target_mean": float(target.mean()),
        "target_max": float(target.max()),
        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
        "split_indices": bundle.split_indices,
        "split_samples": {"train": len(bundle.y_train_raw), "validation": len(bundle.y_val_raw), "test": len(bundle.y_test_raw)},
        "leakage_audit": leakage_audit(bundle),
    }
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(summary, indent=2), encoding="utf-8")
