#!/usr/bin/env python3
"""Run EDA, leakage audit, ANN experiments, baseline tuning, XAI and final test evaluation."""
from __future__ import annotations
import argparse, copy, json, math, sys, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airquality.config import *
from airquality.data import build_data_bundle, save_data_audit, leakage_audit, FeaturePreprocessor
from airquality.models import TemporalANN, fit_ann, predict_ann, set_seed
from airquality.metrics import regression_metrics
from airquality.xai import integrated_gradients, integrated_gradients_full, grouped_permutation_importance


def ensure_dirs():
    for p in [MODELS_DIR, FIGURES_DIR, TABLES_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    def convert(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        return str(x)
    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def inverse_y(bundle, scaled):
    return bundle.target_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()


def make_eda(bundle):
    frame = bundle.raw_frame
    # Missingness table
    missing = frame.isna().mean().mul(100).sort_values(ascending=False).rename("missing_pct").reset_index()
    missing.columns = ["feature", "missing_pct"]
    missing.to_csv(TABLES_DIR / "missingness.csv", index=False)
    desc = frame.describe().T
    desc.to_csv(TABLES_DIR / "descriptive_statistics.csv")
    corr = frame.corr(numeric_only=True)
    corr.to_csv(TABLES_DIR / "correlation_matrix.csv")
    monthly = frame.resample("MS")[TARGET].agg(["count", "mean", "median", "std", "min", "max"])
    monthly.to_csv(TABLES_DIR / "monthly_target_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    frame[TARGET].plot(ax=ax, linewidth=0.7)
    ax.set_title("Hourly Benzene Ground-Truth Concentration")
    ax.set_xlabel("Timestamp"); ax.set_ylabel("Benzene (µg/m³)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "target_time_series.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    missing.sort_values("missing_pct", ascending=False).plot.bar(x="feature", y="missing_pct", ax=ax, legend=False)
    ax.set_title("Missing Values by Column (after -200 conversion)")
    ax.set_ylabel("Missing rows (%)"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "missingness.png", dpi=160); plt.close(fig)

    if len(corr.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix – Descriptive, Not Causal")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    frame[TARGET].dropna().hist(bins=35, ax=ax)
    ax.set_title("Distribution of Benzene Concentration")
    ax.set_xlabel("Benzene (µg/m³)"); ax.set_ylabel("Frequency")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160); plt.close(fig)


def _flatten_windows_from_frame(frame, target_indices, preprocessor, window_size):
    """Build flattened windows from fold-specific, training-fitted preprocessing."""
    rows = preprocessor.transform(frame)
    windows = np.stack([rows[int(j) - window_size:int(j)] for j in target_indices]).astype(np.float32)
    return windows.reshape(len(windows), -1)


def tune_baseline(bundle, n_iter=10, seed=42):
    # Outer training/validation/test split was fixed before this search. Within the
    # outer training period, refit imputation/scaling separately in each CV fold.
    xtr = bundle.x_train.reshape(len(bundle.x_train), -1)
    xval = bundle.x_val.reshape(len(bundle.x_val), -1)
    ytr = bundle.y_train_raw
    yval = bundle.y_val_raw
    train_target_indices = bundle.train_target_index
    grid = {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_iter": [100, 160, 220],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [10, 20, 35],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=seed))
    cv = TimeSeriesSplit(n_splits=3)
    fold_cache = []
    for fold_no, (train_idx, val_idx) in enumerate(cv.split(xtr), start=1):
        # Input rows available at the last training target are strictly earlier
        # than that target; fit preprocessing only on those historical rows.
        fold_fit_end = int(train_target_indices[train_idx].max())
        fold_preprocessor = FeaturePreprocessor(bundle.feature_names).fit(bundle.raw_frame, fold_fit_end)
        fold_x = _flatten_windows_from_frame(bundle.raw_frame, train_target_indices, fold_preprocessor,
                                               bundle.x_train.shape[1])
        fold_cache.append((train_idx, val_idx, fold_x))
    rows = []
    best = None
    best_cv = float("inf")
    cv_started = time.perf_counter()
    for i, params in enumerate(candidates, start=1):
        fold_maes = []
        for train_idx, val_idx, fold_x in fold_cache:
            model = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
            model.fit(fold_x[train_idx], ytr[train_idx])
            pred = model.predict(fold_x[val_idx])
            fold_maes.append(mean_absolute_error(ytr[val_idx], pred))
        cv_mae = float(np.mean(fold_maes))
        rows.append({"experiment_id": f"BASE-{i:02d}", **params, "cv_mae_mean": cv_mae,
                     "cv_mae_std": float(np.std(fold_maes, ddof=1)), "cv_fold_maes_json": json.dumps(fold_maes),
                     "folds": len(fold_maes), "fold_specific_preprocessing": True})
        if cv_mae < best_cv:
            best_cv, best = cv_mae, params
    search = pd.DataFrame(rows).sort_values("cv_mae_mean")
    search.to_csv(EXPERIMENTS_DIR / "baseline_search.csv", index=False)
    cv_search_seconds = time.perf_counter() - cv_started
    model = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **best)
    baseline_fit_started = time.perf_counter()
    model.fit(xtr, ytr)
    baseline_fit_seconds = time.perf_counter() - baseline_fit_started
    val_pred = model.predict(xval)
    val_metrics = regression_metrics(yval, val_pred)
    return model, best, search, val_metrics, {"cv_search_seconds": cv_search_seconds, "fit_seconds": baseline_fit_seconds}


def ann_configs():
    # Four deliberately weak/diagnostic attempts are followed by a compact validation-only search.
    failures = [
        {"name": "ANN-FAIL-01-short-context", "lookback": 3, "hidden_size": 32, "num_layers": 1, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 128, "optimizer": "AdamW"},
        {"name": "ANN-FAIL-02-high-learning-rate", "lookback": 24, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.03, "batch_size": 128, "optimizer": "AdamW"},
        {"name": "ANN-FAIL-03-over-regularised", "lookback": 24, "hidden_size": 8, "num_layers": 1, "dropout": 0.7, "learning_rate": 0.0001, "batch_size": 256, "optimizer": "AdamW"},
        {"name": "ANN-FAIL-04-deep-high-dropout", "lookback": 24, "hidden_size": 128, "num_layers": 3, "dropout": 0.6, "learning_rate": 0.0003, "batch_size": 256, "optimizer": "RMSprop"},
    ]
    search = [
        {"name": "ANN-SEARCH-01", "lookback": 24, "hidden_size": 32, "num_layers": 1, "dropout": 0.10, "learning_rate": 0.001, "batch_size": 64, "optimizer": "AdamW"},
        {"name": "ANN-SEARCH-02", "lookback": 24, "hidden_size": 64, "num_layers": 1, "dropout": 0.20, "learning_rate": 0.001, "batch_size": 128, "optimizer": "AdamW"},
        {"name": "ANN-SEARCH-03", "lookback": 24, "hidden_size": 64, "num_layers": 2, "dropout": 0.20, "learning_rate": 0.0005, "batch_size": 64, "optimizer": "AdamW"},
        {"name": "ANN-SEARCH-04", "lookback": 24, "hidden_size": 96, "num_layers": 2, "dropout": 0.25, "learning_rate": 0.001, "batch_size": 128, "optimizer": "AdamW"},
        {"name": "ANN-SEARCH-05", "lookback": 24, "hidden_size": 64, "num_layers": 2, "dropout": 0.10, "learning_rate": 0.003, "batch_size": 64, "optimizer": "RMSprop"},
        {"name": "ANN-SEARCH-06", "lookback": 24, "hidden_size": 96, "num_layers": 1, "dropout": 0.35, "learning_rate": 0.0005, "batch_size": 128, "optimizer": "RMSprop"},
        {"name": "ANN-SEARCH-07", "lookback": 24, "hidden_size": 32, "num_layers": 2, "dropout": 0.30, "learning_rate": 0.003, "batch_size": 64, "optimizer": "AdamW"},
        {"name": "ANN-SEARCH-08", "lookback": 24, "hidden_size": 96, "num_layers": 2, "dropout": 0.15, "learning_rate": 0.001, "batch_size": 64, "optimizer": "RMSprop"},
    ]
    return failures, search


def run_ann_experiments(bundle, max_epochs=28):
    failures, search = ann_configs()
    all_cfgs = failures + search
    log_rows, candidates = [], []
    for i, cfg in enumerate(all_cfgs):
        exp_id = cfg["name"]
        print(f"[{i+1}/{len(all_cfgs)}] Training {exp_id}", flush=True)
        row = {"experiment_id": exp_id, "seed": 42, **{k: v for k, v in cfg.items() if k != "name"}}
        try:
            result = fit_ann(bundle.x_train, bundle.y_train_scaled, bundle.x_val, bundle.y_val_scaled,
                             cfg, seed=42, max_epochs=max_epochs, patience=6)
            val_scaled = predict_ann(result["model"], bundle.x_val[:, -result["lookback"]:, :], device=result["device"])
            val_pred = inverse_y(bundle, val_scaled)
            metrics = regression_metrics(bundle.y_val_raw, val_pred)
            row.update({"status": "completed", "best_epoch": result["best_epoch"],
                        "training_seconds": result["training_seconds"], "val_mae": metrics["MAE"],
                        "val_rmse": metrics["RMSE"], "val_r2": metrics["R2"],
                        "train_final_huber_loss": result["history"][-1]["train_huber_loss"]})
            candidates.append({"config": cfg, "result": result, "val_metrics": metrics})
            hist = pd.DataFrame(result["history"])
            hist.to_csv(EXPERIMENTS_DIR / f"{exp_id.lower()}_learning_curve.csv", index=False)
        except Exception as exc:
            row.update({"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                        "best_epoch": None, "training_seconds": None, "val_mae": None,
                        "val_rmse": None, "val_r2": None})
        log_rows.append(row)
    if not candidates:
        raise RuntimeError("No ANN experiment completed successfully.")
    # All architecture/hyperparameter choices are made using validation MAE only.
    candidates.sort(key=lambda c: c["val_metrics"]["MAE"])
    best = candidates[0]
    best_val = best["val_metrics"]["MAE"]
    for row in log_rows:
        if row.get("status") == "completed":
            row["selection_outcome"] = "selected" if abs(row["val_mae"] - best_val) < 1e-12 and row["experiment_id"] == best["config"]["name"] else "not_selected_validation_mae_higher"
        else:
            row["selection_outcome"] = "unsuccessful_training"
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(EXPERIMENTS_DIR / "ann_experiment_log.csv", index=False)
    completed = log_df[log_df["status"] == "completed"].copy()
    for parameter in ["learning_rate", "batch_size", "num_layers", "hidden_size", "dropout", "optimizer", "lookback"]:
        if parameter in completed:
            grouped = completed.groupby(parameter, dropna=False)["val_mae"].agg(["count", "mean", "std", "min"]).reset_index()
            grouped.to_csv(EXPERIMENTS_DIR / f"effect_{parameter}.csv", index=False)
    if not completed.empty:
        completed.groupby("optimizer")["val_mae"].agg(["count", "mean", "std", "min"]).reset_index().to_csv(
            EXPERIMENTS_DIR / "optimizer_comparison.csv", index=False)
    return best, log_df


def repeat_best_ann(bundle, best_cfg, max_epochs=35):
    results = []
    for seed in RANDOM_SEEDS:
        print(f"Repeated final ANN fit – seed {seed}", flush=True)
        r = fit_ann(bundle.x_train, bundle.y_train_scaled, bundle.x_val, bundle.y_val_scaled,
                    best_cfg, seed=seed, max_epochs=max_epochs, patience=8)
        p_scaled = predict_ann(r["model"], bundle.x_val[:, -r["lookback"]:, :], device=r["device"])
        p = inverse_y(bundle, p_scaled)
        m = regression_metrics(bundle.y_val_raw, p)
        results.append({"seed": seed, "model": r["model"], "result": r, "val_metrics": m})
    # Select the median validation run, not the best lucky seed; test stays isolated.
    results.sort(key=lambda x: x["val_metrics"]["MAE"])
    selected = results[len(results) // 2]
    summary = {
        "seeds": [x["seed"] for x in results],
        "validation_mae_mean": float(np.mean([x["val_metrics"]["MAE"] for x in results])),
        "validation_mae_std": float(np.std([x["val_metrics"]["MAE"] for x in results], ddof=1)),
        "validation_mae_min": float(min(x["val_metrics"]["MAE"] for x in results)),
        "validation_mae_max": float(max(x["val_metrics"]["MAE"] for x in results)),
        "selected_seed_median_validation_mae": selected["val_metrics"]["MAE"],
    }
    pd.DataFrame([{"seed": x["seed"], **x["val_metrics"]} for x in results]).to_csv(EXPERIMENTS_DIR / "ann_repeated_seeds.csv", index=False)
    return selected, summary


def plot_learning_curves(initial_history, final_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history, ax in [("Initial ANN", initial_history, axes[0]), ("Selected ANN", final_history, axes[1])]:
        h = pd.DataFrame(history)
        ax.plot(h["epoch"], h["train_huber_loss"], label="Train Huber loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Huber loss")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["val_mae_scaled"], linestyle="--", label="Validation MAE (scaled)")
        ax2.set_ylabel("Validation MAE (scaled target)")
        ax.set_title(label)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.suptitle("ANN Learning Curves – Training vs Validation")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "ann_learning_curves.png", dpi=160); plt.close(fig)


def plot_model_outputs(bundle, predictions):
    y = predictions["actual"]
    ann = predictions["ann_prediction"]
    base = predictions["baseline_prediction"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, ann, alpha=.45, s=14, label="ANN")
    ax.scatter(y, base, alpha=.35, s=14, label="Classical baseline")
    lo, hi = float(min(np.min(y), np.min(ann), np.min(base))), float(max(np.max(y), np.max(ann), np.max(base)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("Actual benzene (µg/m³)"); ax.set_ylabel("Predicted benzene (µg/m³)")
    ax.set_title("Held-out Test: Predicted vs Actual"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_predicted_vs_actual.png", dpi=160); plt.close(fig)

    residual = y - ann
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(ann, residual, alpha=.45, s=14)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("ANN predicted benzene (µg/m³)"); ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("ANN Residual Analysis – Held-out Test")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_residuals.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(predictions["timestamp"]), y, label="Actual", linewidth=.8)
    ax.plot(pd.to_datetime(predictions["timestamp"]), ann, label="ANN forecast", linewidth=.8)
    ax.set_title("Held-out Test – Benzene Forecasts Over Time")
    ax.set_xlabel("Time"); ax.set_ylabel("Benzene (µg/m³)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "test_forecast_timeline.png", dpi=160); plt.close(fig)


def run_xai(model, bundle, device):
    ig = integrated_gradients(model, bundle.x_val[0], bundle.feature_names, steps=48, device=device)
    ig.to_csv(TABLES_DIR / "integrated_gradients_example.csv", index=False)
    perm = grouped_permutation_importance(model, bundle.x_val, bundle.y_val_raw, bundle.target_scaler,
                                          bundle.feature_names, repeats=5, device=device)
    perm.to_csv(TABLES_DIR / "ann_grouped_permutation_importance.csv", index=False)

    # Two local sanity checks: perturb the exact lag/feature coordinate with the largest IG magnitude.
    rows = []
    for case_no, sample_idx in enumerate([0, max(0, len(bundle.x_val) // 2)], start=1):
        x0 = bundle.x_val[sample_idx:sample_idx+1].copy()
        local_matrix = integrated_gradients_full(model, x0[0], steps=48, device=device)
        lag_idx, feature_idx = np.unravel_index(np.abs(local_matrix).argmax(), local_matrix.shape)
        top = bundle.feature_names[feature_idx]
        attribution = float(local_matrix[lag_idx, feature_idx])
        p0 = inverse_y(bundle, predict_ann(model, x0, device=device))[0]
        x1 = x0.copy(); x1[0, lag_idx, feature_idx] += 0.10
        p1 = inverse_y(bundle, predict_ann(model, x1, device=device))[0]
        delta = float(p1 - p0)
        consistent = bool(np.sign(delta) == np.sign(attribution)) if abs(delta) > 1e-9 else None
        rows.append({"case": case_no, "feature_perturbed": top, "lag_hours_before_forecast": int(bundle.x_train.shape[1] - 1 - lag_idx),
                     "perturbation_standardised_units": 0.10, "prediction_before": float(p0),
                     "prediction_after": float(p1), "prediction_change": delta,
                     "ig_attribution_for_perturbed_coordinate": attribution,
                     "direction_consistent_with_ig": consistent,
                     "interpretation": "Small local input perturbation; consistency is a sanity check, not a causal claim."})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "xai_sanity_checks.csv", index=False)
    return ig, perm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA_RAW / "AirQualityUCI.csv")
    parser.add_argument("--baseline-configs", type=int, default=10)
    parser.add_argument("--ann-epochs", type=int, default=28)
    parser.add_argument("--final-epochs", type=int, default=35)
    args = parser.parse_args()
    ensure_dirs()
    if not args.csv.exists():
        raise FileNotFoundError(f"Real UCI dataset not found at {args.csv}. First run: python scripts/download_data.py")

    print("1/8 Loading and preparing official UCI data...")
    bundle = build_data_bundle(args.csv)
    save_data_audit(bundle, TABLES_DIR / "data_audit.json")
    audit = leakage_audit(bundle)
    save_json(audit, TABLES_DIR / "leakage_audit.json")
    if not audit["pass"]:
        raise RuntimeError(f"Leakage audit failed: {audit}")
    print("2/8 EDA...")
    make_eda(bundle)

    print("3/8 Tuning classical baseline...")
    baseline, baseline_params, baseline_search, baseline_val_metrics, baseline_timing = tune_baseline(bundle, n_iter=args.baseline_configs)
    x_val_flat = bundle.x_val.reshape(len(bundle.x_val), -1)
    baseline_val_pred = baseline.predict(x_val_flat)

    print("4/8 ANN ablations and hyperparameter search...")
    best_ann, ann_log = run_ann_experiments(bundle, max_epochs=args.ann_epochs)
    print("5/8 Repeating selected ANN across three random seeds...")
    selected, seed_summary = repeat_best_ann(bundle, best_ann["config"], max_epochs=args.final_epochs)
    ann_model = selected["model"]
    ann_config = best_ann["config"]
    initial_cfg = copy.deepcopy(ann_configs()[1][0])
    initial_cfg["name"] = "ANN-INITIAL-BASELINE"
    initial_result = fit_ann(bundle.x_train, bundle.y_train_scaled, bundle.x_val, bundle.y_val_scaled,
                             initial_cfg, seed=42, max_epochs=args.final_epochs, patience=8)
    initial_ann_model = initial_result["model"]
    pd.DataFrame(initial_result["history"]).to_csv(EXPERIMENTS_DIR / "ann_initial_baseline_learning_curve.csv", index=False)
    pd.DataFrame(selected["result"]["history"]).to_csv(EXPERIMENTS_DIR / "ann_final_selected_learning_curve.csv", index=False)
    plot_learning_curves(initial_result["history"], selected["result"]["history"])

    print("6/8 Validation comparison (model choice only; test remains untouched)...")
    ann_val_scaled = predict_ann(ann_model, bundle.x_val[:, -selected["result"]["lookback"]:, :], device=selected["result"]["device"])
    ann_val_pred = inverse_y(bundle, ann_val_scaled)
    initial_val_scaled = predict_ann(initial_ann_model, bundle.x_val[:, -initial_result["lookback"]:, :], device=initial_result["device"])
    initial_val_pred = inverse_y(bundle, initial_val_scaled)
    dummy_val = DummyRegressor(strategy="mean").fit(bundle.x_train.reshape(len(bundle.x_train), -1), bundle.y_train_raw)
    dummy_val_pred = dummy_val.predict(x_val_flat)
    val_compare = pd.DataFrame([
        {"model": "Initial LSTM ANN", **regression_metrics(bundle.y_val_raw, initial_val_pred)},
        {"model": "Final LSTM ANN", **regression_metrics(bundle.y_val_raw, ann_val_pred)},
        {"model": "HistGradientBoosting", **regression_metrics(bundle.y_val_raw, baseline_val_pred)},
        {"model": "Training-mean reference", **regression_metrics(bundle.y_val_raw, dummy_val_pred)},
    ])
    val_compare.to_csv(TABLES_DIR / "validation_model_comparison.csv", index=False)

    print("7/8 Final one-time held-out test evaluation...")
    ann_inf_start = time.perf_counter()
    ann_test_scaled = predict_ann(ann_model, bundle.x_test[:, -selected["result"]["lookback"]:, :], device=selected["result"]["device"])
    ann_inference_total_s = time.perf_counter() - ann_inf_start
    ann_test_pred = inverse_y(bundle, ann_test_scaled)
    initial_inf_start = time.perf_counter()
    initial_test_scaled = predict_ann(initial_ann_model, bundle.x_test[:, -initial_result["lookback"]:, :], device=initial_result["device"])
    initial_inference_total_s = time.perf_counter() - initial_inf_start
    initial_test_pred = inverse_y(bundle, initial_test_scaled)
    flat_test = bundle.x_test.reshape(len(bundle.x_test), -1)
    baseline_inf_start = time.perf_counter()
    baseline_test_pred = baseline.predict(flat_test)
    baseline_inference_total_s = time.perf_counter() - baseline_inf_start
    dummy_final = DummyRegressor(strategy="mean").fit(bundle.x_train.reshape(len(bundle.x_train), -1), bundle.y_train_raw)
    dummy_inf_start = time.perf_counter()
    dummy_test_pred = dummy_final.predict(flat_test)
    dummy_inference_total_s = time.perf_counter() - dummy_inf_start
    ann_test_metrics = regression_metrics(bundle.y_test_raw, ann_test_pred)
    initial_test_metrics = regression_metrics(bundle.y_test_raw, initial_test_pred)
    baseline_test_metrics = regression_metrics(bundle.y_test_raw, baseline_test_pred)
    dummy_test_metrics = regression_metrics(bundle.y_test_raw, dummy_test_pred)
    test_timestamps = bundle.timestamps[bundle.test_target_index]
    input_missing_fraction = [float(bundle.raw_frame[BASE_FEATURES].iloc[j-bundle.x_test.shape[1]:j].isna().to_numpy().mean()) for j in bundle.test_target_index]
    predictions = pd.DataFrame({"timestamp": test_timestamps.astype(str), "actual": bundle.y_test_raw,
                                 "initial_ann_prediction": initial_test_pred, "ann_prediction": ann_test_pred,
                                 "baseline_prediction": baseline_test_pred, "training_mean_prediction": dummy_test_pred,
                                 "missing_input_fraction": input_missing_fraction,
                                 "ann_residual": bundle.y_test_raw - ann_test_pred,
                                 "baseline_residual": bundle.y_test_raw - baseline_test_pred})
    # Include only the final observed input row for optional GenAI handover notes.
    # The future actual target remains separate and is never sent to the LLM.
    for feature in BASE_FEATURES:
        predictions[f"input_last__{feature}"] = [bundle.raw_frame.iloc[int(j)-1][feature] for j in bundle.test_target_index]
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)
    plot_model_outputs(bundle, predictions)

    # Error slices: month and target-value quantile bands; do not use them to retune.
    predictions["month"] = pd.to_datetime(predictions["timestamp"]).dt.month
    predictions.groupby("month").apply(lambda g: pd.Series(regression_metrics(g["actual"], g["ann_prediction"])), include_groups=False).to_csv(TABLES_DIR / "error_by_month.csv")
    predictions["actual_band"] = pd.qcut(predictions["actual"], q=4, duplicates="drop")
    predictions.groupby("actual_band", observed=True).apply(lambda g: pd.Series(regression_metrics(g["actual"], g["ann_prediction"])), include_groups=False).to_csv(TABLES_DIR / "error_by_actual_band.csv")
    predictions["input_quality_band"] = pd.cut(predictions["missing_input_fraction"],
        bins=[-0.000001, 0.05, 0.20, 1.0], labels=["low_missingness", "moderate_missingness", "high_missingness"])
    predictions.groupby("input_quality_band", observed=True).apply(lambda g: pd.Series(regression_metrics(g["actual"], g["ann_prediction"])), include_groups=False).to_csv(TABLES_DIR / "error_by_input_quality.csv")
    worst_idx = int(np.argmax(np.abs(predictions["ann_residual"].to_numpy())))
    worst = predictions.iloc[worst_idx].to_dict()
    save_json(worst, TABLES_DIR / "largest_ann_test_error.json")

    print("8/8 Explainability, artifacts and report metrics...")
    ig, perm = run_xai(ann_model, bundle, selected["result"]["device"])
    worst_test_ig = integrated_gradients(ann_model, bundle.x_test[worst_idx], bundle.feature_names,
                                         steps=48, device=selected["result"]["device"])
    worst_test_ig.to_csv(TABLES_DIR / "largest_test_error_integrated_gradients.csv", index=False)
    for path, obj in [(MODELS_DIR / "preprocessor.joblib", bundle.preprocessor),
                      (MODELS_DIR / "target_scaler.joblib", bundle.target_scaler),
                      (MODELS_DIR / "baseline.joblib", baseline)]:
        joblib.dump(obj, path)
    torch.save(ann_model.state_dict(), MODELS_DIR / "ann_state_dict.pt")
    meta = {
        "project": "Air Quality Intelligence – Next-Hour Benzene Forecasting",
        "target": TARGET, "target_unit": "µg/m³", "window_size": int(selected["result"]["lookback"]),
        "horizon_hours": 1, "input_features": bundle.feature_names,
        "ann_config": {k: v for k, v in ann_config.items() if k != "name"},
        "ann_input_size": len(bundle.feature_names),
        "ann_parameter_count": int(sum(p.numel() for p in ann_model.parameters())),
        "ann_hidden_size": int(ann_config["hidden_size"]), "ann_num_layers": int(ann_config["num_layers"]),
        "ann_dropout": float(ann_config["dropout"]),
        "selected_experiment": ann_config["name"], "selected_seed": int(selected["seed"]),
        "selected_ann_best_epoch": int(selected["result"]["best_epoch"]),
        "initial_ann_best_epoch": int(initial_result["best_epoch"]),
        "validation_ann": regression_metrics(bundle.y_val_raw, ann_val_pred),
        "validation_baseline": baseline_val_metrics,
        "test_ann": ann_test_metrics, "test_initial_ann": initial_test_metrics,
        "test_baseline": baseline_test_metrics, "test_training_mean": dummy_test_metrics,
        "timing_seconds": {"selected_ann_fit": float(selected["result"]["training_seconds"]),
                            "initial_ann_fit": float(initial_result["training_seconds"]),
                            "baseline_fit": float(baseline_timing["fit_seconds"]),
                            "baseline_cv_search": float(baseline_timing["cv_search_seconds"]),
                            "ann_test_inference_per_sample_ms": float(ann_inference_total_s / len(bundle.x_test) * 1000),
                            "initial_ann_test_inference_per_sample_ms": float(initial_inference_total_s / len(bundle.x_test) * 1000),
                            "baseline_test_inference_per_sample_ms": float(baseline_inference_total_s / len(bundle.x_test) * 1000),
                            "training_mean_test_inference_per_sample_ms": float(dummy_inference_total_s / len(bundle.x_test) * 1000)},
        "validation_residual_std": float(np.std(bundle.y_val_raw - ann_val_pred, ddof=1)),
        "seed_summary": seed_summary,
        "baseline_params": baseline_params,
        "device_at_training": selected["result"]["device"],
        "test_evaluated_once_after_selection": True,
        "test_used_for_model_selection": False,
    }
    save_json(meta, MODELS_DIR / "model_metadata.json")
    save_json({"ANN": ann_test_metrics, "Initial_ANN": initial_test_metrics,
               "HistGradientBoosting": baseline_test_metrics, "Training_Mean": dummy_test_metrics,
               "validation_comparison": val_compare.to_dict(orient="records"), "ann_seed_summary": seed_summary,
               "timing_seconds": meta["timing_seconds"]}, TABLES_DIR / "final_metrics.json")
    comparison = pd.DataFrame([
        {"model": "Initial LSTM ANN", "split": "held-out test", "training_time_s": initial_result["training_seconds"], "inference_latency_ms_per_sample": initial_inference_total_s / len(bundle.x_test) * 1000, **initial_test_metrics},
        {"model": "Final LSTM ANN", "split": "held-out test", "training_time_s": selected["result"]["training_seconds"], "inference_latency_ms_per_sample": ann_inference_total_s / len(bundle.x_test) * 1000, **ann_test_metrics},
        {"model": "HistGradientBoosting", "split": "held-out test", "training_time_s": baseline_timing["fit_seconds"], "inference_latency_ms_per_sample": baseline_inference_total_s / len(bundle.x_test) * 1000, **baseline_test_metrics},
        {"model": "Training-mean reference", "split": "held-out test", "training_time_s": 0.0, "inference_latency_ms_per_sample": dummy_inference_total_s / len(bundle.x_test) * 1000, **dummy_test_metrics},
    ])
    comparison.to_csv(TABLES_DIR / "final_model_comparison.csv", index=False)

    # Keep separate detailed ANN log plus one consolidated log for the whole project.
    ann_log = ann_log.copy()
    initial_val_metrics = regression_metrics(bundle.y_val_raw, initial_val_pred)
    initial_log_row = {
        "experiment_id": "ANN-INITIAL-BASELINE", "seed": 42,
        **{k: v for k, v in initial_cfg.items() if k != "name"},
        "status": "completed", "best_epoch": int(initial_result["best_epoch"]),
        "training_seconds": float(initial_result["training_seconds"]),
        "val_mae": initial_val_metrics["MAE"], "val_rmse": initial_val_metrics["RMSE"],
        "val_r2": initial_val_metrics["R2"],
        "train_final_huber_loss": float(initial_result["history"][-1]["train_huber_loss"]),
        "selection_outcome": "initial_reference_model_not_final_selected",
        "experiment_type": "ANN", "final_selected": False,
    }
    ann_log = pd.concat([ann_log, pd.DataFrame([initial_log_row])], ignore_index=True, sort=False)
    ann_log["experiment_type"] = "ANN"
    ann_log["final_selected"] = ann_log["experiment_id"].eq(ann_config["name"])
    ann_log.to_csv(EXPERIMENTS_DIR / "ann_experiment_log.csv", index=False)
    baseline_log = baseline_search.copy()
    best_baseline_id = str(baseline_search.iloc[0]["experiment_id"])
    baseline_log["experiment_type"] = "Classical baseline"
    baseline_log["status"] = "completed"
    baseline_log["selection_outcome"] = np.where(baseline_log["experiment_id"].eq(best_baseline_id),
                                                   "selected_classical_baseline", "not_selected_cv_mae_higher")
    baseline_log["final_selected"] = baseline_log["experiment_id"].eq(best_baseline_id)
    combined_log = pd.concat([ann_log, baseline_log], ignore_index=True, sort=False)
    combined_log.to_csv(EXPERIMENTS_DIR / "experiment_log.csv", index=False)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_report.py")], check=True)
    print("PIPELINE COMPLETE")
    print("ANN test metrics:", ann_test_metrics)
    print("Baseline test metrics:", baseline_test_metrics)
    print("ANN validation seed summary:", seed_summary)
    print("Artifacts:", MODELS_DIR)


if __name__ == "__main__":
    main()
