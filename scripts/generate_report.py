#!/usr/bin/env python3
"""Render an evidence-based final project report from pipeline artefacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "FINAL_REPORT.md"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(v, digits=3):
    try: return f"{float(v):.{digits}f}"
    except Exception: return "NOT AVAILABLE – run the full pipeline"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--report", type=Path, default=OUT); args = p.parse_args()
    audit = read_json(ROOT / "outputs" / "tables" / "data_audit.json")
    metrics = read_json(ROOT / "outputs" / "tables" / "final_metrics.json")
    meta = read_json(ROOT / "models" / "model_metadata.json")
    leakage = read_json(ROOT / "outputs" / "tables" / "leakage_audit.json")
    exp_path = ROOT / "experiments" / "experiment_log.csv"
    exp = pd.read_csv(exp_path) if exp_path.exists() else None
    miss_path = ROOT / "outputs" / "tables" / "missingness.csv"
    miss = pd.read_csv(miss_path) if miss_path.exists() else None
    corr_path = ROOT / "outputs" / "tables" / "correlation_matrix.csv"
    corr = pd.read_csv(corr_path, index_col=0) if corr_path.exists() else None
    optimizer_path = ROOT / "experiments" / "optimizer_comparison.csv"
    optimizer = pd.read_csv(optimizer_path) if optimizer_path.exists() else None
    final_curve_path = ROOT / "experiments" / "ann_final_selected_learning_curve.csv"
    final_curve = pd.read_csv(final_curve_path) if final_curve_path.exists() else None
    largest_error = read_json(ROOT / "outputs" / "tables" / "largest_ann_test_error.json")
    worst_ig_path = ROOT / "outputs" / "tables" / "largest_test_error_integrated_gradients.csv"
    worst_ig = pd.read_csv(worst_ig_path) if worst_ig_path.exists() else None
    sanity_path = ROOT / "outputs" / "tables" / "xai_sanity_checks.csv"
    sanity = pd.read_csv(sanity_path) if sanity_path.exists() else None
    genai_eval_path = ROOT / "outputs" / "tables" / "genai_handover_evaluation.json"
    genai_eval = read_json(genai_eval_path)

    if audit:
        start, end = audit["date_start"], audit["date_end"]
        n, target_n, missing_pct = audit["rows_after_hourly_reindex"], audit["target_observed"], audit["target_missing_pct"]
        train_n, val_n, test_n = (audit["split_samples"][k] for k in ["train", "validation", "test"])
    else:
        start = end = "PENDING REAL-DATA RUN"
        n = target_n = train_n = val_n = test_n = "PENDING"
        missing_pct = None

    if miss is not None and len(miss):
        top_missing = ", ".join(f"{r.feature} ({r.missing_pct:.1f}%)" for r in miss.head(3).itertuples())
    else: top_missing = "PENDING – see outputs/tables/missingness.csv after running the pipeline"
    if corr is not None and "C6H6(GT)" in corr.columns:
        c = corr["C6H6(GT)"].drop("C6H6(GT)", errors="ignore").abs().sort_values(ascending=False).head(3)
        top_corr = ", ".join(f"{k} (|r|={v:.2f})" for k, v in c.items())
    else: top_corr = "PENDING – inspect correlation_matrix.csv; correlations are descriptive, not causal"

    if metrics and meta:
        ann = metrics["ANN"]; base = metrics["HistGradientBoosting"]
        initial = metrics.get("Initial_ANN", {})
        mean_ref = metrics.get("Training_Mean", {})
        timing = metrics.get("timing_seconds", {})
        metric_text = (f"| Training-mean reference | {fmt(mean_ref.get('MAE'))} | {fmt(mean_ref.get('RMSE'))} | {fmt(mean_ref.get('R2'))} | {fmt(timing.get('training_mean_test_inference_per_sample_ms'))} |\n"
                       f"| Initial LSTM ANN | {fmt(initial.get('MAE'))} | {fmt(initial.get('RMSE'))} | {fmt(initial.get('R2'))} | {fmt(timing.get('initial_ann_test_inference_per_sample_ms'))} |\n"
                       f"| Final LSTM ANN | {fmt(ann['MAE'])} | {fmt(ann['RMSE'])} | {fmt(ann['R2'])} | {fmt(timing.get('ann_test_inference_per_sample_ms'))} |\n"
                       f"| HistGradientBoosting | {fmt(base['MAE'])} | {fmt(base['RMSE'])} | {fmt(base['R2'])} | {fmt(timing.get('baseline_test_inference_per_sample_ms'))} |")
        config_text = json.dumps(meta["ann_config"], indent=2)
        selection = meta["selected_experiment"]
        seed_text = json.dumps(meta["seed_summary"], indent=2)
        best_params = json.dumps(meta["baseline_params"], indent=2)
        parameter_count = meta.get("ann_parameter_count", "PENDING")
        ann_fit = fmt(timing.get("selected_ann_fit"))
        base_fit = fmt(timing.get("baseline_fit"))
        best_epoch_text = meta.get("selected_ann_best_epoch", "PENDING")
    else:
        metric_text = "| Training-mean reference | PENDING | PENDING | PENDING | PENDING |\n| Initial LSTM ANN | PENDING | PENDING | PENDING | PENDING |\n| Final LSTM ANN | PENDING REAL-DATA RUN | PENDING | PENDING | PENDING |\n| HistGradientBoosting | PENDING REAL-DATA RUN | PENDING | PENDING | PENDING |"
        config_text = "PENDING REAL-DATA RUN"
        selection = seed_text = best_params = "PENDING REAL-DATA RUN"
        parameter_count = ann_fit = base_fit = best_epoch_text = "PENDING"
    if final_curve is not None and len(final_curve):
        best_row = final_curve.loc[final_curve["val_mae_scaled"].idxmin()]
        last_row = final_curve.iloc[-1]
        if float(last_row["val_mae_scaled"]) > float(best_row["val_mae_scaled"]) * 1.05 and float(last_row["train_huber_loss"]) < float(best_row["train_huber_loss"]):
            curve_interpretation = (f"Validation MAE reached its minimum of {fmt(best_row['val_mae_scaled'])} (standardised target units) at epoch {int(best_row['epoch'])}, then rose to {fmt(last_row['val_mae_scaled'])} by epoch {int(last_row['epoch'])} while training loss fell. This is evidence of later overfitting; early stopping retained the best validation checkpoint.")
        else:
            curve_interpretation = (f"Validation MAE reached its minimum of {fmt(best_row['val_mae_scaled'])} (standardised target units) at epoch {int(best_row['epoch'])}. The saved selected checkpoint is the best validation epoch; inspect `outputs/figures/ann_learning_curves.png` for any train/validation divergence.")
    else:
        curve_interpretation = "PENDING REAL-DATA RUN – inspect `outputs/figures/ann_learning_curves.png` and the learning-curve CSVs."

    if optimizer is not None and len(optimizer):
        optimizer_text = "\n".join(f"| {r.optimizer} | {int(r['count'])} | {fmt(r['mean'])} | {fmt(r['std'])} |" for _, r in optimizer.iterrows())
        opt_names = optimizer.sort_values("mean")["optimizer"].tolist()
        optimizer_interpretation = (f"The lower mean validation MAE among these tested configurations was observed for **{opt_names[0]}**. "
            "AdamW uses decoupled weight decay with adaptive first/second-moment estimates; RMSprop adapts the learning rate using a moving average of squared gradients. "
            "Because the search is small and configurations are not perfectly paired, this is an empirical indication for this search budget, not a universal optimiser ranking.")
    else:
        optimizer_text = "| AdamW | PENDING | PENDING | PENDING |\n| RMSprop | PENDING | PENDING | PENDING |"
        optimizer_interpretation = "PENDING REAL-DATA RUN – compare measured validation MAE, spread and convergence curves for AdamW and RMSprop."

    if exp is not None and "selection_outcome" in exp:
        ann_rows = exp[exp.get("experiment_type", pd.Series(index=exp.index, dtype=str)) == "ANN"] if "experiment_type" in exp else exp
        fail_count = int(ann_rows["selection_outcome"].astype(str).str.contains("not_selected|unsuccessful", regex=True).sum())
        log_status = f"{len(ann_rows)} ANN experiments recorded; {fail_count} not selected or unsuccessful. The consolidated log also contains classical baseline search rows."
        diagnostic_rows = ann_rows[ann_rows["experiment_id"].astype(str).str.startswith("ANN-FAIL-")]
        if len(diagnostic_rows):
            cols = [c for c in ["experiment_id", "lookback", "hidden_size", "num_layers", "dropout", "learning_rate", "batch_size", "optimizer", "status", "best_epoch", "val_mae", "selection_outcome"] if c in diagnostic_rows]
            diagnostic_table = diagnostic_rows[cols].to_markdown(index=False, floatfmt=".4f")
        else:
            diagnostic_table = "No diagnostic attempts recorded in the consolidated experiment log."
    else:
        log_status = "No real-data experiment records yet. Run scripts/run_pipeline.py."
        diagnostic_table = "PENDING REAL-DATA RUN"

    parameter_effect_sections = []
    for parameter, label in [("learning_rate", "Learning rate"), ("batch_size", "Batch size"),
                             ("num_layers", "Recurrent depth"), ("dropout", "Dropout"),
                             ("hidden_size", "Hidden width")]:
        path = ROOT / "experiments" / f"effect_{parameter}.csv"
        if path.exists():
            table = pd.read_csv(path)
            if len(table):
                value_col = parameter if parameter in table.columns else table.columns[0]
                table = table.sort_values("mean")
                rendered = table[[value_col, "count", "mean", "std", "min"]].to_markdown(index=False, floatfmt=".4f")
                best_group = table.iloc[0]
                conclusion = (f"Within the sampled configurations, **{value_col}={best_group[value_col]}** had the lowest mean validation MAE ({fmt(best_group['mean'], 4)}). "
                              "This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.")
                parameter_effect_sections.append(f"#### {label}\n\n{rendered}\n\n{conclusion}")
    parameter_effect_text = "\n\n".join(parameter_effect_sections) if parameter_effect_sections else "PENDING REAL-DATA RUN – parameter effect tables will be created by the ANN search."

    if largest_error and worst_ig is not None and len(worst_ig):
        top_features = ", ".join(f"{r.feature} (|IG|={r.absolute_attribution:.4f})" for r in worst_ig.head(5).itertuples())
        worst_error_text = (f"The largest absolute ANN error occurred at **{largest_error.get('timestamp', 'unknown timestamp')}**. "
                            f"Actual benzene was **{fmt(largest_error.get('actual'))} µg/m³**, the forecast was **{fmt(largest_error.get('ann_prediction'))} µg/m³**, "
                            f"and absolute error was **{fmt(abs(float(largest_error.get('ann_residual', 0))))} µg/m³**. "
                            f"Top absolute Integrated Gradients features for this case were: {top_features}. "
                            "These indicate what the model relied on, not the physical cause of the error.")
    else:
        worst_error_text = "PENDING REAL-DATA RUN – run the test error analysis to identify and explain the largest ANN error."

    if sanity is not None and len(sanity):
        sanity_cols = [c for c in ["case", "feature_perturbed", "lag_hours_before_forecast", "prediction_change", "ig_attribution_for_perturbed_coordinate", "direction_consistent_with_ig"] if c in sanity.columns]
        sanity_text = sanity[sanity_cols].to_markdown(index=False, floatfmt=".4f")
    else:
        sanity_text = "PENDING REAL-DATA RUN – two local perturbation checks will be written by the pipeline."
    if leakage:
        leakage_text = json.dumps(leakage, indent=2)
    else: leakage_text = "PENDING REAL-DATA RUN"

    if isinstance(genai_eval, list) and genai_eval:
        genai_models = sorted({str(x.get("model", "unknown")) for x in genai_eval})
        genai_latency = [float(x["latency_seconds"]) for x in genai_eval if x.get("latency_seconds") is not None]
        genai_tokens = [int(x["total_tokens"]) for x in genai_eval if x.get("total_tokens") is not None]
        genai_costs = [float(x["estimated_cost_usd"]) for x in genai_eval if x.get("estimated_cost_usd") is not None]
        genai_evaluation_text = (
            f"The optional adjacent-task evaluation generated notes for **{len(genai_eval)} held-out test cases** using model(s) **{', '.join(genai_models)}**. "
            f"Mean API latency was **{fmt(pd.Series(genai_latency).mean())} seconds** and mean usage was **{fmt(pd.Series(genai_tokens).mean(), 1)} tokens/case**. "
            + (f"Estimated total cost was **${sum(genai_costs):.6f} USD** using the configured per-token rates. " if genai_costs else "Cost was not estimated because current token-price rates were not configured. ")
            + "Human quality ratings are recorded separately in `outputs/tables/genai_human_review_template.csv`; complete them before discussing text quality."
        )
    else:
        genai_evaluation_text = (
            "The optional LLM was not called in the initial build. To evaluate it, obtain lecturer approval, set `OPENAI_API_KEY` and a currently supported `GENAI_MODEL`, then run `python scripts/evaluate_genai.py --n 5`. "
            "It will use evenly spaced examples from the same held-out test prediction file, never send the future actual benzene values, record tokens/latency/cost where configured, and create a manual quality-review template."
        )

    report = f'''# Air Quality Intelligence: Next-Hour Benzene Concentration Forecasting

**Student:** [Insert name and student ID]  
**Module:** [Insert module]  
**Dataset:** UCI Air Quality, dataset ID 360  
**Problem type:** Multivariate time-series regression  
**Primary metric:** Mean Absolute Error (MAE), in µg/m³  
**Report status:** {'Generated from completed pipeline outputs' if metrics else 'PRE-RUN REPORT SCAFFOLD — real-data metrics have not yet been generated'}

> **Evidence rule:** This report only inserts scores calculated by `scripts/run_pipeline.py`. No synthetic smoke-test score is presented as a UCI result. Any item marked PENDING must be completed after downloading the official dataset and running the pipeline.

## Executive summary

This project develops an LSTM artificial neural network (ANN) to forecast the next hourly benzene concentration (`C6H6(GT)`) from the preceding 24 hours of low-cost gas-sensor responses and meteorological observations. The intended user is an environmental monitoring station operator. The forecast is intended to help prioritise a station inspection or internal air-quality review; it is not a regulatory compliance determination or a health-risk assessment.

The input deliberately excludes the five reference-analyser pollutant columns (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`). This avoids using laboratory measurements that would not be available at prediction time. The target is the reference-analyser benzene measurement one hour after the final input observation.

## 1. Dataset and exploratory data analysis (Question 1)

### 1.1 Source, permission and structure

The dataset is the UCI Machine Learning Repository Air Quality dataset (ID 360), attributed to Saverio Vito (2008), DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F). UCI describes 9,358 hourly observations from an outdoor roadside deployment in an Italian city over March 2004–February 2005, with five metal-oxide sensor responses and reference-analyser pollutant concentrations. Missing values are encoded as `-200`.

The landing page simultaneously states that the dataset is for research use and excludes commercial use, while its licence field states CC BY 4.0. This project therefore treats the data as **non-commercial research/education only** unless UCI confirms otherwise. The submitted code includes attribution and the official download procedure.

### 1.2 Data summary

- Time coverage: **{start} to {end}**
- Hourly rows after continuity reindexing: **{n}**
- Observed benzene targets: **{target_n}**
- Benzene target missingness: **{fmt(missing_pct, 2) if missing_pct is not None else 'PENDING'}%**
- Valid target windows: train **{train_n}**, validation **{val_n}**, test **{test_n}**
- Sensor/weather inputs: {', '.join(['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])}
- Excluded inputs: all reference-analyser pollutant concentrations.

Detailed descriptive statistics and missingness are in `outputs/tables/descriptive_statistics.csv` and `outputs/tables/missingness.csv`. The figures are in `outputs/figures/`.

### 1.3 EDA findings and modelling consequences

1. **Missing-value structure:** the three most incomplete columns are **{top_missing}**. All `-200` markers are converted to missing values. Input values are median-imputed using training-period statistics only, and missingness indicators are retained. Missing targets are not imputed; affected forecast examples are excluded.
2. **Sensor-target relationships:** the largest absolute descriptive correlations with benzene are **{top_corr}**. These are associations only and do not establish causality. The sensor channels provide a defensible signal for regression, but cross-sensitivity and sensor drift constrain transferability.
3. **Temporal structure and drift:** hourly ordering and timestamp continuity are preserved. The target time series and monthly summaries are examined before modelling. Sensor drift and concept drift documented by UCI motivate chronological rather than random splitting, and make a historical single-station model unsuitable for unvalidated deployment at a different site.

### 1.4 Decision losses and metric

The primary metric is MAE. An MAE of *x* µg/m³ means the forecast is wrong by *x* µg/m³ on average in absolute terms. This is more directly interpretable to an operator than a percentage error, especially where target values may approach zero. Underprediction could delay a site review; overprediction could consume staff time on unnecessary checks. Because no approved operational threshold or asymmetric cost ratio is supplied, the project does not invent one and uses MAE as the primary symmetric error measure. RMSE is secondary because it penalises large errors, and R² is reported as a supplementary goodness-of-fit statistic. No MAPE is used because near-zero targets make it unstable.

### 1.5 Leakage and data-quality checks

The leakage audit confirms that the target row is never in its own input window, all inputs precede the target by one hour, reference-analyser pollutant columns are excluded, the preprocessor is fitted only on the training period, and the test split is reserved for the final evaluation. Full machine-readable results: `outputs/tables/leakage_audit.json`.

```json
{leakage_text}
```

## 2. Solution design and alternatives (Question 2)

### 2.1 ANN architecture

The selected model is an LSTM-based recurrent ANN. The input is a tensor of shape `(batch, 24, 22)`: a 24-hour lookback, with eight sensor/weather measurements, eight corresponding missingness indicators, and six cyclical time variables. The LSTM encodes temporal dependencies; the final hidden state is normalised and passed through a dense GELU/dropout regression head to one continuous benzene forecast. The output is a single real number in µg/m³ after inverse target scaling.

Selected configuration:

```json
{config_text}
```

The training objective is Huber loss on a standardised target, with a linear final output so the model can predict a continuous real value. `StandardScaler` is fitted only on training inputs and training targets. Gradient clipping is used for recurrent-training stability. Early stopping is based on validation MAE in standardised target space. The selected ANN has **{parameter_count} trainable parameters**. Parameter count is calculated from the actual model state and can be verified from the saved state dictionary.

Transfer learning was considered. The dataset is a small, single-site multivariate sensor sequence with a domain-specific set of eight measurements; no pretrained backbone with matching channels, units and acquisition context is assumed in this project. Fine-tuning a mismatched generic model would not guarantee a sensible transfer. The compact task-specific LSTM is therefore trained from scratch, and this limitation is disclosed rather than claiming a nonexistent pretraining source.

### 2.2 Alternative approaches

- **HistGradientBoostingRegressor (selected classical baseline):** handles nonlinear tabular interactions and is competitive on small structured data. It uses the same historical windows, same splits and same primary metric. Its flattened-window representation is less explicitly temporal than the LSTM.
- **Random Forest / XGBoost:** reasonable tree-based comparators, but a second tree family adds complexity without changing the key requirement to compare ANN against a tuned classical learner. They may be added as sensitivity analyses.
- **MLP:** a dense neural model on flattened windows is a useful ANN alternative but discards explicit sequence structure. LSTM is preferred because the task is sequential.
- **ARIMA / exponential smoothing:** useful univariate baselines, but they do not naturally use the multivariate low-cost sensor and weather channels; they would be weaker comparators for the stated sensor-only task.

The project selects the LSTM architecture because the forecast depends on temporal patterns and a multivariate history. **Before inspecting final results, the prior expectation is that the tuned tree baseline may be competitive or even stronger** because there are relatively few labelled windows and the sensor inputs are structured numeric measurements; the LSTM may benefit from sequence ordering but has more parameters to estimate. This is a modelling hypothesis, not a claim that either family must win.

## 3. Model development, comparison and optimisation (Question 3)

### 3.1 Split discipline and tuning budget

A single chronological split is used: first 70% training, next 15% validation, final 15% test. Training windows have a 24-hour lookback and target the next hour. No random shuffle is used to create the split. Hyperparameters, early stopping, feature preprocessing and model selection use training/validation information only. The test set is evaluated once after the design is frozen. The selected ANN checkpoint was retained at **epoch {best_epoch_text}**. {curve_interpretation}

The classical baseline uses a randomised search over learning rate, iteration count, leaf nodes, minimum leaf size and L2 regularisation with three expanding-window `TimeSeriesSplit` folds, with imputation/scaling refitted separately inside each fold. Approximate baseline search size: **10 configurations × 3 folds** (configurable). The ANN search includes four diagnostic/weak attempts plus eight structured configurations. These investigate lookback length, hidden width, depth, dropout, learning rate, batch size and optimiser (AdamW versus RMSprop). This is a comparable order of tuning effort and is fully recorded in `experiments/`.

Empirical optimiser comparison on validation data (lower MAE is better):

| Optimiser | Completed configurations | Mean validation MAE | Standard deviation |
|---|---:|---:|---:|
{optimizer_text}

{optimizer_interpretation}

The following tables summarise at least three hyperparameters empirically. Mean validation MAE is aggregated over completed configurations holding the other sampled settings as they happened to occur; because the search is not a full factorial design, these are exploratory comparisons rather than isolated causal effects.

{parameter_effect_text}

### 3.2 Experiment log and failed attempts

**Current experiment status:** {log_status}

The log records each configuration, seed, best epoch, training time, validation score, failure reason and selection outcome. The diagnostic attempts probe short context, excessive learning rate, under-capacity/over-regularisation, and excessive depth/dropout. Training can complete successfully while the modelling attempt is unsuccessful because it fails to improve validation MAE. The measured outcomes are shown below; do not describe a run as unsuccessful unless the log confirms that it failed or was not selected.

{diagnostic_table}

### 3.3 Final held-out test results

| Model | Test MAE (µg/m³) | Test RMSE (µg/m³) | Test R² | Inference latency (ms/sample) |
|---|---:|---:|---:|---:|
{metric_text}

The selected ANN fit time was **{ann_fit} seconds**; the classical model fit time after cross-validation was **{base_fit} seconds**. Full search time is retained in `model_metadata.json`.

The ANN configuration is selected using outer-validation MAE, while the classical baseline hyperparameters are selected using fold-specific training-only cross-validation. Their validation metrics provide the pre-test family comparison. Both frozen models are then evaluated once on the same held-out test period; no model or threshold is changed using those test results. The ANN is deployed in the required application even if the classical model is stronger; in that event, the report must state the result honestly and identify the classical result as a warning against assuming ANN superiority.

Selected baseline parameters:

```json
{best_params}
```

### 3.4 Random-seed stability

Three seeds are fixed and reported (42, 7, 2026). The selected ANN configuration is repeated three times. The median validation run is selected to avoid relying on the luckiest seed. Validation mean/spread:

```json
{seed_text}
```

### 3.5 Error analysis

The pipeline produces residual plots, test predictions, error by month, actual-concentration band and input missingness band. `outputs/tables/largest_ann_test_error.json` and `largest_test_error_integrated_gradients.csv` provide a concrete incorrect-prediction case and its model attributions:

{worst_error_text}

Do not infer that a model-attributed input physically caused the miss. Review the month, concentration band and missing-input tables alongside the plots.

## 4. Web application implementation (Question 4)

A Streamlit web application is provided in `app/streamlit_app.py`. The user uploads a CSV containing at least 24 consecutive hourly records with the five sensor channels and three weather fields. The application validates timestamps and required columns, uses the saved training-only preprocessor, flags readings outside the training-period 1st–99th percentile ranges, loads the trained LSTM and generates a live request-time forecast. It displays the forecast, the historical validation MAE and an Integrated Gradients explanation. It rejects irregular timestamps and insufficient input rows. Reference-analyser pollutant columns are neither required nor used.

The application has no hard-coded predictions. The reported validation MAE is a typical validation error, not a calibrated prediction interval. The page states model limitations and non-regulatory status. To run: `streamlit run app/streamlit_app.py`.

## 5. Explainable AI (Question 5)

### 5.1 Comparison of three approaches

| Method | Family | What it explains | Strengths | Limitations |
|---|---|---|---|---|
| Integrated Gradients | Architecture-specific | Contribution of each input channel/lag relative to a baseline for one forecast | Uses ANN gradients; local; supports temporal channels; cost is roughly one forward/backward evaluation per integration step | Baseline choice matters; can be unstable in saturated regions; not causal |
| Grouped permutation importance | Model-agnostic | Increase in validation MAE when one entire input channel is permuted across cases | Directly measures predictive reliance; easy to explain; cost scales with features × repeats × validation predictions | Correlated features can share importance; permutation can create unrealistic combinations |
| SHAP KernelExplainer | Model-agnostic | Shapley-style feature attributions for a prediction | Unified additive attribution framework; can explain any predictor | Highest cost for hundreds of lagged inputs because many masked model evaluations are required; background distribution and feature dependence matter |

Integrated Gradients is selected for the interactive app because it is suitable for the LSTM and provides local, time-aware attribution. Grouped permutation importance supplies a model-agnostic global cross-check. SHAP is discussed as an alternative; its computational cost and feature-dependence limitations make it a less practical default for this 24 × 22 window.

### 5.2 Explanation sanity checks

The pipeline creates two local perturbation checks in `outputs/tables/xai_sanity_checks.csv`. A top-attributed input is perturbed slightly and the forecast change is observed. The results are:

{sanity_text}

A mismatch is reported rather than hidden. These local perturbations are a sanity check, not a causal test. The largest-error test case is also inspected above.

Attributions describe model behaviour relative to a baseline. They do not prove causal reasoning, nor do they guarantee that the attributed variable was the true cause of a forecast.

## 6. Generative AI / foundation model (Question 6)

### 6.1 Adjacent-task application

A generative language model is not used to produce the numeric benzene forecast. The project uses an optional pretrained instruction-following LLM, configured explicitly through `GENAI_MODEL` (set a currently supported identifier for the connected API account), to turn the already-computed ANN forecast and recent sensor context into a short shift-handover note. This adjacent task is more meaningful than asking a general-purpose LLM to regress a small multivariate sensor sequence without a specialised time-series interface. The language model does not change the ANN output.

The adapter is implemented in `src/airquality/genai.py` using the OpenAI Responses API; the fixed prompt is `prompts/shift_handover_v1.txt`. It requires an API key and incurs provider-dependent cost. The exact model identifier, token usage and latency are saved when run. Optional per-million-token input/output price variables calculate an estimated cost; without current rates, the result explicitly reports cost as unavailable. No LLM call is made by default. Obtain lecturer approval for this adjacent task before treating it as the assessed GenAI component.

{genai_evaluation_text}

### 6.2 Comparative analysis

The ANN is the numerical forecasting component: it requires labelled time-series examples, uses a training/validation/test evaluation, and is measured with MAE/RMSE/R². The LLM is a language-generation component: it receives the forecast and bounded context and produces a human-readable handover note. Since it is not asked to predict concentration, it is not scientifically fair to compare the LLM's text output with ANN regression MAE. Compare the approaches by purpose, data requirements, latency, provider cost, privacy and failure modes. A fair predictive comparison would require an approved time-series foundation model that receives only the same permissible sensor information and is scored on the same held-out test targets and MAE.

The language model may extend accessibility and handover quality, but it is not suitable for authoritative regulatory decisions. It can omit context or generate unsupported statements; the fixed prompt instructs it not to invent causes, thresholds or safety claims.

## 7. Limitations, ethics and deployment considerations

1. The data represent a single roadside site and a historical period in 2004–2005; performance cannot be assumed to transfer to another country, climate, sensor array or current urban environment.
2. Sensor drift, cross-sensitivity, missing values and possible concept drift affect reliability.
3. Benzene reference readings are used as training labels; a certified reference analyser is still needed to maintain trustworthy ground truth.
4. The output is an estimate, not an official AQI, regulatory threshold determination or medical recommendation.
5. The UCI landing page states commercial use is excluded; do not use the dataset commercially without written clarification/permission.
6. The optional hosted LLM sends forecast/context to an external provider; do not send personal or sensitive data. The numeric prediction is generated locally by the trained ANN.

## 8. Reproducibility and file map

- `scripts/download_data.py`: downloads official UCI archive and writes a SHA-256 manifest.
- `scripts/run_pipeline.py`: preprocessing, EDA, leakage audit, tuning, training, test evaluation and XAI.
- `src/airquality/`: reusable dataset, model, metric, inference and explainability modules.
- `experiments/`: complete ANN and baseline search logs and learning curves.
- `models/`: trained model and preprocessing artefacts.
- `outputs/`: EDA figures, metrics, predictions, error analysis and XAI outputs.
- `app/streamlit_app.py`: interactive request-time prediction app.
- `tests/`: unit and smoke tests.
- `docs/`: experiment protocol, defence guide, demo script and submission checklist.

Reproduction commands:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_pipeline.py
python scripts/generate_report.py
pytest -q
streamlit run app/streamlit_app.py
```

## 9. Eight-minute demonstration video and live defence

Use the supplied `docs/demo_script_8min.md`. The demo should show the input upload, actual request-time forecast, explanation, model comparison, evidence log and limitations. Do not demonstrate with hard-coded values or present synthetic smoke-test outputs as real UCI results.

## References

1. Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F.
2. De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical*, 129(2). [See UCI dataset citation page for the paper record.]
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of ICML*.
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*.
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__": main()
