# Air Quality Intelligence: Next-Hour Benzene Concentration Forecasting

**Student:** [Insert name and student ID]  
**Module:** [Insert module]  
**Dataset:** UCI Air Quality, dataset ID 360  
**Problem type:** Multivariate time-series regression  
**Primary metric:** Mean Absolute Error (MAE), in µg/m³  
**Report status:** Generated from completed pipeline outputs

> **Evidence rule:** This report only inserts scores calculated by `scripts/run_pipeline.py`. No synthetic smoke-test score is presented as a UCI result. Any item marked PENDING must be completed after downloading the official dataset and running the pipeline.

## Executive summary

This project develops an LSTM artificial neural network (ANN) to forecast the next hourly benzene concentration (`C6H6(GT)`) from the preceding 24 hours of low-cost gas-sensor responses and meteorological observations. The intended user is an environmental monitoring station operator. The forecast is intended to help prioritise a station inspection or internal air-quality review; it is not a regulatory compliance determination or a health-risk assessment.

The input deliberately excludes the five reference-analyser pollutant columns (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`). This avoids using laboratory measurements that would not be available at prediction time. The target is the reference-analyser benzene measurement one hour after the final input observation.

## 1. Dataset and exploratory data analysis (Question 1)

### 1.1 Source, permission and structure

The dataset is the UCI Machine Learning Repository Air Quality dataset (ID 360), attributed to Saverio Vito (2008), DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F). UCI describes 9,358 hourly observations from an outdoor roadside deployment in an Italian city over March 2004–February 2005, with five metal-oxide sensor responses and reference-analyser pollutant concentrations. Missing values are encoded as `-200`.

The landing page simultaneously states that the dataset is for research use and excludes commercial use, while its licence field states CC BY 4.0. This project therefore treats the data as **non-commercial research/education only** unless UCI confirms otherwise. The submitted code includes attribution and the official download procedure.

### 1.2 Data summary

- Time coverage: **2004-03-10 18:00:00 to 2005-04-04 14:00:00**
- Hourly rows after continuity reindexing: **9357**
- Observed benzene targets: **8991**
- Benzene target missingness: **3.91%**
- Valid target windows: train **6377**, validation **1263**, test **1327**
- Sensor/weather inputs: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
- Excluded inputs: all reference-analyser pollutant concentrations.

Detailed descriptive statistics and missingness are in `outputs/tables/descriptive_statistics.csv` and `outputs/tables/missingness.csv`. The figures are in `outputs/figures/`.

### 1.3 EDA findings and modelling consequences

1. **Missing-value structure:** the three most incomplete columns are **NMHC(GT) (90.2%), CO(GT) (18.0%), NO2(GT) (17.5%)**. All `-200` markers are converted to missing values. Input values are median-imputed using training-period statistics only, and missingness indicators are retained. Missing targets are not imputed; affected forecast examples are excluded.
2. **Sensor-target relationships:** the largest absolute descriptive correlations with benzene are **PT08.S2(NMHC) (|r|=0.98), CO(GT) (|r|=0.93), NMHC(GT) (|r|=0.90)**. These are associations only and do not establish causality. The sensor channels provide a defensible signal for regression, but cross-sensitivity and sensor drift constrain transferability.
3. **Temporal structure and drift:** hourly ordering and timestamp continuity are preserved. The target time series and monthly summaries are examined before modelling. Sensor drift and concept drift documented by UCI motivate chronological rather than random splitting, and make a historical single-station model unsuitable for unvalidated deployment at a different site.

### 1.4 Decision losses and metric

The primary metric is MAE. An MAE of *x* µg/m³ means the forecast is wrong by *x* µg/m³ on average in absolute terms. This is more directly interpretable to an operator than a percentage error, especially where target values may approach zero. Underprediction could delay a site review; overprediction could consume staff time on unnecessary checks. Because no approved operational threshold or asymmetric cost ratio is supplied, the project does not invent one and uses MAE as the primary symmetric error measure. RMSE is secondary because it penalises large errors, and R² is reported as a supplementary goodness-of-fit statistic. No MAPE is used because near-zero targets make it unstable.

### 1.5 Leakage and data-quality checks

The leakage audit confirms that the target row is never in its own input window, all inputs precede the target by one hour, reference-analyser pollutant columns are excluded, the preprocessor is fitted only on the training period, and the test split is reserved for the final evaluation. Full machine-readable results: `outputs/tables/leakage_audit.json`.

```json
{
  "reference_analyser_pollutant_inputs": [],
  "target_row_excluded_from_input_window": true,
  "forecast_horizon_hours": 1,
  "lookback_hours": 24,
  "chronological_split": true,
  "preprocessor_fit_rows": 6548,
  "preprocessor_max_fit_row_index": 6547,
  "preprocessor_uses_only_training_input_rows": true,
  "test_used_for_model_selection": false,
  "train_max_feature_row_index": 6547,
  "validation_min_target_row_index": 6549,
  "validation_max_feature_row_index": 7951,
  "test_min_target_row_index": 7953,
  "pass": true
}
```

## 2. Solution design and alternatives (Question 2)

### 2.1 ANN architecture

The selected model is an LSTM-based recurrent ANN. The input is a tensor of shape `(batch, 24, 22)`: a 24-hour lookback, with eight sensor/weather measurements, eight corresponding missingness indicators, and six cyclical time variables. The LSTM encodes temporal dependencies; the final hidden state is normalised and passed through a dense GELU/dropout regression head to one continuous benzene forecast. The output is a single real number in µg/m³ after inverse target scaling.

Selected configuration:

```json
{
  "lookback": 24,
  "hidden_size": 96,
  "num_layers": 1,
  "dropout": 0.35,
  "learning_rate": 0.0005,
  "batch_size": 128,
  "optimizer": "RMSprop"
}
```

The training objective is Huber loss on a standardised target, with a linear final output so the model can predict a continuous real value. `StandardScaler` is fitted only on training inputs and training targets. Gradient clipping is used for recurrent-training stability. Early stopping is based on validation MAE in standardised target space. The selected ANN has **50977 trainable parameters**. Parameter count is calculated from the actual model state and can be verified from the saved state dictionary.

Transfer learning was considered. The dataset is a small, single-site multivariate sensor sequence with a domain-specific set of eight measurements; no pretrained backbone with matching channels, units and acquisition context is assumed in this project. Fine-tuning a mismatched generic model would not guarantee a sensible transfer. The compact task-specific LSTM is therefore trained from scratch, and this limitation is disclosed rather than claiming a nonexistent pretraining source.

### 2.2 Alternative approaches

- **HistGradientBoostingRegressor (selected classical baseline):** handles nonlinear tabular interactions and is competitive on small structured data. It uses the same historical windows, same splits and same primary metric. Its flattened-window representation is less explicitly temporal than the LSTM.
- **Random Forest / XGBoost:** reasonable tree-based comparators, but a second tree family adds complexity without changing the key requirement to compare ANN against a tuned classical learner. They may be added as sensitivity analyses.
- **MLP:** a dense neural model on flattened windows is a useful ANN alternative but discards explicit sequence structure. LSTM is preferred because the task is sequential.
- **ARIMA / exponential smoothing:** useful univariate baselines, but they do not naturally use the multivariate low-cost sensor and weather channels; they would be weaker comparators for the stated sensor-only task.

The project selects the LSTM architecture because the forecast depends on temporal patterns and a multivariate history. **Before inspecting final results, the prior expectation is that the tuned tree baseline may be competitive or even stronger** because there are relatively few labelled windows and the sensor inputs are structured numeric measurements; the LSTM may benefit from sequence ordering but has more parameters to estimate. This is a modelling hypothesis, not a claim that either family must win.

## 3. Model development, comparison and optimisation (Question 3)

### 3.1 Split discipline and tuning budget

A single chronological split is used: first 70% training, next 15% validation, final 15% test. Training windows have a 24-hour lookback and target the next hour. No random shuffle is used to create the split. Hyperparameters, early stopping, feature preprocessing and model selection use training/validation information only. The test set is evaluated once after the design is frozen. The selected ANN checkpoint was retained at **epoch 17**. Validation MAE reached its minimum of 0.259 (standardised target units) at epoch 17. The saved selected checkpoint is the best validation epoch; inspect `outputs/figures/ann_learning_curves.png` for any train/validation divergence.

The classical baseline uses a randomised search over learning rate, iteration count, leaf nodes, minimum leaf size and L2 regularisation with three expanding-window `TimeSeriesSplit` folds, with imputation/scaling refitted separately inside each fold. Approximate baseline search size: **10 configurations × 3 folds** (configurable). The ANN search includes four diagnostic/weak attempts plus eight structured configurations. These investigate lookback length, hidden width, depth, dropout, learning rate, batch size and optimiser (AdamW versus RMSprop). This is a comparable order of tuning effort and is fully recorded in `experiments/`.

Empirical optimiser comparison on validation data (lower MAE is better):

| Optimiser | Completed configurations | Mean validation MAE | Standard deviation |
|---|---:|---:|---:|
| AdamW | 8 | 2.338 | 0.834 |
| RMSprop | 4 | 2.027 | 0.045 |

The lower mean validation MAE among these tested configurations was observed for **RMSprop**. AdamW uses decoupled weight decay with adaptive first/second-moment estimates; RMSprop adapts the learning rate using a moving average of squared gradients. Because the search is small and configurations are not perfectly paired, this is an empirical indication for this search budget, not a universal optimiser ranking.

The following tables summarise at least three hyperparameters empirically. Mean validation MAE is aggregated over completed configurations holding the other sampled settings as they happened to occur; because the search is not a full factorial design, these are exploratory comparisons rather than isolated causal effects.

#### Learning rate

|   learning_rate |   count |   mean |      std |    min |
|----------------:|--------:|-------:|---------:|-------:|
|          0.0005 |  2.0000 | 1.9935 |   0.0156 | 1.9825 |
|          0.0010 |  5.0000 | 2.0268 |   0.0374 | 2.0013 |
|          0.0030 |  2.0000 | 2.0414 |   0.0006 | 2.0410 |
|          0.0003 |  1.0000 | 2.0826 | nan      | 2.0826 |
|          0.0300 |  1.0000 | 2.1228 | nan      | 2.1228 |
|          0.0001 |  1.0000 | 4.3999 | nan      | 4.3999 |

Within the sampled configurations, **learning_rate=0.0005** had the lowest mean validation MAE (1.9935). This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.

#### Batch size

|   batch_size |   count |   mean |    std |    min |
|-------------:|--------:|-------:|-------:|-------:|
|      64.0000 |  5.0000 | 2.0182 | 0.0212 | 2.0013 |
|     128.0000 |  5.0000 | 2.0472 | 0.0580 | 1.9825 |
|     256.0000 |  2.0000 | 3.2412 | 1.6386 | 2.0826 |

Within the sampled configurations, **batch_size=64.0** had the lowest mean validation MAE (2.0182). This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.

#### Recurrent depth

|   num_layers |   count |   mean |      std |    min |
|-------------:|--------:|-------:|---------:|-------:|
|       2.0000 |  6.0000 | 2.0409 |   0.0439 | 2.0013 |
|       3.0000 |  1.0000 | 2.0826 | nan      | 2.0826 |
|       1.0000 |  5.0000 | 2.4963 |   1.0649 | 1.9825 |

Within the sampled configurations, **num_layers=2.0** had the lowest mean validation MAE (2.0409). This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.

#### Dropout

|   dropout |   count |   mean |      std |    min |
|----------:|--------:|-------:|---------:|-------:|
|    0.3500 |  1.0000 | 1.9825 | nan      | 1.9825 |
|    0.1500 |  1.0000 | 2.0013 | nan      | 2.0013 |
|    0.1000 |  2.0000 | 2.0220 |   0.0279 | 2.0023 |
|    0.2500 |  1.0000 | 2.0338 | nan      | 2.0338 |
|    0.3000 |  1.0000 | 2.0410 | nan      | 2.0410 |
|    0.2000 |  4.0000 | 2.0560 |   0.0594 | 2.0046 |
|    0.6000 |  1.0000 | 2.0826 | nan      | 2.0826 |
|    0.7000 |  1.0000 | 4.3999 | nan      | 4.3999 |

Within the sampled configurations, **dropout=0.35** had the lowest mean validation MAE (1.9825). This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.

#### Hidden width

|   hidden_size |   count |   mean |      std |    min |
|--------------:|--------:|-------:|---------:|-------:|
|       96.0000 |  3.0000 | 2.0059 |   0.0259 | 1.9825 |
|       64.0000 |  4.0000 | 2.0441 |   0.0551 | 2.0046 |
|       32.0000 |  3.0000 | 2.0442 |   0.0436 | 2.0023 |
|      128.0000 |  1.0000 | 2.0826 | nan      | 2.0826 |
|        8.0000 |  1.0000 | 4.3999 | nan      | 4.3999 |

Within the sampled configurations, **hidden_size=96.0** had the lowest mean validation MAE (2.0059). This is an observational summary of a compact, non-factorial search; interactions among hyperparameters mean it is not a causal one-factor estimate.

### 3.2 Experiment log and failed attempts

**Current experiment status:** 13 ANN experiments recorded; 11 not selected or unsuccessful. The consolidated log also contains classical baseline search rows.

The log records each configuration, seed, best epoch, training time, validation score, failure reason and selection outcome. The diagnostic attempts probe short context, excessive learning rate, under-capacity/over-regularisation, and excessive depth/dropout. Training can complete successfully while the modelling attempt is unsuccessful because it fails to improve validation MAE. The measured outcomes are shown below; do not describe a run as unsuccessful unless the log confirms that it failed or was not selected.

| experiment_id                  |   lookback |   hidden_size |   num_layers |   dropout |   learning_rate |   batch_size | optimizer   | status    |   best_epoch |   val_mae | selection_outcome                  |
|:-------------------------------|-----------:|--------------:|-------------:|----------:|----------------:|-------------:|:------------|:----------|-------------:|----------:|:-----------------------------------|
| ANN-FAIL-01-short-context      |     3.0000 |       32.0000 |       1.0000 |    0.2000 |          0.0010 |     128.0000 | AdamW       | completed |      13.0000 |    2.0894 | not_selected_validation_mae_higher |
| ANN-FAIL-02-high-learning-rate |    24.0000 |       64.0000 |       2.0000 |    0.2000 |          0.0300 |     128.0000 | AdamW       | completed |      13.0000 |    2.1228 | not_selected_validation_mae_higher |
| ANN-FAIL-03-over-regularised   |    24.0000 |        8.0000 |       1.0000 |    0.7000 |          0.0001 |     256.0000 | AdamW       | completed |      28.0000 |    4.3999 | not_selected_validation_mae_higher |
| ANN-FAIL-04-deep-high-dropout  |    24.0000 |      128.0000 |       3.0000 |    0.6000 |          0.0003 |     256.0000 | RMSprop     | completed |      22.0000 |    2.0826 | not_selected_validation_mae_higher |

### 3.3 Final held-out test results

| Model | Test MAE (µg/m³) | Test RMSE (µg/m³) | Test R² | Inference latency (ms/sample) |
|---|---:|---:|---:|---:|
| Training-mean reference | 5.867 | 6.823 | -0.256 | 0.000 |
| Initial LSTM ANN | 1.770 | 2.631 | 0.813 | 0.007 |
| Final LSTM ANN | 1.654 | 2.490 | 0.833 | 0.017 |
| HistGradientBoosting | 1.928 | 2.868 | 0.778 | 0.017 |

The selected ANN fit time was **19.051 seconds**; the classical model fit time after cross-validation was **9.792 seconds**. Full search time is retained in `model_metadata.json`.

The ANN configuration is selected using outer-validation MAE, while the classical baseline hyperparameters are selected using fold-specific training-only cross-validation. Their validation metrics provide the pre-test family comparison. Both frozen models are then evaluated once on the same held-out test period; no model or threshold is changed using those test results. The ANN is deployed in the required application even if the classical model is stronger; in that event, the report must state the result honestly and identify the classical result as a warning against assuming ANN superiority.

Selected baseline parameters:

```json
{
  "min_samples_leaf": 10,
  "max_leaf_nodes": 31,
  "max_iter": 220,
  "learning_rate": 0.03,
  "l2_regularization": 0.1
}
```

### 3.4 Random-seed stability

Three seeds are fixed and reported (42, 7, 2026). The selected ANN configuration is repeated three times. The median validation run is selected to avoid relying on the luckiest seed. Validation mean/spread:

```json
{
  "seeds": [
    2026,
    42,
    7
  ],
  "validation_mae_mean": 1.9924535970750477,
  "validation_mae_std": 0.04659553326050169,
  "validation_mae_min": 1.9516219895389093,
  "validation_mae_max": 2.0432129106819867,
  "selected_seed_median_validation_mae": 1.9825258910042467
}
```

### 3.5 Error analysis

The pipeline produces residual plots, test predictions, error by month, actual-concentration band and input missingness band. `outputs/tables/largest_ann_test_error.json` and `largest_test_error_integrated_gradients.csv` provide a concrete incorrect-prediction case and its model attributions:

The largest absolute ANN error occurred at **2005-02-14 11:00:00**. Actual benzene was **33.900 µg/m³**, the forecast was **8.327 µg/m³**, and absolute error was **25.573 µg/m³**. Top absolute Integrated Gradients features for this case were: PT08.S4(NO2) (|IG|=0.4436), PT08.S2(NMHC) (|IG|=0.3029), AH (|IG|=0.2487), hour_sin (|IG|=0.1635), doy_sin (|IG|=0.1437). These indicate what the model relied on, not the physical cause of the error.

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

|   case | feature_perturbed   |   lag_hours_before_forecast |   prediction_change |   ig_attribution_for_perturbed_coordinate | direction_consistent_with_ig   |
|-------:|:--------------------|----------------------------:|--------------------:|------------------------------------------:|:-------------------------------|
|      1 | PT08.S2(NMHC)       |                           0 |              0.2175 |                                   -0.1931 | False                          |
|      2 | PT08.S2(NMHC)       |                           0 |              0.1769 |                                   -0.6185 | False                          |

A mismatch is reported rather than hidden. These local perturbations are a sanity check, not a causal test. The largest-error test case is also inspected above.

Attributions describe model behaviour relative to a baseline. They do not prove causal reasoning, nor do they guarantee that the attributed variable was the true cause of a forecast.

## 6. Generative AI / foundation model (Question 6)

### 6.1 Adjacent-task application

A generative language model is not used to produce the numeric benzene forecast. The project uses an optional pretrained instruction-following LLM, configured explicitly through `GENAI_MODEL` (set a currently supported identifier for the connected API account), to turn the already-computed ANN forecast and recent sensor context into a short shift-handover note. This adjacent task is more meaningful than asking a general-purpose LLM to regress a small multivariate sensor sequence without a specialised time-series interface. The language model does not change the ANN output.

The adapter is implemented in `src/airquality/genai.py` using the OpenAI Responses API; the fixed prompt is `prompts/shift_handover_v1.txt`. It requires an API key and incurs provider-dependent cost. The exact model identifier, token usage and latency are saved when run. Optional per-million-token input/output price variables calculate an estimated cost; without current rates, the result explicitly reports cost as unavailable. No LLM call is made by default. Obtain lecturer approval for this adjacent task before treating it as the assessed GenAI component.

The optional LLM was not called in the initial build. To evaluate it, obtain lecturer approval, set `OPENAI_API_KEY` and a currently supported `GENAI_MODEL`, then run `python scripts/evaluate_genai.py --n 5`. It will use evenly spaced examples from the same held-out test prediction file, never send the future actual benzene values, record tokens/latency/cost where configured, and create a manual quality-review template.

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
# Windows: .venv\Scripts\activate
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
