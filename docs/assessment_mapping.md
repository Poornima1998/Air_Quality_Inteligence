# Assignment requirements and evidence map

| Requirement / assessment area | Implementation | Evidence produced |
|---|---|---|
| One specific real-world application, intended user and supported decision | Next-hour benzene forecast for environmental monitoring station operator | README; FINAL_REPORT.md §1 |
| Appropriate dataset and source permission | UCI Air Quality ID 360; official downloader and citation | data/raw/AirQualityUCI.csv; dataset_manifest.json; docs/data note in README |
| Data preparation, EDA and leakage checks | `load_uci_csv`, -200 conversion, hourly continuity reindex, training-only medians/scaler, cyclical time features | `outputs/tables/data_audit.json`, `leakage_audit.json`, missingness/correlation/time plots |
| ANN specifically designed for task | LSTM sequence regressor, Huber loss, AdamW/RMSprop, gradient clipping and early stopping | `src/airquality/models.py`, model metadata, architecture description in report |
| Tuned classical baseline | HistGradientBoosting with time-series cross-validation | baseline_search.csv; final comparison table |
| Four or more unsuccessful attempts and experiment log | Four diagnostic ANN configurations plus structured search; unsuccessful classification based on measured outcome | `experiments/experiment_log.csv`, `ann_experiment_log.csv`, learning curves |
| Primary metric and one held-out test set | MAE primary, RMSE/R² secondary; one chronological test set | `final_metrics.json`, `test_predictions.csv` |
| Functional web application | Streamlit file-upload app with live request-time prediction, input validation and training-range/OOD warnings | `app/streamlit_app.py` |
| Explainable AI | Integrated Gradients in app; grouped permutation global analysis; local sensitivity checks | `integrated_gradients_example.csv`, `ann_grouped_permutation_importance.csv`, `xai_sanity_checks.csv` |
| Generative/foundation model | Optional LLM for adjacent shift-handover-note generation; no numerical forecast changes | `src/airquality/genai.py`, fixed prompt, model identifier, token usage, latency and optional cost estimate in `evaluate_genai.py` output |
| Reproducible code, dataset/download, report, demo, defence | requirements, download manifest, run scripts, report generator, demo/defence docs | whole project tree |

## Assessment sections

- Q1 Dataset and EDA: data audit, missingness, target distribution, correlation, monthly summaries, leakage audit, primary metric rationale.
- Q2a ANN design: LSTM architecture, input tensor, activations, parameter count, loss, regularisation.
- Q2b Alternatives: classical gradient boosting, MLP, ARIMA/exponential smoothing; selection rationale.
- Q3a Model development and comparison: learning curves, baseline search, model metrics, residual/error slices.
- Q3b Optimisation: learning rate, batch size, hidden width, depth, dropout, optimiser; AdamW versus RMSprop; failed/unsuccessful attempts in log.
- Q4 Web application: upload validation, request-time prediction, score explanation, limitations.
- Q5a XAI methodology comparison: Integrated Gradients, permutation importance, SHAP KernelExplainer.
- Q5b XAI implementation/validation: local explanation, two perturbation sanity checks, largest test error case.
- Q6a Generative model: optional pretrained LLM for approved adjacent handover-note task.
- Q6b Comparative analysis: not a fair regression MAE comparison because the LLM performs a different task; compare by use, latency, cost, privacy, limitations and state evidence.

**Submission warning:** The assessment asks for lecturer approval of the unique project problem, dataset and GenAI adjacent task where applicable. Submit `lecturer_approval_request.md` before claiming approval.
