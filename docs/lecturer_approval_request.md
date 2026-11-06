# Lecturer approval request – proposed problem and GenAI adjacent task

**Proposed project:** Air Quality Intelligence – Next-Hour Benzene Concentration Forecasting Using an LSTM ANN.

**Intended user:** An environmental monitoring station operator responsible for reviewing hourly air-quality observations.

**Decision supported:** Whether to prioritise an internal air-quality review or a monitoring-station inspection based on a one-hour-ahead benzene concentration estimate and recent sensor-data quality.

**Dataset:** UCI Machine Learning Repository Air Quality, dataset ID 360, DOI 10.24432/C59K5F. It contains hourly sensor responses, meteorological variables and reference-analyser pollutant concentrations. `-200` is the missing-value marker. The modelling input excludes all reference-analyser pollutant columns; the one-hour-ahead reference benzene value is used as the supervised target.

**Target / task:** Regression forecast of `C6H6(GT)` (µg/m³) one hour ahead using a preceding 24-hour window of five low-cost gas sensor channels and three weather variables.

**ANN:** LSTM-based recurrent ANN. **Classical baseline:** tuned Histogram-Based Gradient Boosting Regressor. **Evaluation:** chronological 70/15/15 split, MAE primary, RMSE/R² secondary, final test set isolated until model selection is complete.

**Proposed GenAI adjacent task (please approve):** Use a pretrained instruction-following language model to generate a short shift-handover note from an already computed ANN forecast and bounded recent sensor context. The LLM will not predict or modify the concentration. The prompt is fixed, the actual future test target is never sent to the LLM, and the exact model/version, latency and token usage will be recorded. A human reviewer will assess factuality, completeness, clarity and unsupported claims. Direct numeric predictive comparison is not applicable because this is an adjacent text-generation task; the report will compare role, data requirements, latency, cost, privacy and limitations.

**Dataset usage note:** The UCI page reports CC BY 4.0 but also states that commercial use is excluded and that the dataset is for research. This project is for non-commercial academic research/education; any ambiguity will be clarified with UCI if needed.

Please confirm whether the proposed problem, dataset and GenAI adjacent task are acceptable before substantial implementation and final submission.
