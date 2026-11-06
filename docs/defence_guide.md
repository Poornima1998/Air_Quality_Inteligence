# Live defence preparation: likely questions and defensible answers

Use the actual values from `outputs/tables/final_metrics.json` and `experiments/experiment_log.csv` when answering. Never memorise placeholder scores.

## 1. Why did you select this problem?

The dataset is a multivariate hourly time series from an outdoor roadside air-quality station. A next-hour benzene forecast has a defined user, target, input availability and operational decision. It allows a meaningful sequence model and a controlled comparison against a tuned classical baseline. The intended user is a station operator who can use the estimate to prioritise an internal review or inspection.

## 2. Why benzene, not CO or NO2?

Benzene has a direct reference-analyser target, `C6H6(GT)`, and the task can be framed as forecasting the next hour using the low-cost sensor array and weather. The target is a continuous concentration in µg/m³, so this is a regression task. The choice is fixed before modelling rather than chosen after looking at test scores.

## 3. What does `-200` mean?

It is UCI's missing-value sentinel, not a real negative concentration or sensor reading. It is converted to NaN before EDA and modelling. For input features, missing values are imputed using medians fitted on the training period and corresponding missingness indicators are retained. Missing targets are excluded rather than imputed.

## 4. How do you prevent target leakage?

The input window contains rows `t-23` through `t`, and the target is `C6H6(GT)` at `t+1`. All five reference-analyser pollutant columns are excluded from the feature list. The feature preprocessor is fitted only on the training period. A chronological split is used and the test set is not touched during tuning. `outputs/tables/leakage_audit.json` records these checks.

## 5. Why use a 24-hour lookback?

It provides a full daily cycle of hourly observations. It is an explicit modelling hypothesis, not a fact guaranteed by the data. The project also tests a shorter lookback as an ablation and inspects validation performance. The window is fixed before final test evaluation.

## 6. Why an LSTM?

The input is a multivariate sequence. LSTM gates can preserve or forget information across a sequence, which is suitable for learning temporal dependencies. It consumes the `(batch, 24, 22)` tensor and returns a continuous regression output. The architecture is not assumed to be automatically superior to a tree model; the validation and held-out comparisons decide.

## 7. Why not a simple ANN/MLP?

A flattened MLP can learn nonlinear interactions but does not explicitly model sequence order. An LSTM is a more direct sequence architecture for a time-series task. An MLP remains a reasonable alternative and could be added if compute and time permit.

## 8. Why this classical baseline?

HistGradientBoosting handles nonlinear feature interactions and is effective on small to medium tabular datasets. It uses the same lookback-derived inputs and target examples, so the comparison is aligned. Its hyperparameters are tuned using expanding-window cross-validation on the training period.

## 9. Why MAE?

MAE is in the target unit, µg/m³, and expresses the typical absolute concentration error. RMSE highlights larger misses; R² is a supplementary fit statistic. MAPE is not used because benzene values may be close to zero and make percentage errors unstable.

## 10. Why not randomly split the dataset?

Random splitting would mix earlier and later timestamps and could give an optimistic estimate when temporal dependence and sensor drift exist. A chronological split better approximates future prediction using past observations.

## 11. Why can validation windows use history from the training period?

At deployment, the model is allowed to use the most recent past observations available at prediction time. The validation target itself and its future inputs are not used to fit the preprocessor or model. The target row is always after the input window. This mirrors the operational forecasting setup.

## 12. What if the classical model beats the ANN?

Report it honestly. The assignment requires an ANN to be developed and deployed, but does not justify hiding a stronger baseline. We can state that the ANN architecture was less suitable for this dataset or available sample size, while the classical baseline produced better predictive accuracy. The ANN still satisfies the model-development and application requirement, and the comparison is an important result.

## 13. How did you tune the ANN?

I compared hidden width, depth, dropout, learning rate, batch size and two optimisers, AdamW and RMSprop. I first ran four diagnostic attempts to probe short context, high learning rate, excessive dropout/under-capacity and an over-complex network. Every run is recorded with a seed, validation metrics, best epoch and outcome. The final selected configuration is repeated with three fixed seeds.

## 14. How did you tune the classical baseline fairly?

I used a comparable order of search effort, a set of about ten parameter configurations and three expanding-window time-series folds. It used the same training examples and 24-hour input history. The test set was not used for search.

## 15. How does the web app work?

A user uploads a CSV containing at least 24 consecutive hourly records with five gas-sensor values and three weather measurements. The app validates the timestamps and columns, applies the saved training-fitted preprocessing, passes the tensor into the saved LSTM weights and returns a next-hour concentration. It also displays validation MAE and a local Integrated Gradients explanation.

## 16. What happens with missing input values?

They are recognised as missing, flagged with missingness indicators and median-imputed using medians learned from the training period. The application tells the user when imputation was required. If required sensor columns are absent or the timestamps are irregular, the app rejects the upload.

## 17. What does Integrated Gradients explain?

It estimates how input values contribute to the model output relative to a baseline input. The contributions are aggregated across lags to show the most influential channels for a forecast. It is a model-specific local attribution method, not a causal explanation.

## 18. Why also use permutation importance?

It is model-agnostic and provides a global measure: how much validation MAE increases when a feature channel's entire lag profile is permuted across cases. It provides an independent check on feature reliance, although correlated inputs can share or mask importance.

## 19. Why discuss SHAP but not make it the app default?

SHAP is a valuable additive attribution framework, but a 24 × 22 sequence produces hundreds of lagged input dimensions. Kernel SHAP can be computationally expensive and sensitive to background choices and feature dependence. Integrated Gradients is more practical for a local LSTM explanation, supported by grouped permutation importance for global interpretation.

## 20. How did you validate the explanations?

I perturb a highly attributed input slightly in two separate validation examples and observe the direction and size of the forecast change. I also inspect the largest held-out ANN error and check whether the highlighted features make sense. A disagreement is reported as a limitation rather than hidden. Attribution is not evidence of causality.

## 21. What is the role of GenAI?

The numerical forecast is produced by the trained LSTM. The optional generative model performs an adjacent language task: it writes a shift-handover note from the ANN forecast and bounded recent context. It cannot change the numeric prediction. This is more appropriate than using a generic LLM for numeric regression without a suitable time-series interface. Lecturer approval is required for this adjacent-task interpretation.

## 22. Why not compare the LLM with ANN using MAE?

The LLM is not asked to forecast benzene. It creates text. Comparing its prose with concentration MAE would be invalid. We compare complementary use, data requirements, latency, provider cost, privacy and limitations. A fair predictive comparison would require a time-series foundation model with the same permitted sensor inputs and same held-out test examples and MAE metric.

## 23. What are the biggest limitations?

The dataset is from one Italian roadside station in 2004–2005; the model may not generalise to Sri Lanka or another sensor. Sensor drift, cross-sensitivity and target missingness limit the data. It is a prototype, not an approved environmental monitoring system. A current local dataset and certified reference measurements are required before operational use.

## 24. What would you do next?

Obtain current local sensor and reference-analyser data, validate sensor calibration, use rolling-origin backtesting, monitor drift, recalibrate periodically, test generalisation across stations, calibrate prediction intervals and conduct an operational pilot with human review. An additional tree/MLP/sequence-foundation baseline could then be added.

## Five statements to avoid

- Do not say the model proves pollution is safe or unsafe.
- Do not say the explanations identify causes.
- Do not say the model generalises to Sri Lanka; the training data are from an Italian site.
- Do not say an ANN is best unless the test metrics actually support it.
- Do not present synthetic smoke-test scores as UCI results or claim an experiment ran before it appears in the log.
