# Experiment protocol and research integrity

## Locked task
Forecast target `C6H6(GT)` one hour ahead from 24 previous hourly rows of five low-cost sensor responses and three weather variables. Never place any reference-analyser pollutant values in the input. The model predicts a continuous concentration in µg/m³.

## Split
One chronological 70% / 15% / 15% split of the hourly time index. Training windows use targets before the training boundary; validation targets lie in the next period; test targets lie in the final period. Windows may use earlier observations from the preceding period as past context. This is operationally legitimate; no future row is used. All preprocessing parameters are fitted on training rows only.

## Metrics
Primary: MAE in µg/m³. Secondary: RMSE and R². MAPE is omitted because a near-zero true concentration makes it unstable. The primary metric reflects absolute concentration error, which is interpretable to the monitoring operator.

## Classical search
HistGradientBoostingRegressor, random search over learning rate, max iterations, max leaf nodes, minimum samples per leaf and L2 regularisation. Expanding TimeSeriesSplit folds are used inside training data only. The imputer and scaler are refitted separately in each fold using only historical feature rows available before that fold's training cutoff; no preprocessing statistics are carried across from the outer validation/test periods. Ten configurations by default, each evaluated on three folds. The selected baseline is then fitted on all outer training windows and assessed on the untouched outer validation period.

## ANN attempts and tuning
Four initial diagnostic attempts are deliberately chosen to probe failure modes:

1. Short lookback (3 hours) – tests whether a short context discards relevant daily structure.
2. Excessively high learning rate (0.03) – tests instability/divergence.
3. Tiny network (8 hidden units) with 0.70 dropout – tests under-capacity/over-regularisation.
4. Three-layer, 128-unit LSTM with 0.60 dropout and a large batch – tests excessive depth/regularisation for a small dataset.

Eight structured configurations then investigate hidden width, depth, dropout, learning rate, batch size and AdamW/RMSprop. Early stopping is based on validation only. Every completed or failed attempt is logged. An experiment is called unsuccessful only when its real run is worse than the selected validation score or fails; do not relabel an unrun attempt as failed.

The chosen configuration is repeated with seeds 42, 7 and 2026. The median validation run is selected. The final test set is evaluated once after selection and is never used for model tuning, threshold selection, feature selection or prompt tuning.

## Reproducibility
The log contains experiment ID, parameters, random seed, best epoch, training time, train loss, validation MAE/RMSE/R², failure reason and final selection outcome. `requirements.txt` records the software environment. `dataset_manifest.json` records the source, file size and SHA-256 hash.

## XAI checks
Integrated Gradients uses a zero-standardised baseline and reports lag-aggregated sensor feature attribution. Grouped permutation importance permutes an entire input channel across validation cases. Two small positive perturbations test local sensitivity. These checks evaluate consistency, not causality.

## GenAI protocol
The optional LLM receives no actual future test target. Its frozen prompt only transforms ANN output and bounded sensor context into a handover note. Record exact model identifier, latency and token usage. Have a human reviewer score factuality, completeness, clarity and unsupported assertions. Do not compare LLM prose to regression MAE. Obtain lecturer approval before assessment submission.
