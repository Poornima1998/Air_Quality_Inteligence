Trained artifacts are written here only after the real UCI pipeline completes:

- `ann_state_dict.pt` – selected LSTM ANN weights
- `baseline.joblib` – tuned classical baseline
- `preprocessor.joblib` – training-only input imputer/scaler
- `target_scaler.joblib` – inverse-transform for benzene concentration
- `model_metadata.json` – architecture, selected hyperparameters, splits, metrics and seed summary

The initial source archive contains no trained model because the UCI data could not be downloaded during assembly.
