from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airquality.config import BASE_FEATURES, TARGET
from airquality.data import load_uci_csv, build_data_bundle, leakage_audit
from airquality.models import fit_ann, predict_ann
from airquality.xai import integrated_gradients
from airquality.inference import predict_from_sensor_history, check_training_range


def make_synthetic_uci_csv(path: Path, n=420):
    rng = np.random.default_rng(2026)
    timestamps = pd.date_range("2004-03-10", periods=n, freq="h")
    hour = np.arange(n)
    sensors = {c: 500 + 40*np.sin(hour/17) + rng.normal(0, 8, n) for c in BASE_FEATURES[:5]}
    sensors.update({"T": 18 + 5*np.sin(hour/24/20) + rng.normal(0, .5, n),
                    "RH": 60 + 8*np.cos(hour/24/18) + rng.normal(0, 1, n),
                    "AH": 8 + .8*np.sin(hour/24/16) + rng.normal(0, .1, n)})
    target = 4 + 0.018*(sensors["PT08.S2(NMHC)"]-500) + 0.01*(sensors["PT08.S1(CO)"]-500) + rng.normal(0, .35, n)
    frame = pd.DataFrame({"Date": timestamps.strftime("%d/%m/%Y"), "Time": timestamps.strftime("%H.%M.%S")})
    frame["CO(GT)"] = 1.0 + rng.normal(0, .1, n)
    frame["PT08.S1(CO)"] = sensors["PT08.S1(CO)"]
    frame["NMHC(GT)"] = 100 + rng.normal(0, 5, n)
    frame[TARGET] = target
    frame["PT08.S2(NMHC)"] = sensors["PT08.S2(NMHC)"]
    frame["NOx(GT)"] = 50 + rng.normal(0, 4, n)
    frame["PT08.S3(NOx)"] = sensors["PT08.S3(NOx)"]
    frame["NO2(GT)"] = 20 + rng.normal(0, 2, n)
    frame["PT08.S4(NO2)"] = sensors["PT08.S4(NO2)"]
    frame["PT08.S5(O3)"] = sensors["PT08.S5(O3)"]
    frame["T"] = sensors["T"]; frame["RH"] = sensors["RH"]; frame["AH"] = sensors["AH"]
    # Add sentinel values to prove the parser catches the original UCI convention.
    frame.loc[7, "PT08.S1(CO)"] = -200
    frame.loc[31, TARGET] = -200
    frame.to_csv(path, sep=";", decimal=",", index=False)
    return frame


def test_parser_handles_uci_format_and_minus_200(tmp_path):
    path = tmp_path / "synthetic.csv"
    raw = make_synthetic_uci_csv(path)
    with path.open("a", encoding="utf-8") as csv_file:
        csv_file.write(";" * (len(raw.columns) - 1) + "\n" * 3)
    got = load_uci_csv(path)
    assert len(got) == len(raw)
    assert got.index.is_monotonic_increasing
    assert pd.isna(got.iloc[7]["PT08.S1(CO)"])
    assert pd.isna(got.iloc[31][TARGET])


def test_bundle_split_and_leakage(tmp_path):
    path = tmp_path / "synthetic.csv"
    make_synthetic_uci_csv(path)
    bundle = build_data_bundle(path)
    audit = leakage_audit(bundle)
    assert audit["pass"]
    assert audit["reference_analyser_pollutant_inputs"] == []
    assert bundle.x_train.shape[1] == 24
    assert bundle.x_train.shape[2] == 22
    assert len(bundle.x_test) > 0
    assert bundle.train_target_index.max() < bundle.split_indices["train_end"]
    assert bundle.split_indices["preprocessor_fit_rows"] - 1 == bundle.train_target_index.max() - 1
    assert audit["preprocessor_uses_only_training_input_rows"] is True
    assert bundle.val_target_index.min() >= bundle.split_indices["train_end"]
    assert bundle.test_target_index.min() >= bundle.split_indices["validation_end"]


def test_ann_training_and_integrated_gradients(tmp_path):
    path = tmp_path / "synthetic.csv"
    make_synthetic_uci_csv(path)
    bundle = build_data_bundle(path)
    config = {"lookback": 24, "hidden_size": 16, "num_layers": 1, "dropout": .1,
              "learning_rate": .001, "batch_size": 64, "optimizer": "AdamW"}
    result = fit_ann(bundle.x_train, bundle.y_train_scaled, bundle.x_val, bundle.y_val_scaled,
                     config, seed=42, max_epochs=2, patience=2, device="cpu")
    pred = predict_ann(result["model"], bundle.x_val[:3], device="cpu")
    assert pred.shape == (3,)
    attr = integrated_gradients(result["model"], bundle.x_val[0], bundle.feature_names, steps=4)
    assert len(attr) == len(bundle.feature_names)
    assert np.isfinite(attr["absolute_attribution"]).all()


def test_request_time_inference_and_distribution_checks(tmp_path):
    path = tmp_path / "synthetic.csv"
    make_synthetic_uci_csv(path)
    bundle = build_data_bundle(path)
    from airquality.models import TemporalANN
    cfg = {"hidden_size": 16, "num_layers": 1, "dropout": .1}
    model = TemporalANN(len(bundle.feature_names), **cfg)
    history = bundle.raw_frame[BASE_FEATURES].iloc[-24:].copy()
    pred, tensor = predict_from_sensor_history(history, model, bundle.preprocessor, bundle.target_scaler,
        {"window_size": 24})
    assert np.isfinite(pred) and tensor.shape == (1, 24, len(bundle.feature_names))
    range_info = check_training_range(history, bundle.preprocessor)
    assert "outside_training_range_count" in range_info
    extreme = history.copy()
    extreme.iloc[0, 0] = 1_000_000
    assert check_training_range(extreme, bundle.preprocessor)["outside_training_range_count"] > 0
    # Missing the required sensor column must fail visibly.
    with np.testing.assert_raises(ValueError):
        predict_from_sensor_history(history.drop(columns=[BASE_FEATURES[0]]), model, bundle.preprocessor, bundle.target_scaler, {"window_size": 24})
    # An irregular hourly index must be rejected.
    irregular = history.copy(); irregular.index = irregular.index.to_series().reset_index(drop=True).apply(lambda x: x)
    irregular.index = pd.DatetimeIndex([history.index[0] + pd.Timedelta(hours=i + (1 if i == 6 else 0)) for i in range(24)])
    with np.testing.assert_raises(ValueError):
        predict_from_sensor_history(irregular, model, bundle.preprocessor, bundle.target_scaler, {"window_size": 24})
