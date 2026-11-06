# Air Quality Intelligence – Next-Hour Benzene Forecasting

**Task:** forecast the next hourly benzene concentration (`C6H6(GT)`, µg/m³) from the preceding 24 hours of low-cost sensor and weather observations.  
**ANN:** LSTM-based recurrent ANN.  
**Classical baseline:** tuned `HistGradientBoostingRegressor`.  
**Explainability:** Integrated Gradients (local, ANN-specific) and grouped permutation importance (global, model-agnostic).  
**Web app:** Streamlit.  
**Generative AI:** optional pretrained LLM creates a shift-handover note; it does not change the numeric prediction.

## Important project status and dataset note

The source code and project structure are complete, but this ZIP does **not** contain the official UCI CSV or trained real-data model outputs. The UCI repository metadata was verified; however, the official archive could not be downloaded in the build environment. No model score or experiment outcome has been fabricated. Download the official file and run the pipeline to produce the measured results, trained artifacts, full experiment log, EDA figures and final report.

The dataset is about 767 KB and therefore small enough to include in a normal submission. After downloading it, retain `data/raw/AirQualityUCI.csv` and `data/raw/dataset_manifest.json` in the submission ZIP. The `data/demo/input_template.csv` is an empty inference-input template, **not** the UCI dataset.

UCI source: <https://archive.ics.uci.edu/dataset/360/air+quality>  
Download endpoint: <https://archive.ics.uci.edu/static/public/360/air+quality.zip>  
Citation: Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C59K5F>.

The UCI page says the dataset is for research use and excludes commercial use, while its licence field reports CC BY 4.0. This project treats it as non-commercial research/education data unless UCI clarifies the apparent conflict.

## 1. Real-world problem definition

**Intended user:** an environmental monitoring station operator responsible for reviewing hourly air-quality observations.

**Decision supported:** whether the next hour should be prioritised for an additional site check or internal air-quality review, based on the forecast and sensor data-quality context.

**Prediction:** one-hour-ahead benzene concentration. It is an estimate, not a regulatory AQI, legal compliance result or health-risk diagnosis.

## 2. Why this design avoids leakage

- Target: reference-analyser `C6H6(GT)` at time `t+1`.
- Input: sensor channels (`PT08.S1` through `PT08.S5`) plus `T`, `RH`, `AH` over hours `t-23` through `t`.
- The five reference-analyser pollutant fields (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) are excluded from all input features.
- `-200` is converted to missing. Inputs are imputed using training-period medians only and missingness indicators are included. Missing targets are not imputed.
- Chronological split: 70% train / 15% validation / 15% test. The validation split controls selection; the final test split is isolated until the final evaluation.
- Input windows contain only observations before the target timestamp. Missing clock hours are inserted so a one-step forecast means exactly one hour.

## 3. Project structure

```text
air-quality-intelligence/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                    # Downloaded UCI CSV + SHA-256 manifest (not in this initial ZIP)
│   ├── processed/              # Reproducible processed files if created
│   └── demo/input_template.csv # 24-hour app sample with optional actuals for charting
├── docs/
│   ├── architecture.md
│   ├── assessment_mapping.md
│   ├── defence_guide.md
│   ├── demo_script_8min.md
│   ├── experiment_protocol.md
│   ├── genai_human_review.md
│   └── lecturer_approval_request.md
├── experiments/                # Full experiment logs generated after running
├── models/                     # Trained ANN, baseline, preprocessors and model metadata
├── outputs/
│   ├── figures/                 # EDA, learning, residual and prediction plots
│   └── tables/                  # EDA, leakage, test predictions, XAI and metrics
├── prompts/shift_handover_v1.txt
├── reports/FINAL_REPORT.md     # Report scaffold until the real-data pipeline is run
├── scripts/
│   ├── download_data.py
│   ├── run_pipeline.py
│   ├── generate_report.py
│   ├── evaluate_genai.py
│   └── make_template.py
├── src/airquality/              # Data, model, metrics, XAI, inference and GenAI modules
├── tests/test_core.py
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## 4. Setup and full execution

Python 3.10–3.12 is recommended (the Conda environment uses Python 3.11). Create it with `conda env create -f environment.yml` and activate it with `conda activate air-quality-intelligence`, or use a virtual environment. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_pipeline.py
python scripts/generate_report.py
pytest -q
streamlit run app/streamlit_app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

`run_pipeline.py` writes all measured outputs under `models/`, `experiments/`, `outputs/` and `reports/`. You can adjust only the compute budget without changing the test-set rules:

```bash
python scripts/run_pipeline.py --baseline-configs 10 --ann-epochs 28 --final-epochs 35
```

For a quick code-level test before obtaining the dataset:

```bash
pytest -q
```

The tests generate a **synthetic fixture in a temporary folder** to validate parsing, leakage safeguards, tensor dimensions, ANN training and Integrated Gradients. It is not used as a substitute for the UCI experiment.

## 5. Web app input format

Download `data/demo/input_template.csv`, fill in the 24 most recent consecutive hourly sensor readings and upload it to the app. The required columns are:

- `timestamp` (e.g. `2005-02-01 00:00:00`)
- `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`
- `T`, `RH`, `AH`

Do not include ground-truth pollutant concentration fields. Missing sensor values can be left blank; the application uses training-fitted imputers and flags. Rows must be consecutive hourly observations.

## 6. Generative model option

The LLM is for an adjacent text-generation task: a short shift handover note based on an already computed ANN forecast. It is optional and disabled unless `OPENAI_API_KEY` is set. Copy `.env.example` to `.env` only for local development; the project loads it with `python-dotenv`. Set `GENAI_MODEL` to a currently supported model identifier for the connected API account. The prompt is fixed in `prompts/shift_handover_v1.txt`. The LLM never generates or changes the numerical forecast. Obtain lecturer approval for this adjacent task before submission.

## 7. Main result files after a full run

- `outputs/tables/final_metrics.json`, `final_model_comparison.csv`
- `outputs/tables/test_predictions.csv`, `largest_ann_test_error.json`
- `outputs/tables/data_audit.json`, `leakage_audit.json`
- `outputs/tables/integrated_gradients_example.csv`, `ann_grouped_permutation_importance.csv`, `xai_sanity_checks.csv`
- `experiments/experiment_log.csv`, `ann_experiment_log.csv`, `ann_repeated_seeds.csv`, `baseline_search.csv`
- `models/ann_state_dict.pt`, `baseline.joblib`, `preprocessor.joblib`, `target_scaler.joblib`, `model_metadata.json`
- `reports/FINAL_REPORT.md`
- `docs/genai_human_review.md`

## 8. Limitations

The historical dataset is from one Italian roadside station over one year. Generalisation to other locations, modern sensors or different climates is unproven. Sensor drift and cross-sensitivity are documented dataset concerns. Validate with current local data and a certified reference analyser before any operational deployment.
