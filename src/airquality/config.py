from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

UCI_DATASET_ID = 360
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_LANDING_PAGE = "https://archive.ics.uci.edu/dataset/360/air+quality"
UCI_CITATION = "Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F."

TARGET = "C6H6(GT)"
TARGET_UNIT = "µg/m³"
BASE_FEATURES = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
# Ground-truth pollutant columns are explicitly prohibited from the model input.
FORBIDDEN_INPUTS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
WINDOW_SIZE = 24
HORIZON_HOURS = 1
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_SEEDS = [42, 7, 2026]
PRIMARY_METRIC = "MAE"

for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, TABLES_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
