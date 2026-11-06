#!/usr/bin/env python3
"""Regenerate a blank 24-hour inference template (not a dataset sample)."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airquality.config import BASE_FEATURES
idx = pd.date_range("2005-02-01 00:00:00", periods=24, freq="h")
df = pd.DataFrame({"timestamp": idx.astype(str)})
for col in BASE_FEATURES: df[col] = ""
out = ROOT / "data" / "demo" / "input_template.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote blank template: {out}")
