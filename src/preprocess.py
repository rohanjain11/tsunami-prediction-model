import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# Load first CSV
csvs = sorted(RAW.glob("*.csv"))
if not csvs:
    raise SystemExit("No CSVs in data/raw. Run the download step first.")
df = pd.read_csv(csvs[0])

TARGET = "tsunami"
if TARGET not in df.columns:
    raise SystemExit("Expected a 'tsunami' column as target.")
df[TARGET] = df[TARGET].astype(int)

# --- Feature engineering ---
# Log1p for skewed variables (clip negatives to 0 first)
for c in ["sig", "nst", "dmin", "gap", "depth"]:
    if c in df.columns:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))

# Month cyclic encoding
if "Month" in df.columns:
    df["month_sin"] = np.sin(2*np.pi*(df["Month"]-1)/12.0)
    df["month_cos"] = np.cos(2*np.pi*(df["Month"]-1)/12.0)

# Simple interaction
if {"magnitude","depth"}.issubset(df.columns):
    df["mag_depth"] = df["magnitude"] * np.log1p(df["depth"].clip(lower=0))

# Base numeric features present in your file
base = [c for c in ["magnitude","cdi","mmi","sig","nst","dmin","gap","depth","latitude","longitude"] if c in df.columns]
extra = [c for c in ["log1p_sig","log1p_nst","log1p_dmin","log1p_gap","log1p_depth","month_sin","month_cos","mag_depth"] if c in df.columns]
reserve = [c for c in ["Year","Month"] if c in df.columns]

features = base + extra
out = df[features + reserve + [TARGET]].copy()

out_path = PROC / "earthquakes_processed.csv"
meta_path = PROC / "meta.json"
out.to_csv(out_path, index=False)
meta_path.write_text(json.dumps({"features": features, "reserve_columns": reserve, "target": TARGET, "source": csvs[0].name}, indent=2))
print(f"Wrote {out_path} with {len(features)} features.")
