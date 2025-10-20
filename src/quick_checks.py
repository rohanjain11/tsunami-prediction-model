from pathlib import Path
import pandas as pd, numpy as np

RAW = Path("data/raw")
csvs = sorted(RAW.glob("*.csv"))
if not csvs:
    raise SystemExit("No CSVs in data/raw. Run the download step first.")
csv_path = csvs[0]
df = pd.read_csv(csv_path)

report = []
report.append(f"FILE: {csv_path.name}")
report.append(f"SHAPE: {df.shape}")
report.append("COLUMNS:\n" + ", ".join(map(str, df.columns.tolist())))

# Missing, duplicates
nulls = df.isna().sum().sort_values(ascending=False)
report.append("TOP NULLS:\n" + nulls.head(20).to_string())
report.append(f"DUPLICATES: {int(df.duplicated().sum())}")

# Target guess
target = next((c for c in df.columns if "tsunami" in c.lower()), None)
if target:
    vc = df[target].value_counts(dropna=False)
    vcp = df[target].value_counts(normalize=True, dropna=False).mul(100).round(2)
    report.append(f"TARGET: {target}\nCounts:\n{vc.to_string()}\nPercent:\n{vcp.to_string()}%")
else:
    report.append("TARGET: <not found – confirm column name>")

# Numeric summary
num = df.select_dtypes(include=np.number)
if not num.empty:
    report.append("NUMERIC DESCRIBE (first 20 rows):\n" + num.describe().T.head(20).to_string())

# Plausibility checks
checks = []
if "latitude" in df:
    checks.append(("latitude outside [-90,90]", int(((df["latitude"]<-90)|(df["latitude"]>90)).sum())))
if "longitude" in df:
    checks.append(("longitude outside [-180,180]", int(((df["longitude"]<-180)|(df["longitude"]>180)).sum())))
for mag_col in ["mag", "magnitude", "mag_value"]:
    if mag_col in df:
        checks.append((f"{mag_col} outside [0,10]", int(((df[mag_col]<0)|(df[mag_col]>10)).sum())))
        break
for dep in ["depth", "depth_km"]:
    if dep in df:
        checks.append((f"{dep} < 0", int((df[dep]<0).sum())))
        break
if checks:
    report.append("VALIDITY CHECKS:\n" + "\n".join(f"- {k}: {v}" for k,v in checks))

# Time coverage (if any)
for dtcol in ["time", "date", "event_time", "origin_time"]:
    if dtcol in df:
        s = pd.to_datetime(df[dtcol], errors="coerce")
        report.append(f"TIME RANGE ({dtcol}): {s.min()} → {s.max()}")
        break

out = Path("reports") / "eda_summary.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n\n".join(report))
print(f"Wrote {out}")
