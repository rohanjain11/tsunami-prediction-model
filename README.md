# Tsunami Prediction from Earthquake Attributes
**Single-notebook pipeline: download → EDA → feature engineering → time-aware split → baselines (LogReg/RF) → Histogram Gradient Boosting (+ calibration) → explainability & error analysis**

---

## Dataset
- **Global Earthquake–Tsunami Risk Assessment Dataset** (Kaggle)  
  Slug: `ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset`
- Data is fetched on demand with **kagglehub** into `data/raw/` and is **not** committed to the repo.

---

## Environment
Minimal requirements:
```
pandas>=2.0
numpy>=1.25
scikit-learn>=1.3
matplotlib>=3.7
kagglehub>=0.2.5
jupyter
```

---

## What the Notebook Does
1. **Create project folders** (`data/raw`, `data/processed`, etc.).
2. **Download dataset** with `kagglehub` and copy CSVs to `data/raw/`.
3. **Load & sanity checks**  
   - Shape: **(782, 13)**  
   - Columns: `magnitude, cdi, mmi, sig, nst, dmin, gap, depth, latitude, longitude, Year, Month, tsunami`  
   - **No nulls** and **no duplicates** detected.  
   - Target `tsunami`: **0 → 478 (61.13%)**, **1 → 304 (38.87%)**.
   - Plausibility checks passed: lat/long in valid ranges, magnitude in [0,10], non-negative depth.
4. **EDA**
   - Class balance bar plot.
   - Distributions for `magnitude`, `depth`, `mmi`, `cdi`, `sig`, `nst`, `dmin`, `gap`.
   - Numeric correlation heatmap.
5. **Feature engineering**
   - `log1p` for skewed: **`sig, nst, dmin, gap, depth`**.
   - Cyclic month: **`month_sin, month_cos`**.
   - Interactions & nonlinearity:  
     **`magnitude_sq`**, **`mag_logdepth`** (= magnitude × log1p(depth)), **`mag_x_mmi`**, **`mag_x_cdi`**.
   - Geo simples: **`abs_lat`, `abs_lon`**.
   - Depth bins: **shallow (≤70 km), intermediate (70–300 km), deep (>300 km)**.
6. **Split strategy**
   - **Time-aware** when `Year` exists: **train ≤ 2018**, **val 2019–2020**, **test ≥ 2021**; falls back to stratified 80/20 if not feasible.
7. **Models**
   - **Logistic Regression** (scaled; `class_weight="balanced"`).
   - **Random Forest** (imputation only).
   - **Histogram Gradient Boosting (HGB)** with randomized hyperparameter search over:
     - `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_bins`.
   - Threshold tuning by **best F1**.
   - **Isotonic calibration** and reliability curve.
8. **Explainability & errors**
   - **Permutation importance** (PR-AUC drop).
   - Misclassification analysis by **magnitude bins** and **depth categories**.
   - FP/FN tables for the test split.

---

## Results (Held-out Test)

### Main metrics
> Numbers below are from the shared notebook outputs.

| Model | Threshold | ROC-AUC | PR-AUC | Accuracy | Balanced Acc. | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HGB** | 0.50 | **0.769** | **0.902** | **0.732** | **0.700** | 0.891 | 0.754 | 0.817 |
| **HGB (tuned F1)** | **0.0477** | **0.769** | **0.902** | **0.878** | **0.706** | 0.867 | **1.000** | **0.929** |
| RandomForest | 0.50 | 0.712 | 0.900 | 0.720 | 0.649 | 0.862 | 0.769 | 0.813 |
| RandomForest (tuned F1) | 0.25 | 0.712 | 0.900 | 0.805 | 0.551 | 0.810 | 0.985 | 0.889 |
| Logistic Regression | 0.50 | 0.393 | 0.753 | 0.561 | 0.419 | 0.754 | 0.662 | 0.705 |

### Calibration & reliability
- **Isotonic calibration** improved probability calibration (reliability curve closer to diagonal at mid/high scores).  
- Example summary (calibrated, best-F1 thr ≈ **0.154**):
  - PR-AUC ≈ **0.905**, ROC-AUC ≈ **0.745**  
  - Confusion matrix:  
    ```
    [[TN=8,  FP=9],
     [FN=5,  TP=60]]
    ```
  - Precision ≈ **0.870**, Recall ≈ **0.923**, F1 ≈ **0.896** (values depend on the final tuned threshold; the non-calibrated tuned model above achieved higher F1 with a lower threshold).

> Takeaway: calibration makes scores more trustworthy without materially changing ranking metrics (PR-AUC), but the **optimal decision threshold** may shift.

### Feature importance (Permutation; PR-AUC drop)
Top drivers found:
- **longitude** (~0.072)  
- **abs_lon** (~0.054)  
- **magnitude** (~0.0069)  
- **latitude** (~0.0051)  
- **sig** (~0.0013)  
Other engineered features contributed smaller but non-zero value; many had near-zero impact for this dataset/fit.

### Error analysis highlights
- **False Positives:** 9, **False Negatives:** 5 (for the shown test slice and threshold).  
- **By magnitude (misclassification rate)**: highest for **6.0–6.5** (~0.41), then elevated for **7.5–8.0** (~0.33) and **8.0–8.5** (~0.25); lower around **6.5–7.5**.  
- **By depth category:** intermediate **70–300 km** has the highest misclassification (~0.27); shallow (≤70 km) is lower (~0.15); deep (>300 km) moderate (~0.22).

---

## What We Learned
- The dataset is **clean** (no missing/duplicates) and class skew is moderate (~61/39).  
- **Location (lon/lat) and magnitude** are dominant predictors of tsunami occurrence; some engineered features help, but effects are modest beyond the top few.  
- **Histogram Gradient Boosting** clearly outperforms Linear/Logistic and is on par or better than RandomForest on PR-AUC, especially after **threshold tuning**.  
- **Calibration** makes predicted probabilities more reliable; threshold selection should be revisited after calibration to match the precision/recall target.

---

## How to Run (Notebook-first)
1. Install the requirements.
2. Open the notebook in VS Code/Jupyter.
3. Run all cells top-to-bottom:
   - Creates folders → downloads data with `kagglehub` → EDA → feature engineering → time-aware split → train LogReg/RF/HGB → tune threshold → (optionally) calibrate → importance & error analysis.
4. All plots and tables render **inside the notebook**.

---

## Possible Next Steps
- Increase HGB random search budget; focus on `learning_rate ∈ [0.03, 0.07]`, `max_leaf_nodes ∈ [31, 63]`, `min_samples_leaf ∈ [20, 80]`.
- Try **cost-sensitive thresholds** (e.g., enforce Precision ≥ 0.90 and maximize Recall).
- Consider **LightGBM/XGBoost** for comparison if external libraries are allowed.
- Add geophysical context features if available (e.g., distance to trench/coastline, focal mechanism proxies, regional bins).

---

## License
- **Code**: MIT  
- **Data**: Subject to the original Kaggle dataset license/terms; do **not** re-host raw data in this repository.
