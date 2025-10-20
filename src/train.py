import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

PROC = Path("data/processed")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "earthquakes_processed.csv")
meta = json.loads((PROC / "meta.json").read_text())
features = [c for c in meta["features"] if c in df.columns]
target = meta["target"]

# ----- Time-aware split if possible -----
if "Year" in df.columns:
    train_df = df[df["Year"] <= 2018]
    val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2020)]
    test_df  = df[df["Year"] >= 2021]
    if len(train_df)==0 or len(test_df)==0:
        X = df[features]; y = df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                                  stratify=y if y.nunique()==2 else None)
    else:
        X_tr = pd.concat([train_df[features], val_df[features]], ignore_index=True)
        y_tr = pd.concat([train_df[target],  val_df[target]],  ignore_index=True)
        X_te = test_df[features]; y_te = test_df[target]
else:
    X = df[features]; y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y if y.nunique()==2 else None)

# ----- Pipelines -----
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), features)
])

logreg = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))])
logreg.fit(X_tr, y_tr)
pr_log = logreg.predict_proba(X_te)[:, 1]

rf = Pipeline([("pre", ColumnTransformer([("num", SimpleImputer(strategy="median"), features)])),
               ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42))]
             )
rf.fit(X_tr, y_tr)
pr_rf = rf.predict_proba(X_te)[:, 1]

def summarize(name, scores):
    y_prob = scores
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_te, y_prob)),
        "pr_auc": float(average_precision_score(y_te, y_prob)),
        "report": classification_report(y_te, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0,1]).tolist(),
        "n_test": int(len(y_te))
    }
    (MODELS / f"{name}_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[{name}] ROC-AUC={out['roc_auc']:.3f} | PR-AUC={out['pr_auc']:.3f}")
    return y_prob, out

# Metrics & plots
y_prob_log, m_log = summarize("baseline_logreg", pr_log)
y_prob_rf,  m_rf  = summarize("baseline_randomforest", pr_rf)

# ROC/PR curves
def save_curves(y_true, y_score, label_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {label_prefix}")
    plt.savefig(FIG / f"roc_{label_prefix}.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {label_prefix}")
    plt.savefig(FIG / f"pr_{label_prefix}.png", bbox_inches="tight"); plt.close()

save_curves(y_te, y_prob_log, "logreg")
save_curves(y_te, y_prob_rf, "rf")
