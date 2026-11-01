"""Experiment 1 (document-only) + feature-group ablation on the synthetic cards.

Evaluation: repeated GroupKFold (groups = identity, so a genuine card and its
tampered twin never straddle train/test). Threshold fixed at 0.5.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from extract_features import group_of

warnings.filterwarnings("ignore")
df = pd.read_csv("data/features.csv")
meta = {"doc_id", "identity", "label", "kind", "edited_fields", "path", "_ocr_text", "_error"}
cols = [c for c in df.columns if c not in meta]
if len(cols) < 20:
    raise SystemExit(f"data/features.csv has only {len(cols)} feature columns: feature extraction failed. "
                     "Fix extract_features.py errors (usually Tesseract not installed) and re-run it first.")
G ={g: [c for c in cols if group_of(c) == g] for g in ["consistency", "ocr_quality", "forensic"]}
SETS = {
    "consistency only": G["consistency"],
    "ocr_quality only": G["ocr_quality"],
    "forensic only": G["forensic"],
    "consistency + ocr_quality": G["consistency"] + G["ocr_quality"],
    "all document features": cols,
}
MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=2),
    "XGBoost": lambda: XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.9,
                                     eval_metric="logloss", random_state=0, n_jobs=2),
}
y = df["label"].values
groups = df["identity"].values
X_all = df[cols].fillna(0).values.astype(float)
idx = {c: i for i, c in enumerate(cols)}


def cv_oof(make_model, feat, seed):
    """One repeat of 5-fold group CV; returns out-of-fold probabilities."""
    rng = np.random.RandomState(seed)
    ids = np.unique(groups)
    perm = dict(zip(ids, rng.permutation(len(ids))))
    g = np.array([perm[i] for i in groups])
    X = X_all[:, [idx[c] for c in feat]]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def metrics(y_true, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return pr, rc, f1, roc_auc_score(y_true, p), accuracy_score(y_true, pred)


REPEATS = 5
rows, per_type = [], []
for sname, feat in SETS.items():
    for mname, mk in MODELS.items():
        ms, oofs = [], []
        for s in range(REPEATS):
            oof = cv_oof(mk, feat, s)
            ms.append(metrics(y, oof))
            oofs.append(oof)
        ms = np.array(ms)
        rows.append({"feature_set": sname, "model": mname, "n_features": len(feat),
                     **{f"{k}_mean": ms[:, i].mean() for i, k in enumerate(["precision", "recall", "f1", "auc", "accuracy"])},
                     **{f"{k}_std": ms[:, i].std() for i, k in enumerate(["precision", "recall", "f1", "auc"])}})
        # per-kind detection rate (recall for each forgery type; FPR for genuine)
        mean_pred = np.mean([(o >= 0.5).astype(int) for o in oofs], axis=0)
        for kind in sorted(df["kind"].unique()):
            m = (df["kind"] == kind).values
            val = mean_pred[m].mean()
            per_type.append({"feature_set": sname, "model": mname, "kind": kind,
                             "metric": "false_positive_rate" if kind == "genuine" else "detection_rate",
                             "value": val})

res = pd.DataFrame(rows)
pt = pd.DataFrame(per_type)

# rule baseline (no training): flag if any consistency check fails
def rule(r):
    return int(min(r["name_sim_surname"], r["name_sim_given"], r["doc_sim"], r["dob_sim"], r["exp_sim"]) < 0.99
               or min(r["mrz_chk_doc"], r["mrz_chk_dob"], r["mrz_chk_exp"], r["mrz_chk_all"]) < 1
               or min(r["exp_after_issue"], r["issue_age_ok"], r["validity_span_ok"]) < 1)

rp = df.apply(rule, axis=1).values
pr, rc, f1, _ = precision_recall_fscore_support(y, rp, average="binary", zero_division=0)
acc = accuracy_score(y, rp)
rule_row = {"feature_set": "rule: any consistency check fails", "model": "no training", "n_features": 0,
            "precision_mean": pr, "recall_mean": rc, "f1_mean": f1, "auc_mean": np.nan, "accuracy_mean": acc,
            "precision_std": 0, "recall_std": 0, "f1_std": 0, "auc_std": 0, "accuracy_std": 0}
res = pd.concat([pd.DataFrame([rule_row]), res], ignore_index=True)
for kind in sorted(df["kind"].unique()):
    m = (df["kind"] == kind).values
    pt = pd.concat([pt, pd.DataFrame([{"feature_set": "rule: any consistency check fails", "model": "no training",
                                       "kind": kind, "metric": "false_positive_rate" if kind == "genuine" else "detection_rate",
                                       "value": rp[m].mean()}])], ignore_index=True)

import os
os.makedirs("results", exist_ok=True)
res.to_csv("results/experiment1_metrics.csv", index=False)
pt.to_csv("results/experiment1_per_forgery_type.csv", index=False)

# RF feature importance on all features (fit on everything; descriptive only)
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=2).fit(X_all, y)
imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
imp.head(15).to_csv("results/rf_top_features.csv", header=["importance"])

pd.set_option("display.width", 200, "display.max_columns", 20)
show = res.copy()
for k in ["precision", "recall", "f1", "auc", "accuracy"]:
    show[k] = show.apply(lambda r: f"{r[k+'_mean']:.3f}±{r[k+'_std']:.3f}" if r[k + "_std"] > 0 else f"{r[k+'_mean']:.3f}", axis=1)
print(show[["feature_set", "model", "n_features", "accuracy", "precision", "recall", "f1", "auc"]].to_string(index=False))
print("\nPer forgery type (RandomForest, mean over repeats):")
print(pt[pt.model.isin(["RandomForest", "no training"])].pivot_table(index="feature_set", columns="kind", values="value").round(3).to_string())
print("\nTop RF features:\n", imp.head(10).round(3).to_string())
print("\nclass counts:", df["kind"].value_counts().to_dict())