"""Experiment 2: document features + a SIMULATED face-match score.

No real face data is used. The face score is drawn from distributions (see SIM below) so we can ask:
"how much does adding a face signal help the document model, as a function of how many forgeries
involve a swapped photo?"  Replace `simulate_face_score` with real similarity scores
(e.g. from DeepFace / InsightFace on a real dataset) to turn this into a real experiment.

Usage:  python run_experiment2.py        (needs data/features.csv from extract_features.py)
Output: results/experiment2_metrics.csv
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

warnings.filterwarnings("ignore")
df = pd.read_csv("data/features.csv")
meta = {"doc_id", "identity", "label", "kind", "edited_fields", "path", "_ocr_text", "_error"}
DOC = [c for c in df.columns if c not in meta]
if len(DOC) < 20:
    raise SystemExit("data/features.csv has too few feature columns; run extract_features.py first.")
y = df["label"].values
groups = df["identity"].values

# SIM: cosine-like similarity between ID-photo and selfie.
# same person: mean 0.75, sd 0.10 ; swapped photo (different person): mean 0.35, sd 0.12
SAME, SWAP = (0.75, 0.10), (0.35, 0.12)


def simulate_face_score(p_swap, seed):
    rng = np.random.RandomState(seed)
    swapped = (y == 1) & (rng.rand(len(y)) < p_swap)
    mu = np.where(swapped, SWAP[0], SAME[0])
    sd = np.where(swapped, SWAP[1], SAME[1])
    return np.clip(rng.normal(mu, sd), 0, 1)


MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=2),
    "XGBoost": lambda: XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.9,
                                     eval_metric="logloss", random_state=0, n_jobs=2),
}
REPEATS = 5


def cv_scores(mk, X, seed):
    rng = np.random.RandomState(seed)
    ids = np.unique(groups)
    perm = dict(zip(ids, rng.permutation(len(ids))))
    g = np.array([perm[i] for i in groups])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = mk()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    pr, rc, f1, _ = precision_recall_fscore_support(y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    return pr, rc, f1, roc_auc_score(y, oof), accuracy_score(y, (oof >= 0.5).astype(int))


X_doc = df[DOC].fillna(0).values.astype(float)
rows = []
for p_swap in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for fset in ["face only", "documents only", "documents + face"]:
        for mname, mk in MODELS.items():
            ms = []
            for s in range(REPEATS):
                face = simulate_face_score(p_swap, 100 + s).reshape(-1, 1)
                X = {"face only": face, "documents only": X_doc, "documents + face": np.hstack([X_doc, face])}[fset]
                ms.append(cv_scores(mk, X, s))
            ms = np.array(ms)
            rows.append({"p_swap": p_swap, "feature_set": fset, "model": mname,
                         **{f"{k}_mean": ms[:, i].mean() for i, k in enumerate(["precision", "recall", "f1", "auc", "accuracy"])},
                         **{f"{k}_std": ms[:, i].std() for i, k in enumerate(["precision", "recall", "f1", "auc"])}})
res = pd.DataFrame(rows)
import os
os.makedirs("results", exist_ok=True)
res.to_csv("results/experiment2_metrics.csv", index=False)
pd.set_option("display.width", 200)
print("F1:")
print(res.pivot_table(index=["p_swap", "model"], columns="feature_set", values="f1_mean").round(3).to_string())
print("\nAccuracy:")
print(res.pivot_table(index=["p_swap", "model"], columns="feature_set", values="accuracy_mean").round(3).to_string())