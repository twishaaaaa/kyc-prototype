"""Make report figures + paste-ready tables from the CSVs written by run_experiments.py.

Usage (inside your activated venv, from the kyc_prototype folder):
    pip install matplotlib pandas
    python make_figures.py
Outputs go to results/: fig1_f1_by_feature_set.png, fig2_detection_by_forgery_type.png,
fig3_top_features.png, tables.md
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results"
m = pd.read_csv(f"{R}/experiment1_metrics.csv")
pt = pd.read_csv(f"{R}/experiment1_per_forgery_type.csv")
top = pd.read_csv(f"{R}/rf_top_features.csv", index_col=0)

INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
C = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated categorical slots 1-3)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1
order = ["consistency only", "ocr_quality only", "forensic only", "consistency + ocr_quality", "all document features"]
labels = ["Consistency\nonly", "OCR quality\nonly", "Forensic\nonly", "Consistency +\nOCR quality", "All document\nfeatures"]
models = ["LogReg", "RandomForest", "XGBoost"]
names = {"LogReg": "Logistic Regression", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
fig, ax = plt.subplots(figsize=(9.5, 4.8))
w, x = 0.26, np.arange(len(order))
for i, mod in enumerate(models):
    sub = m[m.model == mod].set_index("feature_set").loc[order]
    xs = x + (i - 1) * (w + 0.02)
    ax.bar(xs, sub.f1_mean, w, yerr=sub.f1_std, color=C[i], label=names[mod], error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for xv, v in zip(xs, sub.f1_mean):
        ax.text(xv, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
rule = m[m.model == "no training"]
if len(rule):
    rv = float(rule.f1_mean.iloc[0])
    ax.axhline(rv, color=GRAY, ls="--", lw=1.2)
    ax.text(1.0, rv + 0.015, f"Rule baseline F1 = {rv:.2f}", ha="center", fontsize=8.5, color=INK2)  # sits above the short OCR-quality bars
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("F1 score (mean ± SD, 5 repeats of 5-fold CV)")
ax.set_title("Experiment 1: document-only detection by feature group", loc="left", fontsize=11.5)
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
style(ax)
fig.tight_layout()
fig.savefig(f"{R}/fig1_f1_by_feature_set.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2
sets = [("consistency only", "Consistency only"), ("forensic only", "Forensic only"), ("all document features", "All document features")]
rf = pt[pt.model.isin(["RandomForest", "no training"])]
kinds = [("forged_A", "A: visible fields\nedited only"), ("forged_B", "B: MRZ edited,\ncheck digits fixed"), ("forged_C", "C: MRZ edited,\ncheck digits not fixed")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 1.35]})
w, x = 0.2, np.arange(len(kinds))
series = [(s, lab, c) for (s, lab), c in zip(sets, C)] + [("rule: any consistency check fails", "Rule baseline", GRAY)]
for i, (s, lab, col) in enumerate(series):
    v = [float(rf[(rf.feature_set == s) & (rf.kind == k)].value.iloc[0]) for k, _ in kinds]
    xs = x + (i - 1.5) * (w + 0.02)
    a1.bar(xs, v, w, color=col, label=lab)
    for xv, vv in zip(xs, v):
        a1.text(xv, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8, color=INK2)
a1.set_xticks(x)
a1.set_xticklabels([l for _, l in kinds])
a1.set_ylim(0, 1.26)
a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a1.set_ylabel("Detection rate")
a1.set_title("Detection rate by forgery type (Random Forest)", loc="left", fontsize=11.5)
a1.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
style(a1)
fpr = [float(rf[(rf.feature_set == s) & (rf.kind == "genuine")].value.iloc[0]) for s, _, _ in series]
cols = [c for _, _, c in series]
a2.bar(np.arange(4), fpr, 0.6, color=cols)
for i, v in enumerate(fpr):
    a2.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
a2.set_xticks(np.arange(4))
a2.set_xticklabels(["Cons.", "Foren.", "All", "Rule"], fontsize=9)
a2.set_ylim(0, max(fpr) * 1.25 + 0.02)
a2.set_ylabel("False positive rate")
a2.set_title("Genuine cards flagged", loc="left", fontsize=11.5)
style(a2)
fig.tight_layout()
fig.savefig(f"{R}/fig2_detection_by_forgery_type.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3
t = top.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(t.index, t.importance, color=C[0], height=0.62)
for yv, v in enumerate(t.importance):
    ax.text(v + 0.002, yv, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_xlabel("Random Forest feature importance")
ax.set_title("Top 10 features (all document features)", loc="left", fontsize=11.5)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{R}/fig3_top_features.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- tables.md
def f(r, k):
    return f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}" if r[k + "_std"] > 0 else f"{r[k + '_mean']:.3f}"

lines = ["## Table 1. Experiment 1 results (mean ± SD over 5 repeats)\n",
         "| Feature set | Model | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|"]
for r in m.itertuples(index=False):
    d = r._asdict()
    auc = "n/a" if pd.isna(d["auc_mean"]) else f(d, "auc")
    lines.append(f"| {d['feature_set']} | {d['model']} | {f(d, 'precision')} | {f(d, 'recall')} | {f(d, 'f1')} | {auc} |")
lines += ["", "## Table 2. Detection rate by forgery type (Random Forest) and false-positive rate on genuine cards\n",
          "| Feature set | A | B | C | Genuine (FPR) |", "|---|---|---|---|---|"]
piv = rf.pivot_table(index="feature_set", columns="kind", values="value")
for s in piv.index:
    lines.append(f"| {s} | {piv.loc[s, 'forged_A']:.2f} | {piv.loc[s, 'forged_B']:.2f} | {piv.loc[s, 'forged_C']:.2f} | {piv.loc[s, 'genuine']:.2f} |")
lines += ["", "## Table 3. Top 10 Random Forest features\n", "| Feature | Importance |", "|---|---|"]
lines += [f"| {n} | {v:.3f} |" for n, v in top.head(10).importance.items()]
open(f"{R}/tables.md", "w").write("\n".join(lines) + "\n")
print("wrote:", ", ".join(sorted(p for p in os.listdir(R) if p.endswith((".png", ".md")))))
