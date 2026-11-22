"""Static figures (matplotlib). Colours: fixed categorical order, recessive axes."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9b9a95"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


SERIES = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4"]  # categorical slots 1,3,7,5 (orange kept for baseline)


def roc_plot(y, p_sig, auc_sig, path, p_base=None, auc_base=None, datasets=None,
             title="Out-of-fold ROC"):
    """K-fold: one signature curve (+ covariate baseline). LODO: one curve per held-out dataset."""
    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="Chance")
    y, p_sig = np.asarray(y), np.asarray(p_sig)
    if datasets is None:
        fpr, tpr, _ = roc_curve(y, p_sig)
        ax.plot(fpr, tpr, lw=2, color=BLUE, label=f"Signature (AUROC {auc_sig:.2f})")
        if p_base is not None:
            fpr, tpr, _ = roc_curve(y, p_base)
            ax.plot(fpr, tpr, lw=2, color=ORANGE, label=f"Covariates only (AUROC {auc_base:.2f})")
    else:
        from sklearn.metrics import roc_auc_score
        for i, d in enumerate(np.unique(datasets)):
            m = np.asarray(datasets) == d
            if len(np.unique(y[m])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y[m], p_sig[m])
            ax.plot(fpr, tpr, lw=2, color=SERIES[i % len(SERIES)],
                    label=f"Held-out {d} (AUROC {roc_auc_score(y[m], p_sig[m]):.2f})")
            if p_base is not None:
                fpr, tpr, _ = roc_curve(y[m], np.asarray(p_base)[m])
                ax.plot(fpr, tpr, lw=1.2, ls=":", color=SERIES[i % len(SERIES)])
        if p_base is not None:
            ax.plot([], [], lw=1.2, ls=":", color=INK2, label=f"Covariates only (mean {auc_base:.2f})")
        title = "Leave-one-dataset-out ROC"
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def permutation_plot(null, observed, pval, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    ax.hist(null, bins=20, color=GREY, edgecolor="white", linewidth=1)
    ax.axvline(observed, color=BLUE, lw=2)
    ax.annotate(f"observed {observed:.2f}\np = {pval:.3f}", xy=(observed, ax.get_ylim()[1] * 0.9),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set(xlabel="AUROC under donor-level label permutation", ylabel="Permutations",
           title="Permutation null (full nested CV each time)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gene_frequency_plot(freq: pd.Series, path, top=20):
    f = freq.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 0.22 * len(f) + 1.0), dpi=150)
    ax.barh(f.index, f.values, color=BLUE, height=0.7)
    ax.set(xlabel="Fraction of outer folds selecting gene", xlim=(0, 1),
           title=f"Signature stability (top {len(f)})")
    _style(ax)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def score_by_group_plot(df, group_col, case, control, path, celltype_col="cell_type", seed=0):
    rng = np.random.default_rng(seed)
    cts = sorted(df[celltype_col].unique(), key=lambda c: (c != "ALL_PBMC", c))
    fig, ax = plt.subplots(figsize=(1.3 * len(cts) + 1.5, 3.4), dpi=150)
    for i, ct in enumerate(cts):
        sub = df[df[celltype_col] == ct]
        z = (sub["ir_score"] - sub["ir_score"].mean()) / sub["ir_score"].std(ddof=1)
        for j, (grp, col) in enumerate([(control, GREY), (case, ORANGE)]):
            v = z[sub[group_col].astype(str) == grp]
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(v)), v, s=14, color=col, alpha=0.85,
                       edgecolor="white", linewidth=0.5, label=grp if i == 0 else None, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.13, x + 0.13, color=INK, lw=1.5, zorder=4)
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.set_xticks(range(len(cts)), [c.replace("_", " ") for c in cts], rotation=20, ha="right")
    ax.set(ylabel="IR score (z within cell type)", title="Signature applied to PBMC pseudobulk")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", bbox_to_anchor=(1, 1))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
