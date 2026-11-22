"""Nested, group-aware cross-validation for an insulin-resistance classifier.

Leakage rules enforced here:
  * gene ranking, feature selection, scaling and hyper-parameter search are all fitted
    inside the training part of every outer fold (one sklearn Pipeline + GridSearchCV);
  * outer *and* inner splits are grouped by donor, so several samples of one person
    (e.g. pseudobulk cell types, repeated visits) never sit on both sides of a split;
  * with >= 2 datasets the outer loop can be leave-one-dataset-out (LODO);
  * the permutation null shuffles labels between donors and re-runs the full nested CV.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TopKWelch, WithinSampleRank


def _elastic_net(seed):
    kw = dict(solver="saga", l1_ratio=0.5, class_weight="balanced", max_iter=20000, random_state=seed)
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 8):  # older sklearn needs the explicit penalty name
        kw["penalty"] = "elasticnet"
    return LogisticRegression(**kw)


def make_pipeline(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("rank", WithinSampleRank()),
        ("select", TopKWelch(k=50)),
        ("scale", StandardScaler()),
        ("clf", _elastic_net(seed)),
    ])


def _n_inner(y, groups, max_splits=3):
    """Largest usable number of inner folds (each fold needs both classes)."""
    df = pd.DataFrame({"y": y, "g": groups}).drop_duplicates("g")
    return int(max(2, min(max_splits, df["y"].value_counts().min())))


def make_search(y, groups, k_grid, C_grid, seed=0) -> GridSearchCV:
    return GridSearchCV(
        make_pipeline(seed),
        {"select__k": list(k_grid), "clf__C": list(C_grid)},
        scoring="roc_auc",
        cv=StratifiedGroupKFold(n_splits=_n_inner(y, groups), shuffle=True, random_state=seed),
        refit=True, n_jobs=1, error_score=np.nan,
    )


def outer_splits(y, groups, datasets=None, n_folds=5, seed=0):
    if datasets is not None and len(np.unique(datasets)) >= 2:
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, datasets))
    n_groups_min = pd.DataFrame({"y": y, "g": groups}).drop_duplicates("g")["y"].value_counts().min()
    k = int(min(n_folds, n_groups_min))
    if k < 2:
        raise ValueError("Need at least two donors per class for cross-validation.")
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y, groups))


@dataclass
class CVResult:
    oof: np.ndarray                 # out-of-fold P(IR), averaged over repeats
    auc_per_repeat: list[float]
    fold_records: list[dict] = field(default_factory=list)
    gene_frequency: pd.Series | None = None
    lodo_auc: dict | None = None

    @property
    def primary_auc(self) -> float:
        """LODO: mean AUROC within each held-out dataset (pooling across datasets mixes
        calibration shifts into discrimination). K-fold: mean AUROC over repeats."""
        if self.lodo_auc:
            vals = [v for v in self.lodo_auc.values() if v is not None]
            return float(np.mean(vals))
        return float(np.mean(self.auc_per_repeat))


def nested_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, k_grid, C_grid,
              datasets=None, n_folds=5, repeats=5, seed=0) -> CVResult:
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    genes = np.asarray(X.columns)
    Xv = X.to_numpy(dtype=float)
    lodo = datasets is not None and len(np.unique(datasets)) >= 2
    repeats = 1 if lodo else repeats
    preds = np.full((repeats, len(y)), np.nan)
    records, selected = [], []
    for r in range(repeats):
        for f, (tr, te) in enumerate(outer_splits(y, groups, datasets, n_folds, seed + r)):
            search = make_search(y[tr], groups[tr], k_grid, C_grid, seed + r)
            search.fit(Xv[tr], y[tr], groups=groups[tr])
            preds[r, te] = search.predict_proba(Xv[te])[:, 1]
            mask = search.best_estimator_.named_steps["select"].get_support()
            selected.append(genes[mask])
            rec = {"repeat": r, "fold": f, "n_train": len(tr), "n_test": len(te),
                   "inner_auc": float(search.best_score_), **search.best_params_}
            if lodo:
                rec["held_out_dataset"] = str(np.unique(datasets[te])[0])
            records.append(rec)
    aucs = [float(roc_auc_score(y, p)) for p in preds]
    freq = pd.Series(np.concatenate(selected)).value_counts() / len(selected)
    res = CVResult(oof=np.nanmean(preds, axis=0), auc_per_repeat=aucs,
                   fold_records=records, gene_frequency=freq.rename("selection_frequency"))
    if lodo:
        res.lodo_auc = {}
        for d in np.unique(datasets):
            m = datasets == d
            res.lodo_auc[str(d)] = float(roc_auc_score(y[m], res.oof[m])) if len(np.unique(y[m])) == 2 else None
    return res


def bootstrap_auc_ci(y, p, n_boot=2000, seed=0, groups=None, datasets=None):
    """Percentile 95% CI for AUROC, resampling donors (groups) with replacement.
    With datasets, donors are resampled within each dataset and the statistic is the
    mean within-dataset AUROC (matching ``CVResult.primary_auc``)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)).astype(str) if groups is None else np.asarray(groups)
    datasets = np.zeros(len(y), dtype=int) if datasets is None else np.asarray(datasets)
    blocks = {}
    for d in np.unique(datasets):
        gs = np.unique(groups[datasets == d])
        blocks[d] = (gs, {g: np.flatnonzero((groups == g) & (datasets == d)) for g in gs})
    stats = []
    for _ in range(n_boot):
        aucs = []
        for d, (gs, idx_by_g) in blocks.items():
            idx = np.concatenate([idx_by_g[g] for g in rng.choice(gs, len(gs), replace=True)])
            if len(np.unique(y[idx])) == 2:
                aucs.append(roc_auc_score(y[idx], p[idx]))
        if len(aucs) == len(blocks):
            stats.append(np.mean(aucs))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def permute_by_group(y, groups, rng):
    g = pd.Series(y, index=groups)
    donor_labels = g.groupby(level=0).first()
    shuffled = pd.Series(rng.permutation(donor_labels.to_numpy()), index=donor_labels.index)
    return shuffled.loc[groups].to_numpy()


def permutation_test(X, y, groups, k_grid, C_grid, observed_auc, n_perm=100, datasets=None,
                     n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        yp = permute_by_group(np.asarray(y), np.asarray(groups), rng)
        res = nested_cv(X, yp, groups, k_grid, C_grid, datasets, n_folds, repeats=1, seed=seed + 1000 + i)
        null.append(res.primary_auc)
    null = np.asarray(null)
    p = (1 + np.sum(null >= observed_auc)) / (1 + len(null))
    return float(p), null


def covariate_baseline(cov: pd.DataFrame, y, groups, datasets=None, n_folds=5, repeats=5, seed=0):
    """Out-of-fold AUROC of a covariates-only logistic model (same split scheme)."""
    Xc = pd.get_dummies(cov, drop_first=True, dtype=float)
    Xc = Xc.fillna(Xc.median())
    y = np.asarray(y).astype(int)
    lodo = datasets is not None and len(np.unique(datasets)) >= 2
    repeats = 1 if lodo else repeats
    preds = np.full((repeats, len(y)), np.nan)
    for r in range(repeats):
        for tr, te in outer_splits(y, np.asarray(groups), datasets, n_folds, seed + r):
            m = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))])
            m.fit(Xc.iloc[tr], y[tr])
            preds[r, te] = m.predict_proba(Xc.iloc[te])[:, 1]
    oof = np.nanmean(preds, axis=0)
    if lodo:
        per = [roc_auc_score(y[datasets == d], oof[datasets == d]) for d in np.unique(datasets)
               if len(np.unique(y[datasets == d])) == 2]
        return float(np.mean(per)), oof
    return float(np.mean([roc_auc_score(y, p) for p in preds])), oof


def calibration_summary(y, p):
    return {"brier": float(brier_score_loss(y, p)),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def fit_final(X: pd.DataFrame, y, groups, k_grid, C_grid, seed=0):
    search = make_search(np.asarray(y), np.asarray(groups), k_grid, C_grid, seed)
    search.fit(X.to_numpy(dtype=float), np.asarray(y).astype(int), groups=np.asarray(groups))
    est = search.best_estimator_
    mask = est.named_steps["select"].get_support()
    coefs = pd.DataFrame({"gene": np.asarray(X.columns)[mask],
                          "coefficient": est.named_steps["clf"].coef_.ravel()})
    coefs = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
    return est, search.best_params_, coefs
