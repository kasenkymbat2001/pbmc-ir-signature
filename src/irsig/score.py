"""Apply a trained signature to a new cohort and test group / severity associations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def align_to_model(expr_genes_x_samples: pd.DataFrame, gene_universe: list[str],
                   min_coverage: float = 0.9) -> tuple[pd.DataFrame, float]:
    """Return samples x model-genes. Missing genes get the sample's median value
    (i.e. a neutral mid rank) and coverage is reported; below min_coverage we refuse."""
    present = [g for g in gene_universe if g in expr_genes_x_samples.index]
    coverage = len(present) / len(gene_universe)
    if coverage < min_coverage:
        raise ValueError(f"Only {coverage:.1%} of model genes present (< {min_coverage:.0%}). "
                         "Retrain with --restrict-genes <target detected genes>.")
    X = expr_genes_x_samples.reindex(gene_universe).T
    X = X.T.fillna(X.median(axis=1)).T
    return X, coverage


def score_samples(model, X: pd.DataFrame) -> pd.DataFrame:
    Xv = X.to_numpy(dtype=float)
    return pd.DataFrame({"ir_score": model.decision_function(Xv),
                         "ir_probability": model.predict_proba(Xv)[:, 1]}, index=X.index)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def association_tests(df: pd.DataFrame, group_col: str, case: str, control: str,
                      covariates: list[str] | None = None, celltype_col: str = "cell_type",
                      severity_col: str | None = None) -> pd.DataFrame:
    """Per cell type: OLS ir_score ~ case + covariates, Mann-Whitney, optional Spearman with severity.
    Scores are z-scored within cell type so effects are comparable across cell types."""
    covariates = [c for c in (covariates or []) if c in df.columns]
    rows = []
    cts = df[celltype_col].unique() if celltype_col in df else ["all"]
    for ct in cts:
        sub = df[df[celltype_col] == ct].copy() if celltype_col in df else df.copy()
        sub = sub[sub[group_col].astype(str).isin([case, control])]
        if sub[group_col].nunique() < 2 or len(sub) < 6:
            continue
        sub["z"] = (sub["ir_score"] - sub["ir_score"].mean()) / sub["ir_score"].std(ddof=1)
        sub["is_case"] = (sub[group_col].astype(str) == case).astype(int)
        d = sub.rename(columns={c: _safe(c) for c in covariates})
        cov = [_safe(c) for c in covariates]
        formula = "z ~ is_case" + "".join(f" + {c}" for c in cov)
        fit = smf.ols(formula, data=d.dropna(subset=["z", "is_case", *cov])).fit()
        lo, hi = fit.conf_int().loc["is_case"]
        a = sub.loc[sub.is_case == 1, "ir_score"]
        b = sub.loc[sub.is_case == 0, "ir_score"]
        row = {"cell_type": ct, "n_case": len(a), "n_control": len(b),
               "beta_case_vs_control_sd": fit.params["is_case"], "ci_low": lo, "ci_high": hi,
               "p_ols": fit.pvalues["is_case"], "covariates": ",".join(covariates) or "none",
               "p_mannwhitney": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue}
        if severity_col and severity_col in sub:
            cases = sub[sub.is_case == 1].dropna(subset=[severity_col])
            if len(cases) >= 5:
                rho, p = stats.spearmanr(cases["ir_score"], pd.to_numeric(cases[severity_col], errors="coerce"),
                                         nan_policy="omit")
                row.update({"spearman_rho_severity": rho, "p_severity": p})
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_ols"] = multipletests(out["p_ols"], method="fdr_bh")[1]
        out = out.sort_values("p_ols")
    return out
