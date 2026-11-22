"""Command-line interface: prepare-geo | pseudobulk | train | apply | demo."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from . import __version__, geo, model as M, plotting, pseudobulk as PB, score as S


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _floats(s):
    return [float(x) for x in s.split(",")]


def _ints(s):
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------- prepare-geo
def cmd_prepare_geo(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    expr, meta = geo.read_series_matrix(a.series_matrix)
    if a.platform_table:
        ann = geo.read_platform_table(a.platform_table)
        expr = geo.collapse_probes(expr, ann, a.id_col, a.symbol_col)
    expr = geo.log2_if_needed(expr)
    meta["dataset"] = a.dataset
    meta["donor"] = meta.index
    expr.to_csv(out / "expression.csv")
    meta.to_csv(out / "metadata.csv")
    print(f"{expr.shape[0]} genes x {expr.shape[1]} samples -> {out}")
    print("Metadata columns:", ", ".join(meta.columns))


# ----------------------------------------------------------------- pseudobulk
def cmd_pseudobulk(a):
    import anndata as ad
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(a.h5ad)
    counts, meta, props = PB.pseudobulk(adata, a.donor_col, a.celltype_col, a.layer, a.min_cells)
    counts.index.name = "gene"
    counts.to_csv(out / "pseudobulk_counts.csv")
    meta.to_csv(out / "pseudobulk_metadata.csv")
    if props is not None:
        props.to_csv(out / "cell_type_proportions.csv")
    all_cols = meta.index[meta["cell_type"] == PB.ALL_CELLS]
    genes = PB.detected_genes(counts[all_cols], min_frac=a.detect_frac)
    (out / "detected_genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    print(f"{counts.shape[1]} pseudobulk samples, {len(genes)} detected genes -> {out}")


# ---------------------------------------------------------------------- train
def _load_training(a):
    if len(a.expr) != len(a.meta):
        sys.exit("--expr and --meta need the same number of files")
    exprs, metas = [], []
    for e, m in zip(a.expr, a.meta):
        exprs.append(pd.read_csv(e, index_col=0))
        md = pd.read_csv(m, index_col=0)
        md.index = md.index.astype(str)
        metas.append(md)
    genes = set.intersection(*[set(e.index.astype(str)) for e in exprs])
    if a.restrict_genes:
        keep = set(Path(a.restrict_genes).read_text().split())
        genes &= keep
    genes = sorted(genes)
    expr = pd.concat([e.loc[genes] for e in exprs], axis=1)
    meta = pd.concat(metas)
    expr = expr.dropna(axis=0, how="any")
    return expr, meta


def cmd_train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    expr, meta = _load_training(a)
    lab = meta[a.label_col].astype(str)
    neg = lab.isin(a.negative) if a.negative else ~lab.isin(a.positive)
    keep = lab.isin(a.positive) | neg
    meta = meta.loc[keep]
    X = expr[meta.index].T
    y = lab.loc[meta.index].isin(a.positive).astype(int).to_numpy()
    groups = meta[a.group_col].astype(str).to_numpy() if a.group_col else meta.index.to_numpy()
    datasets = None
    if a.lodo:
        datasets = meta[a.dataset_col].astype(str).to_numpy()
        if len(np.unique(datasets)) < 2:
            print("Only one dataset found; falling back to grouped K-fold CV.")
            datasets = None
    k_grid, C_grid = _ints(a.k_grid), _floats(a.c_grid)
    print(f"Training on {X.shape[0]} samples ({y.sum()} IR / {len(y)-y.sum()} non-IR), "
          f"{X.shape[1]} genes, {len(np.unique(groups))} donors")

    res = M.nested_cv(X, y, groups, k_grid, C_grid, datasets, a.outer_folds, a.repeats, a.seed)
    auc = res.primary_auc
    auc_pooled = float(roc_auc_score(y, res.oof))
    # Bootstrap donors within each dataset so the CI matches the primary (per-dataset) metric.
    ci = M.bootstrap_auc_ci(y, res.oof, groups=groups, seed=a.seed, datasets=datasets)
    metrics = {"n_samples": int(len(y)), "n_positive": int(y.sum()), "n_donors": int(len(np.unique(groups))),
               "n_genes_universe": int(X.shape[1]),
               "cv_scheme": "leave-one-dataset-out" if datasets is not None else
               f"{a.outer_folds}-fold StratifiedGroupKFold x {a.repeats} repeats (donor-grouped)",
               "primary_auroc": auc,
               "primary_auroc_definition": "mean within-held-out-dataset AUROC" if datasets is not None
               else "mean AUROC over CV repeats",
               "auroc_95ci_donor_bootstrap": ci,
               "auroc_sd_over_repeats": None if datasets is not None else float(np.std(res.auc_per_repeat)),
               "auroc_pooled_oof_predictions": auc_pooled,
               "calibration": M.calibration_summary(y, res.oof), "lodo_auroc": res.lodo_auc}

    p_base = auc_base = None
    if a.covariates:
        cov = meta[a.covariates]
        auc_base, p_base = M.covariate_baseline(cov, y, groups, datasets, a.outer_folds, a.repeats, a.seed)
        metrics["covariate_only_auroc"] = auc_base
        # Does the out-of-fold signature score add information beyond covariates?
        d = pd.get_dummies(cov, drop_first=True, dtype=float)
        d = d.fillna(d.median())
        eps = 1e-6
        d["signature_logit"] = np.log((res.oof + eps) / (1 - res.oof + eps))
        try:
            fit = sm.Logit(y, sm.add_constant(d)).fit(disp=0)
            metrics["signature_beyond_covariates"] = {
                "odds_ratio_per_logit": float(np.exp(fit.params["signature_logit"])),
                "p_value": float(fit.pvalues["signature_logit"])}
        except Exception as exc:  # perfect separation etc.
            metrics["signature_beyond_covariates"] = {"error": str(exc)}

    if a.n_perm > 0:
        print(f"Permutation test ({a.n_perm} label shuffles, full nested CV each)...")
        pval, null = M.permutation_test(X, y, groups, k_grid, C_grid, auc,
                                        a.n_perm, datasets, a.outer_folds, a.seed)
        metrics["permutation_p"] = pval
        metrics["permutation_null_mean_auroc"] = float(null.mean())
        plotting.permutation_plot(null, auc, pval, out / "permutation_null.png")

    est, params, coefs = M.fit_final(X, y, groups, k_grid, C_grid, a.seed)
    metrics["final_model_params"] = params
    bundle = {"model": est, "genes": list(X.columns), "params": params, "label_col": a.label_col,
              "positive": a.positive, "irsig_version": __version__, "python": platform.python_version()}
    joblib.dump(bundle, out / "model.joblib")

    pd.DataFrame({"y": y, "p_oof": res.oof, "group": groups}, index=X.index).to_csv(out / "oof_predictions.csv")
    pd.DataFrame(res.fold_records).to_csv(out / "cv_folds.csv", index=False)
    res.gene_frequency.to_csv(out / "gene_selection_frequency.csv", header=True)
    coefs.to_csv(out / "final_signature_coefficients.csv", index=False)
    plotting.roc_plot(y, res.oof, auc, out / "roc.png", p_base, auc_base, datasets=datasets)
    plotting.gene_frequency_plot(res.gene_frequency, out / "gene_stability.png")
    _write_json(metrics, out / "metrics.json")
    print(json.dumps(metrics, indent=2, default=float))


# ---------------------------------------------------------------------- apply
def cmd_apply(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(a.model)
    expr = pd.read_csv(a.expr, index_col=0)
    expr.index = expr.index.astype(str)
    meta = pd.read_csv(a.meta, index_col=0)
    X, coverage = S.align_to_model(expr, bundle["genes"], a.min_coverage)
    scores = S.score_samples(bundle["model"], X).join(meta)
    if a.proportions:
        props = pd.read_csv(a.proportions, index_col=0).add_prefix("prop_")
        props.index = props.index.astype(str)
        scores = scores.join(props, on=a.donor_col)
    scores.to_csv(out / "scores.csv")
    tests = S.association_tests(scores, a.group_col, a.case, a.control, a.covariates,
                                a.celltype_col, a.severity_col)
    tests.to_csv(out / "association_tests.csv", index=False)
    if a.celltype_col in scores:
        plotting.score_by_group_plot(scores, a.group_col, a.case, a.control, out / "scores_by_group.png",
                                     a.celltype_col)
    _write_json({"gene_coverage": coverage, "n_scored": int(len(scores))}, out / "apply_summary.json")
    print(f"Gene coverage {coverage:.1%}")
    print(tests.to_string(index=False))


# ----------------------------------------------------------------------- demo
def cmd_demo(a):
    from . import simulate
    out = Path(a.out)
    data = out / "synthetic_data"; data.mkdir(parents=True, exist_ok=True)
    world = simulate.make_world(seed=a.seed)
    for name, plat, seed in [("cohortA", "array", 11), ("cohortB", "rnaseq", 12)]:
        e, m = simulate.simulate_bulk(world, n_per_class=20, dataset=name, platform=plat, seed=seed)
        e.to_csv(data / f"{name}_expression.csv"); m.to_csv(data / f"{name}_metadata.csv")
    simulate.simulate_singlecell(world, seed=a.seed + 3).write_h5ad(data / "pd_pbmc_synthetic.h5ad")

    run = lambda argv: main(argv)
    run(["pseudobulk", "--h5ad", str(data / "pd_pbmc_synthetic.h5ad"), "--donor-col", "donor",
         "--celltype-col", "cell_type", "--out", str(out / "pseudobulk")])
    run(["train", "--expr", str(data / "cohortA_expression.csv"), str(data / "cohortB_expression.csv"),
         "--meta", str(data / "cohortA_metadata.csv"), str(data / "cohortB_metadata.csv"),
         "--label-col", "group", "--positive", "insulin_resistant", "--group-col", "donor",
         "--restrict-genes", str(out / "pseudobulk" / "detected_genes.txt"),
         "--covariates", "bmi", "age", "sex", "--lodo", "--n-perm", str(a.n_perm), "--out", str(out / "train")])
    run(["apply", "--model", str(out / "train" / "model.joblib"),
         "--expr", str(out / "pseudobulk" / "pseudobulk_counts.csv"),
         "--meta", str(out / "pseudobulk" / "pseudobulk_metadata.csv"),
         "--proportions", str(out / "pseudobulk" / "cell_type_proportions.csv"),
         "--group-col", "diagnosis", "--case", "PD", "--control", "Control",
         "--covariates", "age", "sex", "prop_Monocyte", "--severity-col", "updrs_iii",
         "--out", str(out / "apply")])
    # Sanity check only possible on simulated data: does the score recover the hidden IR state?
    sc = pd.read_csv(out / "apply" / "scores.csv", index_col=0)
    allp = sc[sc["cell_type"] == PB.ALL_CELLS]
    truth = {"auroc_score_vs_hidden_simulated_ir_ALL_PBMC":
             float(roc_auc_score(allp["simulated_ir"], allp["ir_score"]))}
    _write_json(truth, out / "apply" / "simulation_ground_truth_check.json")
    print(truth)
    print(f"\nSYNTHETIC demo finished -> {out}. These numbers are not findings.")


def build_parser():
    p = argparse.ArgumentParser(prog="irsig", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("prepare-geo", help="GEO series matrix -> gene x sample CSV + metadata")
    g.add_argument("--series-matrix", required=True)
    g.add_argument("--platform-table", help="GPL annotation table for probe -> gene mapping")
    g.add_argument("--id-col", default="ID")
    g.add_argument("--symbol-col", default="GENE_SYMBOL")
    g.add_argument("--dataset", required=True, help="Short dataset label, e.g. GSE87005")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_prepare_geo)

    b = sub.add_parser("pseudobulk", help="h5ad raw counts -> donor x cell-type pseudobulk")
    b.add_argument("--h5ad", required=True)
    b.add_argument("--donor-col", required=True)
    b.add_argument("--celltype-col")
    b.add_argument("--layer", help="Layer with raw counts (default: X)")
    b.add_argument("--min-cells", type=int, default=20)
    b.add_argument("--detect-frac", type=float, default=0.5)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_pseudobulk)

    t = sub.add_parser("train", help="Nested group-aware CV + final model")
    t.add_argument("--expr", nargs="+", required=True, help="gene x sample CSV(s)")
    t.add_argument("--meta", nargs="+", required=True, help="sample metadata CSV(s), first column = sample id")
    t.add_argument("--label-col", required=True)
    t.add_argument("--positive", nargs="+", required=True, help="Label value(s) meaning insulin resistant")
    t.add_argument("--negative", nargs="+", help="Label value(s) for the other class (default: all others)")
    t.add_argument("--group-col", help="Donor column (default: one donor per sample)")
    t.add_argument("--dataset-col", default="dataset")
    t.add_argument("--lodo", action="store_true", help="Leave-one-dataset-out outer loop")
    t.add_argument("--restrict-genes", help="Text file of genes detected in the target cohort")
    t.add_argument("--covariates", nargs="*", default=[])
    t.add_argument("--k-grid", default="10,25,50,100")
    t.add_argument("--c-grid", default="0.05,0.2,1.0")
    t.add_argument("--outer-folds", type=int, default=5)
    t.add_argument("--repeats", type=int, default=5)
    t.add_argument("--n-perm", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("apply", help="Score a new cohort and test associations")
    s.add_argument("--model", required=True)
    s.add_argument("--expr", required=True, help="gene x sample CSV (counts, CPM or log - rank-invariant)")
    s.add_argument("--meta", required=True)
    s.add_argument("--proportions", help="donor x cell-type proportions CSV (adds prop_<type> columns)")
    s.add_argument("--donor-col", default="donor")
    s.add_argument("--group-col", required=True)
    s.add_argument("--case", required=True)
    s.add_argument("--control", required=True)
    s.add_argument("--covariates", nargs="*", default=[])
    s.add_argument("--celltype-col", default="cell_type")
    s.add_argument("--severity-col")
    s.add_argument("--min-coverage", type=float, default=0.9)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_apply)

    d = sub.add_parser("demo", help="Run the whole pipeline on SYNTHETIC data")
    d.add_argument("--out", default="results/demo")
    d.add_argument("--n-perm", type=int, default=50)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
