import gzip

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from irsig import geo, model as M, pseudobulk as PB, score as S
from irsig.features import TopKWelch, WithinSampleRank


# ------------------------------------------------------------------ GEO parsing
SERIES = """!Series_title\t"toy"
!Sample_title\t"s1"\t"s2"\t"s3"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"
!Sample_characteristics_ch1\t"group: IR"\t"group: IS"\t"group: IR"
!Sample_characteristics_ch1\t"bmi: 31"\t"bmi: 24"\t"bmi: 29"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"p1"\t1\t2\t3
"p2"\t5\t6\t7
"p3"\t0.5\t0.5\t0.5
"p4"\t9\t9\t9
!series_matrix_table_end
"""


def test_read_series_matrix_and_collapse(tmp_path):
    f = tmp_path / "GSE_toy_series_matrix.txt.gz"
    with gzip.open(f, "wt") as fh:
        fh.write(SERIES)
    expr, meta = geo.read_series_matrix(f)
    assert expr.shape == (4, 3)
    assert list(meta["group"]) == ["IR", "IS", "IR"]
    assert meta.loc["GSM2", "bmi"] == "24"
    ann = pd.DataFrame({"ID": ["p1", "p2", "p3", "p4"], "GENE_SYMBOL": ["INSR", "INSR", "AKT2", "A///B"]})
    g = geo.collapse_probes(expr, ann, "ID", "GENE_SYMBOL")
    assert sorted(g.index) == ["AKT2", "INSR"]          # ambiguous probe dropped
    assert g.loc["INSR"].tolist() == [5, 6, 7]           # highest-mean probe kept


# ------------------------------------------------------------------ pseudobulk
def _toy_adata():
    import anndata as ad
    X = sparse.csr_matrix(np.array([[1, 0], [2, 1], [0, 3], [4, 4], [1, 1]], dtype=np.float32))
    obs = pd.DataFrame({"donor": ["a", "a", "a", "b", "b"], "ct": ["T", "T", "M", "T", "T"],
                        "dx": ["PD", "PD", "PD", "Ctrl", "Ctrl"]}, index=[f"c{i}" for i in range(5)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g1", "g2"]))


def test_pseudobulk_sums_and_metadata():
    counts, meta, props = PB.pseudobulk(_toy_adata(), "donor", "ct", min_cells=1)
    assert counts["a|T"].tolist() == [3, 1]
    assert counts["b|ALL_PBMC"].tolist() == [5, 5]
    assert meta.loc["a|M", "n_cells"] == 1 and meta.loc["a|M", "dx"] == "PD"
    assert props.loc["a", "M"] == pytest.approx(1 / 3)
    counts2, _, _ = PB.pseudobulk(_toy_adata(), "donor", "ct", min_cells=2)
    assert "a|M" not in counts2.columns


def test_pseudobulk_rejects_normalised_data():
    a = _toy_adata()
    a.X = a.X * 0.37
    with pytest.raises(ValueError):
        PB.pseudobulk(a, "donor", "ct")


# ------------------------------------------------------------------ features
def test_rank_is_invariant_to_monotone_platform_transform():
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(5, 100))
    r1 = WithinSampleRank().fit_transform(X)
    r2 = WithinSampleRank().fit_transform(np.log2(X * 300 + 1))
    assert np.allclose(r1, r2)


def test_topk_picks_informative_genes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 200))
    y = np.r_[np.ones(20), np.zeros(20)]
    X[y == 1, :5] += 3
    sel = TopKWelch(k=5).fit(X, y)
    assert set(np.flatnonzero(sel.get_support())) == set(range(5))


# ------------------------------------------------------------------ leakage
def test_nested_cv_is_honest_on_pure_noise():
    """Feature selection on the full data before CV inflates AUROC on noise; ours must not."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    n, p = 40, 3000
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"g{i}" for i in range(p)])
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    ids = np.arange(n).astype(str)

    honest = M.nested_cv(X, y, ids, [20], [1.0], n_folds=5, repeats=2, seed=0).primary_auc

    leaky_cols = TopKWelch(k=20).fit(X.to_numpy(), y).get_support()   # uses ALL labels
    leaky_p = cross_val_predict(LogisticRegression(max_iter=2000), X.loc[:, leaky_cols], y,
                                cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    leaky = roc_auc_score(y, leaky_p)
    assert honest < 0.75
    assert leaky > 0.9
    assert leaky - honest > 0.25


def test_outer_splits_never_split_a_donor():
    y = np.repeat([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 3)
    donors = np.repeat([f"d{i}" for i in range(10)], 3)
    for tr, te in M.outer_splits(y, donors, n_folds=5, seed=0):
        assert not set(donors[tr]) & set(donors[te])


def test_lodo_holds_out_whole_datasets():
    y = np.tile([1, 0], 10)
    ds = np.repeat(["A", "B"], 10)
    splits = M.outer_splits(y, np.arange(20).astype(str), ds)
    assert len(splits) == 2
    for tr, te in splits:
        assert len(set(ds[te])) == 1 and not set(ds[tr]) & set(ds[te])


def test_permutation_is_donor_level():
    rng = np.random.default_rng(0)
    y = np.repeat([1, 0, 1, 0], 3)
    donors = np.repeat(list("abcd"), 3)
    yp = M.permute_by_group(y, donors, rng)
    assert all(len(set(yp[donors == d])) == 1 for d in "abcd")
    assert yp.sum() == y.sum()


# ------------------------------------------------------------------ scoring
def test_align_refuses_low_coverage_and_fills_missing():
    expr = pd.DataFrame(np.arange(12).reshape(4, 3), index=list("abcd"), columns=["s1", "s2", "s3"])
    with pytest.raises(ValueError):
        S.align_to_model(expr, list("abcdefgh"), min_coverage=0.9)
    X, cov = S.align_to_model(expr, list("abcde"), min_coverage=0.8)
    assert cov == 0.8 and X.shape == (3, 5)
    assert X.loc["s1", "e"] == pytest.approx(np.median([0, 3, 6, 9]))


def test_association_detects_shift():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"ir_score": np.r_[rng.normal(1, 1, n), rng.normal(0, 1, n)],
                       "dx": ["PD"] * n + ["Ctrl"] * n, "age": rng.normal(65, 8, 2 * n),
                       "cell_type": "ALL_PBMC"})
    res = S.association_tests(df, "dx", "PD", "Ctrl", ["age"])
    assert res.iloc[0]["beta_case_vs_control_sd"] > 0 and res.iloc[0]["p_ols"] < 0.01
