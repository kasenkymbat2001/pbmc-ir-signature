"""Donor x cell-type pseudobulk aggregation of single-cell raw counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

ALL_CELLS = "ALL_PBMC"


def _counts(adata, layer):
    X = adata.layers[layer] if layer else adata.X
    X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
    sample = X[: min(1000, X.shape[0])].data
    if sample.size and not np.allclose(sample, np.round(sample)):
        raise ValueError("Pseudobulk needs raw integer counts; pass --layer with raw counts.")
    return X


def _sum_by(X, labels: pd.Series) -> tuple[pd.Index, sparse.csr_matrix]:
    codes, uniques = pd.factorize(labels, sort=True)
    ind = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                            shape=(len(uniques), len(codes)))
    return pd.Index(uniques), ind @ X


def pseudobulk(adata, donor_col: str, celltype_col: str | None = None, layer: str | None = None,
               min_cells: int = 20, include_all: bool = True):
    """Aggregate raw counts into donor x cell-type profiles.

    Returns
    -------
    counts : DataFrame genes x pseudobulk samples (summed raw counts)
    meta   : DataFrame per pseudobulk sample (donor, cell_type, n_cells, donor-level obs columns)
    props  : DataFrame donor x cell type proportions (None if no celltype_col)
    """
    obs = adata.obs.copy()
    if donor_col not in obs:
        raise KeyError(f"{donor_col!r} not in adata.obs")
    X = _counts(adata, layer)
    genes = pd.Index(adata.var_names.astype(str))
    obs[donor_col] = obs[donor_col].astype(str)

    blocks, metas = [], []
    if celltype_col:
        if celltype_col not in obs:
            raise KeyError(f"{celltype_col!r} not in adata.obs")
        obs[celltype_col] = obs[celltype_col].astype(str)
        key = obs[donor_col] + "|" + obs[celltype_col]
        names, summed = _sum_by(X, key)
        n = key.value_counts().reindex(names).to_numpy()
        donor, ct = zip(*[s.split("|", 1) for s in names])
        blocks.append((names, summed))
        metas.append(pd.DataFrame({"donor": donor, "cell_type": ct, "n_cells": n}, index=names))
    if include_all or not celltype_col:
        names, summed = _sum_by(X, obs[donor_col])
        n = obs[donor_col].value_counts().reindex(names).to_numpy()
        pb_names = names + "|" + ALL_CELLS
        blocks.append((pb_names, summed))
        metas.append(pd.DataFrame({"donor": names, "cell_type": ALL_CELLS, "n_cells": n}, index=pb_names))

    mat = sparse.vstack([b[1] for b in blocks]).toarray()
    index = np.concatenate([np.asarray(b[0]) for b in blocks])
    counts = pd.DataFrame(mat.T, index=genes, columns=index)
    meta = pd.concat(metas)
    keep = meta["n_cells"] >= min_cells
    counts, meta = counts.loc[:, keep], meta.loc[keep]

    # Carry over donor-level metadata (columns constant within each donor).
    donor_level = [c for c in obs.columns if c not in (donor_col, celltype_col)
                   and (obs.groupby(donor_col, observed=True)[c].nunique(dropna=False) <= 1).all()]
    if donor_level:
        dmeta = obs.groupby(donor_col, observed=True)[donor_level].first()
        meta = meta.join(dmeta, on="donor")
    meta.index.name = "sample_id"

    props = None
    if celltype_col:
        props = pd.crosstab(obs[donor_col], obs[celltype_col], normalize="index")
        props.index.name = "donor"
    return counts, meta, props


def log_cpm(counts: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    lib = counts.sum(axis=0).replace(0, np.nan)
    return np.log2(counts / lib * 1e6 + prior)


def detected_genes(counts: pd.DataFrame, min_frac: float = 0.5, min_count: int = 1) -> list[str]:
    """Genes with >= min_count reads in at least min_frac of pseudobulk samples."""
    frac = (counts >= min_count).mean(axis=1)
    return sorted(frac.index[frac >= min_frac].astype(str))
