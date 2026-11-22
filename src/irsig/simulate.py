"""SYNTHETIC data for tests and the end-to-end demo. Never report these as findings.

Bulk PBMC cohorts and single-cell PBMC data are generated from the same cell-type
expression profiles, so a signature trained on bulk can be applied to pseudobulk.
An insulin-resistance (IR) programme acts mainly in monocytes; IR also shifts the
monocyte fraction slightly, which creates the composition confound the pipeline checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

CELL_TYPES = ["CD4_T", "CD8_T", "NK", "B", "Monocyte"]
BASE_PROPS = np.array([0.40, 0.18, 0.10, 0.12, 0.20])


def _profiles(n_genes, rng):
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    base = rng.lognormal(mean=0.0, sigma=1.5, size=n_genes)
    prof = np.vstack([base * rng.lognormal(0, 0.6, n_genes) for _ in CELL_TYPES])
    up = rng.choice(n_genes, 25, replace=False)
    down = rng.choice(np.setdiff1d(np.arange(n_genes), up), 15, replace=False)
    return genes, prof, up, down


def _ir_effect(prof, up, down, strength):
    """Return per-cell-type multiplicative IR effect (mostly monocytes, weaker in T cells)."""
    eff = np.ones_like(prof)
    per_ct = {"Monocyte": 1.0, "CD4_T": 0.3, "CD8_T": 0.3, "NK": 0.2, "B": 0.1}
    for i, ct in enumerate(CELL_TYPES):
        eff[i, up] *= 1 + strength * per_ct[ct]
        eff[i, down] /= 1 + strength * per_ct[ct]
    return eff


def make_world(n_genes=1500, seed=0):
    rng = np.random.default_rng(seed)
    genes, prof, up, down = _profiles(n_genes, rng)
    return {"genes": genes, "profiles": prof, "up": up, "down": down, "seed": seed}


def simulate_bulk(world, n_per_class=20, dataset="cohortA", strength=0.8, platform="array",
                  seed=1):
    """Bulk PBMC cohort. platform='array' gives log2 intensities with a probe-like affinity
    distortion; platform='rnaseq' gives log2 CPM. Ranks within a sample stay comparable."""
    rng = np.random.default_rng(seed)
    prof = world["profiles"]
    n = 2 * n_per_class
    ir = np.r_[np.ones(n_per_class), np.zeros(n_per_class)].astype(int)
    rng.shuffle(ir)
    bmi = np.where(ir == 1, rng.normal(30, 4, n), rng.normal(26, 4, n))
    age = rng.normal(58, 9, n).round()
    sex = rng.choice(["F", "M"], n)
    affinity = rng.lognormal(0, 0.7, prof.shape[1]) if platform == "array" else np.ones(prof.shape[1])
    X = np.empty((n, prof.shape[1]))
    for s in range(n):
        props = rng.dirichlet(BASE_PROPS * 60)
        if ir[s]:
            props = props * np.r_[1, 1, 1, 1, 1.25]
            props /= props.sum()
        p = prof * (_ir_effect(prof, world["up"], world["down"], strength) if ir[s] else 1)
        mix = props @ p * rng.lognormal(0, 0.35, prof.shape[1])  # biological noise
        if platform == "array":
            X[s] = np.log2(mix * affinity * 200 + rng.normal(30, 5, prof.shape[1]).clip(1))
        else:
            counts = rng.poisson(mix / mix.sum() * 2e7)
            X[s] = np.log2(counts / counts.sum() * 1e6 + 1)
    samples = [f"{dataset}_S{i:03d}" for i in range(n)]
    expr = pd.DataFrame(X.T, index=world["genes"], columns=samples)
    expr.index.name = "gene"
    meta = pd.DataFrame({"group": np.where(ir == 1, "insulin_resistant", "insulin_sensitive"),
                         "bmi": bmi.round(1), "age": age, "sex": sex, "dataset": dataset,
                         "donor": samples}, index=pd.Index(samples, name="sample_id"))
    return expr, meta


def simulate_singlecell(world, n_pd=18, n_ctrl=18, cells_per_donor=500, pd_ir_rate=0.6,
                        ctrl_ir_rate=0.3, strength=0.8, seed=2):
    """scRNA-seq PBMC counts. PD donors are more often 'IR' than controls (an assumption
    built into the simulation). Returns an AnnData with raw counts in X."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    prof = world["profiles"]
    donors, rows, obs = [], [], []
    for d in range(n_pd + n_ctrl):
        is_pd = d < n_pd
        ir = rng.random() < (pd_ir_rate if is_pd else ctrl_ir_rate)
        donor = f"D{d:03d}"
        props = rng.dirichlet(BASE_PROPS * 60)
        if ir:
            props = props * np.r_[1, 1, 1, 1, 1.25]
            props /= props.sum()
        p = prof * (_ir_effect(prof, world["up"], world["down"], strength) if ir else 1)
        p = p * rng.lognormal(0, 0.25, prof.shape[1])  # donor-level noise
        ct_idx = rng.choice(len(CELL_TYPES), cells_per_donor, p=props)
        depth = rng.lognormal(np.log(3000), 0.4, cells_per_donor)
        mu = p[ct_idx] / p[ct_idx].sum(axis=1, keepdims=True) * depth[:, None]
        rows.append(sparse.csr_matrix(rng.poisson(mu).astype(np.float32)))
        donors.append({"donor": donor, "diagnosis": "PD" if is_pd else "Control",
                       "age": float(rng.normal(66, 8)).__round__(), "sex": rng.choice(["F", "M"]),
                       "updrs_iii": float(rng.normal(30, 10)).__round__() if is_pd else np.nan,
                       "simulated_ir": int(ir)})
        obs.append(pd.DataFrame({"donor": donor, "cell_type": np.asarray(CELL_TYPES)[ct_idx]}))
    obs = pd.concat(obs, ignore_index=True)
    dmeta = pd.DataFrame(donors).set_index("donor")
    obs = obs.join(dmeta, on="donor")
    obs.index = [f"cell{i}" for i in range(len(obs))]
    adata = ad.AnnData(X=sparse.vstack(rows).tocsr(), obs=obs,
                       var=pd.DataFrame(index=world["genes"]))
    return adata
