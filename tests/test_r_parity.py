"""The R pseudobulk script must give exactly the same matrix as the Python implementation."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from irsig import pseudobulk as PB, simulate

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")


def test_r_matches_python(tmp_path):
    world = simulate.make_world(n_genes=200, seed=0)
    adata = simulate.simulate_singlecell(world, n_pd=4, n_ctrl=4, cells_per_donor=150, seed=1)
    scipy.io.mmwrite(tmp_path / "counts.mtx", adata.X.T.tocoo())
    adata.obs.to_csv(tmp_path / "meta.csv")
    (tmp_path / "genes.txt").write_text("\n".join(adata.var_names) + "\n")
    build = (f'library(Matrix); m <- as(readMM("{tmp_path}/counts.mtx"), "CsparseMatrix");'
             f'meta <- read.csv("{tmp_path}/meta.csv", row.names = 1);'
             f'rownames(m) <- readLines("{tmp_path}/genes.txt"); colnames(m) <- rownames(meta);'
             f'saveRDS(list(counts = m, meta = meta), "{tmp_path}/toy.rds")')
    subprocess.run(["Rscript", "-e", build], check=True)
    subprocess.run(["Rscript", str(ROOT / "scripts" / "seurat_pseudobulk.R"), f"rds={tmp_path}/toy.rds",
                    "donor_col=donor", "celltype_col=cell_type", f"out={tmp_path}/r"], check=True)

    r = pd.read_csv(tmp_path / "r" / "pseudobulk_counts.csv", index_col=0)
    py, py_meta, _ = PB.pseudobulk(adata, "donor", "cell_type", min_cells=20)
    assert set(r.columns) == set(py.columns)
    assert np.array_equal(r.loc[py.index, py.columns].to_numpy(), py.to_numpy())
    r_meta = pd.read_csv(tmp_path / "r" / "pseudobulk_metadata.csv", index_col=0)
    assert (r_meta.loc[py_meta.index, "n_cells"] == py_meta["n_cells"]).all()
