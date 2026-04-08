"""Shared test fixtures – synthetic h5ad files for the sjanpy test suite."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="session")
def tmp_h5ad_dir(tmp_path_factory):
    """Create a temporary directory with three synthetic h5ad files.

    Returns the *Path* to the temp directory.  Files inside:
      - sparse_rawX.h5ad  (200 cells, 50 genes, CSR in raw.X)
      - dense_X.h5ad      (100 cells, 30 genes, dense Poisson in .X)
      - tiny.h5ad          (10 cells, 5 genes, sparse in .X and raw.X)
    """
    rng = np.random.default_rng(42)
    tmpdir = tmp_path_factory.mktemp("h5ad_fixtures")

    # ── sparse_rawX.h5ad ──────────────────────────────────────────
    n_obs, n_vars = 200, 50
    counts = csr_matrix(rng.poisson(1.5, size=(n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice(["T-cell", "B-cell", "Monocyte", "NK"], size=n_obs)
            ),
            "batch": pd.Categorical(
                rng.choice(["batch0", "batch1", "batch2"], size=n_obs)
            ),
            "tissue": pd.Categorical(
                rng.choice(["lung", "blood"], size=n_obs)
            ),
            "extra_col": rng.standard_normal(n_obs).astype(np.float32),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=csr_matrix((n_obs, n_vars), dtype=np.float32), obs=obs, var=var)
    adata.raw = ad.AnnData(X=counts, var=var)
    adata.obsm["X_umap"] = rng.standard_normal((n_obs, 2)).astype(np.float32)
    adata.write_h5ad(tmpdir / "sparse_rawX.h5ad")

    # ── dense_X.h5ad ─────────────────────────────────────────────
    n_obs, n_vars = 100, 30
    X_dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice(["Epithelial", "Fibroblast", "Immune"], size=n_obs)
            ),
            "batch": pd.Categorical(
                rng.choice(["A", "B"], size=n_obs)
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X_dense, obs=obs, var=var)
    adata.write_h5ad(tmpdir / "dense_X.h5ad")

    # ── tiny.h5ad ────────────────────────────────────────────────
    n_obs, n_vars = 10, 5
    counts = csr_matrix(rng.poisson(3.0, size=(n_obs, n_vars)).astype(np.float32))
    cell_types = ["Common"] * 9 + ["Rare"]
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(cell_types),
            "batch": pd.Categorical(["only_batch"] * n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.raw = ad.AnnData(X=counts.copy(), var=var)
    adata.write_h5ad(tmpdir / "tiny.h5ad")

    return tmpdir
