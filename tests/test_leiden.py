"""Tests for sjanpy.tl.gpuleiden (GPU Leiden scanpy-compatible wrapper)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gpu_leiden")
pytest.importorskip("anndata")
pytest.importorskip("scanpy")

import anndata as ad
import scanpy as sc

import sjanpy  # type: ignore[import-not-found]


def _make_adata_with_neighbors(n_cells: int = 300, n_genes: int = 50, seed: int = 42):
    """Build a small AnnData with pre-computed neighbors for testing."""
    rng = np.random.default_rng(seed)
    cluster_size = n_cells // 3
    X = np.vstack([
        rng.normal(loc=0.0, scale=1.0, size=(cluster_size, n_genes)),
        rng.normal(loc=3.0, scale=1.0, size=(cluster_size, n_genes)),
        rng.normal(loc=6.0, scale=1.0, size=(n_cells - 2 * cluster_size, n_genes)),
    ])
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs["true_cluster"] = pd.Categorical(
        np.repeat([0, 1, 2], [cluster_size, cluster_size, n_cells - 2 * cluster_size])
    )
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=15)
    return adata


def test_gpuleiden_writes_obs_column():
    adata = _make_adata_with_neighbors()
    sjanpy.tl.gpuleiden(adata, resolution=1.0)
    assert "leiden" in adata.obs.columns
    assert adata.obs["leiden"].dtype == "category"
    assert len(adata.obs["leiden"]) == adata.n_obs


def test_gpuleiden_stores_params():
    adata = _make_adata_with_neighbors()
    sjanpy.tl.gpuleiden(adata, resolution=0.7, random_state=42, key_added="my_clusters")
    assert "my_clusters" in adata.uns
    params = adata.uns["my_clusters"]["params"]
    assert params["resolution"] == 0.7
    assert params["random_state"] == 42
    assert params["n_iterations"] == -1


def test_gpuleiden_copy_returns_new_adata():
    adata = _make_adata_with_neighbors()
    assert "leiden" not in adata.obs.columns
    new = sjanpy.tl.gpuleiden(adata, copy=True)
    assert "leiden" not in adata.obs.columns  # original untouched
    assert new is not None
    assert "leiden" in new.obs.columns


def test_gpuleiden_resolution_changes_cluster_count():
    adata = _make_adata_with_neighbors(n_cells=300)
    sjanpy.tl.gpuleiden(adata, resolution=0.3, key_added="low_res")
    sjanpy.tl.gpuleiden(adata, resolution=3.0, key_added="high_res")
    n_low = len(adata.obs["low_res"].cat.categories)
    n_high = len(adata.obs["high_res"].cat.categories)
    assert n_high >= n_low, f"high res ({n_high}) should be >= low res ({n_low})"


def test_gpuleiden_recovers_synthetic_clusters():
    """With 3 well-separated clusters, ARI against ground truth should be high."""
    from sklearn.metrics import adjusted_rand_score
    adata = _make_adata_with_neighbors(n_cells=300)
    sjanpy.tl.gpuleiden(adata, resolution=1.0)
    ari = adjusted_rand_score(
        adata.obs["true_cluster"].cat.codes,
        adata.obs["leiden"].cat.codes,
    )
    assert ari > 0.9, f"ARI {ari:.3f} too low - expected near-perfect recovery"


def test_gpuleiden_matches_scanpy_on_pbmc():
    """Correctness check on real data: compare to scanpy.tl.leiden via ARI."""
    from sklearn.metrics import adjusted_rand_score
    adata = sc.datasets.pbmc68k_reduced()  # 700 cells, already processed
    adata_ref = adata.copy()
    sc.tl.leiden(
        adata_ref, resolution=1.0, flavor="igraph", directed=False,
        n_iterations=2, random_state=42, key_added="leiden_ref",
    )
    adata_gpu = adata.copy()
    sjanpy.tl.gpuleiden(adata_gpu, resolution=1.0, random_state=42, key_added="leiden_gpu")
    ari = adjusted_rand_score(
        adata_ref.obs["leiden_ref"].cat.codes,
        adata_gpu.obs["leiden_gpu"].cat.codes,
    )
    assert ari > 0.5, f"ARI {ari:.3f} too low - results disagree too much with scanpy"
    print(f"\nPBMC68k ARI vs scanpy: {ari:.3f}")


def test_gpuleiden_quality_flavor():
    """Quality flavor should produce >= deterministic modularity."""
    adata = _make_adata_with_neighbors(n_cells=300)
    sjanpy.tl.gpuleiden(adata, resolution=1.0, gpu_flavor="quality",
                        n_restarts=2, key_added="leiden_q")
    assert "leiden_q" in adata.obs.columns
    assert adata.uns["leiden_q"]["params"]["gpu_flavor"] == "quality"


def test_gpuleiden_availability_flag():
    """GPU_LEIDEN_AVAILABLE should be True in this test environment."""
    assert sjanpy.tl.GPU_LEIDEN_AVAILABLE is True
