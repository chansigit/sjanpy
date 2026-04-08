"""Tests for sjanpy.pp.hvg — HVG computation with stratified sampling."""

import numpy as np
import pandas as pd
import pytest

from sjanpy.pp.hvg import prepare_hvg_sample, compute_hvg


# ---------------------------------------------------------------------------
# TestPrepareHvgSample
# ---------------------------------------------------------------------------

class TestPrepareHvgSample:
    """Unit tests for prepare_hvg_sample."""

    def _make_obs(self, cell_types: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"cell_type": pd.Categorical(cell_types)},
            index=[f"cell_{i}" for i in range(len(cell_types))],
        )

    def test_returns_none_when_small(self):
        obs = self._make_obs(["A"] * 100)
        train_indices = np.arange(100)
        result = prepare_hvg_sample(obs, train_indices, "cell_type", target_size=200)
        assert result is None

    def test_returns_subset_when_large(self):
        obs = self._make_obs(["A"] * 500 + ["B"] * 500)
        train_indices = np.arange(1000)
        result = prepare_hvg_sample(obs, train_indices, "cell_type", target_size=200)
        assert result is not None
        assert len(result) == 200
        # All returned indices must be within train_indices
        assert np.all(np.isin(result, train_indices))

    def test_preserves_small_types(self):
        # 5 Rare + 500 Big cells
        cell_types = ["Rare"] * 5 + ["Big"] * 500
        obs = self._make_obs(cell_types)
        train_indices = np.arange(505)
        result = prepare_hvg_sample(
            obs, train_indices, "cell_type", target_size=100, min_cells=10
        )
        assert result is not None
        # All 5 rare cells (indices 0-4) must be in the result
        rare_indices = np.arange(5)
        assert np.all(np.isin(rare_indices, result))
        assert len(result) == 100

    def test_reproducible(self):
        obs = self._make_obs(["A"] * 300 + ["B"] * 300)
        train_indices = np.arange(600)
        r1 = prepare_hvg_sample(obs, train_indices, "cell_type", target_size=100, seed=99)
        r2 = prepare_hvg_sample(obs, train_indices, "cell_type", target_size=100, seed=99)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestComputeHvg
# ---------------------------------------------------------------------------

class TestComputeHvg:
    """Integration tests for compute_hvg using synthetic h5ad fixtures."""

    def test_returns_gene_list_and_mask(self, tmp_h5ad_dir):
        path = tmp_h5ad_dir / "tiny.h5ad"
        cell_indices = np.arange(10)
        hvg_genes, hvg_mask = compute_hvg(
            h5ad_path=path,
            matrix_source="raw.X",
            cell_indices=cell_indices,
            batch_key="batch",
            matrix_value_type="counts",
        )
        assert isinstance(hvg_genes, list)
        assert all(isinstance(g, str) for g in hvg_genes)
        assert isinstance(hvg_mask, np.ndarray)
        assert hvg_mask.dtype == bool
        assert len(hvg_mask) == 5  # tiny.h5ad has 5 genes

    def test_from_X_source(self, tmp_h5ad_dir):
        path = tmp_h5ad_dir / "dense_X.h5ad"
        cell_indices = np.arange(100)
        hvg_genes, hvg_mask = compute_hvg(
            h5ad_path=path,
            matrix_source="X",
            cell_indices=cell_indices,
            batch_key="batch",
            matrix_value_type="counts",
        )
        assert isinstance(hvg_genes, list)
        assert isinstance(hvg_mask, np.ndarray)
        assert hvg_mask.dtype == bool
        assert len(hvg_mask) == 30  # dense_X.h5ad has 30 genes

    def test_invalid_batch_key_raises(self, tmp_h5ad_dir):
        from sjanpy.pp.hvg import compute_hvg
        with pytest.raises(ValueError, match="batch_key"):
            compute_hvg(
                h5ad_path=tmp_h5ad_dir / "tiny.h5ad",
                matrix_source="raw.X",
                cell_indices=np.arange(10),
                batch_key="nonexistent_column",
            )


# ---------------------------------------------------------------------------
# TestMakeVarNamesUnique
# ---------------------------------------------------------------------------

class TestMakeVarNamesUnique:
    def test_no_duplicates_unchanged(self):
        from sjanpy.pp.hvg import _make_var_names_unique
        var = pd.DataFrame(index=["A", "B", "C"])
        result = _make_var_names_unique(var)
        assert list(result.index) == ["A", "B", "C"]

    def test_duplicates_get_suffix(self):
        from sjanpy.pp.hvg import _make_var_names_unique
        var = pd.DataFrame(index=["X", "X", "Y", "X"])
        result = _make_var_names_unique(var)
        assert list(result.index) == ["X", "X-1", "Y", "X-2"]

    def test_empty_dataframe(self):
        from sjanpy.pp.hvg import _make_var_names_unique
        var = pd.DataFrame(index=[])
        result = _make_var_names_unique(var)
        assert len(result) == 0
