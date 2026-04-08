"""Tests for sjanpy.ml.standardize."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sjanpy.ml.standardize import build_standardized_obs, build_standardized_h5ads


# ---------------------------------------------------------------------------
# TestBuildStandardizedObs
# ---------------------------------------------------------------------------

class TestBuildStandardizedObs:
    """Tests for the build_standardized_obs helper."""

    @pytest.fixture()
    def sample_obs(self):
        return pd.DataFrame(
            {
                "cell_type": ["T-cell", "B-cell", "Mono", "NK", "T-cell"],
                "batch": ["b0", "b1", "b0", "b1", "b0"],
                "tissue": ["lung", "blood", "lung", "blood", "lung"],
                "score": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            index=[f"cell_{i}" for i in range(5)],
        )

    def test_standard_columns(self, sample_obs):
        indices = np.array([0, 2, 4])
        lib = np.array([100.0, 200.0, 300.0])
        result = build_standardized_obs(
            sample_obs, indices, "cell_type", "batch", "my_dataset", lib
        )
        assert list(result.columns) == ["cell_type", "batch", "tissue", "dataset", "library_size"]
        assert list(result["cell_type"]) == ["T-cell", "Mono", "T-cell"]
        assert list(result["batch"]) == ["b0", "b0", "b0"]
        assert list(result["tissue"]) == ["lung", "lung", "lung"]
        assert (result["dataset"] == "my_dataset").all()
        np.testing.assert_array_equal(result["library_size"].values, lib)

    def test_extra_columns(self, sample_obs):
        indices = np.array([1, 3])
        lib = np.array([50.0, 60.0])
        result = build_standardized_obs(
            sample_obs, indices, "cell_type", "batch", "ds",
            lib, extra_columns={"my_score": "score"},
        )
        assert "my_score" in result.columns
        np.testing.assert_allclose(result["my_score"].values, [0.2, 0.4])

    def test_missing_tissue_uses_dataset_name(self):
        obs_no_tissue = pd.DataFrame(
            {"cell_type": ["A", "B"], "batch": ["x", "y"]},
            index=["c0", "c1"],
        )
        lib = np.array([10.0, 20.0])
        result = build_standardized_obs(
            obs_no_tissue, np.array([0, 1]), "cell_type", "batch", "fallback_ds", lib
        )
        assert (result["tissue"] == "fallback_ds").all()


# ---------------------------------------------------------------------------
# TestBuildStandardizedH5ads
# ---------------------------------------------------------------------------

class TestBuildStandardizedH5ads:
    """Tests for the main build_standardized_h5ads function."""

    def _run_build(self, tmp_h5ad_dir, tmp_path, streaming: bool, chunk_size: int):
        """Shared helper to build standardized h5ads."""
        h5ad_path = tmp_h5ad_dir / "sparse_rawX.h5ad"
        adata = ad.read_h5ad(h5ad_path)
        obs = adata.obs.copy()
        all_var = adata.raw.var.copy()
        n_obs = adata.n_obs
        n_vars = adata.raw.n_vars

        # 160/20/20 split
        rng = np.random.default_rng(99)
        split_col = np.array(["train"] * 160 + ["val"] * 20 + ["test"] * 20)
        rng.shuffle(split_col)

        hvg_mask = np.zeros(n_vars, dtype=bool)
        hvg_mask[:30] = True  # first 30 genes are HVGs

        out_dir = tmp_path / ("streaming" if streaming else "accumulate")

        stats = build_standardized_h5ads(
            h5ad_path=h5ad_path,
            output_dir=out_dir,
            split_col=split_col,
            hvg_mask=hvg_mask,
            all_var=all_var,
            obs=obs,
            cell_type_col="cell_type",
            batch_key="batch",
            dataset_name="test_ds",
            matrix_source="raw.X",
            chunk_size=chunk_size,
            target_sum=1e4,
            extra_obs_columns={"extra_col": "extra_col"},
            streaming=streaming,
        )

        return stats, out_dir, split_col, n_vars, hvg_mask

    def test_creates_split_files(self, tmp_h5ad_dir, tmp_path):
        stats, out_dir, split_col, n_vars, hvg_mask = self._run_build(
            tmp_h5ad_dir, tmp_path, streaming=False, chunk_size=64
        )

        for split_name in ["train", "val", "test"]:
            assert (out_dir / f"{split_name}.h5ad").exists(), f"{split_name}.h5ad missing"
            expected_n = int((split_col == split_name).sum())

            s = stats[split_name]
            assert s["n_cells"] == expected_n
            assert s["nnz_counts"] > 0
            assert s["nnz_normalized"] > 0
            assert s["library_size_mean"] > 0
            assert s["file_size_mb"] > 0

            # Read back and verify contents
            a = ad.read_h5ad(out_dir / f"{split_name}.h5ad")
            assert a.shape[0] == expected_n
            assert a.shape[1] == n_vars
            assert "normalized" in a.layers
            assert "cell_type" in a.obs.columns
            assert "batch" in a.obs.columns
            assert "tissue" in a.obs.columns
            assert "dataset" in a.obs.columns
            assert "library_size" in a.obs.columns
            assert "extra_col" in a.obs.columns
            assert "highly_variable" in a.var.columns
            assert a.var["highly_variable"].sum() == int(hvg_mask.sum())

            # Check obsm carry-over
            assert "X_umap" in a.obsm
            assert a.obsm["X_umap"].shape == (expected_n, 2)

    def test_streaming_mode(self, tmp_h5ad_dir, tmp_path):
        stats, out_dir, split_col, n_vars, hvg_mask = self._run_build(
            tmp_h5ad_dir, tmp_path, streaming=True, chunk_size=50
        )

        for split_name in ["train", "val", "test"]:
            assert (out_dir / f"{split_name}.h5ad").exists()
            expected_n = int((split_col == split_name).sum())

            s = stats[split_name]
            assert s["n_cells"] == expected_n
            assert s["nnz_counts"] > 0

            # Read back — streaming files should be readable by anndata
            a = ad.read_h5ad(out_dir / f"{split_name}.h5ad")
            assert a.shape[0] == expected_n
            assert a.shape[1] == n_vars
            assert "normalized" in a.layers
            assert "cell_type" in a.obs.columns
            assert "library_size" in a.obs.columns
            assert "highly_variable" in a.var.columns

            # Check obsm
            assert "X_umap" in a.obsm
            assert a.obsm["X_umap"].shape == (expected_n, 2)
