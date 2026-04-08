"""Tests for sjanpy.ml.h5ad_io – h5py-based h5ad readers."""

import h5py
import numpy as np
import pytest
from scipy.sparse import issparse

from sjanpy.ml.h5ad_io import (
    locate_matrix,
    get_matrix_shape,
    read_matrix_rows,
    read_obs,
    read_sparse_chunk,
    read_var,
    validate_matrix_values,
)


# ──────────────────────────────────────────────────────────────────────
# TestReadObs
# ──────────────────────────────────────────────────────────────────────

class TestReadObs:
    def test_reads_all_columns_sparse(self, tmp_h5ad_dir):
        """sparse_rawX.h5ad has 200 cells with cell_type, batch, tissue, extra_col."""
        df = read_obs(tmp_h5ad_dir / "sparse_rawX.h5ad")
        assert len(df) == 200
        assert set(df.columns) >= {"cell_type", "batch", "tissue", "extra_col"}
        assert set(df["cell_type"].unique()) == {"T-cell", "B-cell", "Monocyte", "NK"}

    def test_reads_dense_file(self, tmp_h5ad_dir):
        """dense_X.h5ad has 100 cells."""
        df = read_obs(tmp_h5ad_dir / "dense_X.h5ad")
        assert len(df) == 100
        assert "cell_type" in df.columns
        assert "batch" in df.columns


# ──────────────────────────────────────────────────────────────────────
# TestReadVar
# ──────────────────────────────────────────────────────────────────────

class TestReadVar:
    def test_reads_gene_names(self, tmp_h5ad_dir):
        """sparse_rawX.h5ad has 50 genes."""
        df = read_var(tmp_h5ad_dir / "sparse_rawX.h5ad")
        assert len(df) == 50
        assert df.index[0] == "gene_0"

    def test_reads_raw_var_group(self, tmp_h5ad_dir):
        """sparse_rawX.h5ad stores raw/var with the same 50 genes."""
        df = read_var(tmp_h5ad_dir / "sparse_rawX.h5ad", group="raw/var")
        assert len(df) == 50


# ──────────────────────────────────────────────────────────────────────
# TestLocateMatrix
# ──────────────────────────────────────────────────────────────────────

class TestLocateMatrix:
    def test_locate_raw_X(self, tmp_h5ad_dir):
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, var_grp, label = locate_matrix(f, "raw.X")
            assert label == "raw.X"
            assert "indptr" in mat or hasattr(mat, "shape")

    def test_locate_X(self, tmp_h5ad_dir):
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            mat, var_grp, label = locate_matrix(f, "X")
            assert label == "X"

    def test_invalid_source_raises(self, tmp_h5ad_dir):
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            with pytest.raises(ValueError, match="Unknown matrix source"):
                locate_matrix(f, "bad_source")


# ──────────────────────────────────────────────────────────────────────
# TestReadMatrixRows
# ──────────────────────────────────────────────────────────────────────

class TestReadMatrixRows:
    def test_read_sparse_rows(self, tmp_h5ad_dir):
        """Read specific rows from the sparse raw.X in sparse_rawX.h5ad."""
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            rows = np.array([0, 5, 10, 199])
            result = read_matrix_rows(mat, rows)
            assert issparse(result)
            assert result.shape == (4, 50)

    def test_read_dense_rows(self, tmp_h5ad_dir):
        """Read specific rows from the dense X in dense_X.h5ad."""
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            rows = np.array([0, 50, 99])
            result = read_matrix_rows(mat, rows)
            assert issparse(result)
            assert result.shape == (3, 30)

    def test_empty_indices(self, tmp_h5ad_dir):
        """Empty row_indices should return (0, n_vars) matrix."""
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            result = read_matrix_rows(mat, np.array([], dtype=np.int64))
            assert result.shape == (0, 50)


# ──────────────────────────────────────────────────────────────────────
# TestReadSparseChunk
# ──────────────────────────────────────────────────────────────────────

class TestReadSparseChunk:
    def test_contiguous_chunk(self, tmp_h5ad_dir):
        """Read rows 0..5 from tiny.h5ad."""
        with h5py.File(tmp_h5ad_dir / "tiny.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            shape = get_matrix_shape(mat)
            chunk = read_sparse_chunk(mat, 0, 5, shape[1])
            assert issparse(chunk)
            assert chunk.shape == (5, 5)


# ──────────────────────────────────────────────────────────────────────
# TestValidateMatrixValues
# ──────────────────────────────────────────────────────────────────────

class TestValidateMatrixValues:
    def test_counts_pass(self, tmp_h5ad_dir):
        """Poisson counts should pass the 'counts' check."""
        with h5py.File(tmp_h5ad_dir / "tiny.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            assert validate_matrix_values(mat, "counts")

    def test_counts_pass_dense(self, tmp_h5ad_dir):
        """Dense integer matrix should pass counts validation."""
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            assert validate_matrix_values(mat, "counts", strict=True)

    def test_wrong_type_strict_raises(self, tmp_h5ad_dir):
        """Poisson counts flagged as 'normalized' should raise in strict mode."""
        with h5py.File(tmp_h5ad_dir / "tiny.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            with pytest.raises(ValueError, match="do not match expected type"):
                validate_matrix_values(mat, "normalized", strict=True)
