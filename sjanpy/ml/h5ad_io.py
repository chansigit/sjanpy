"""Unified h5py-based readers for h5ad files.

Provides low-level, memory-efficient access to obs, var, and expression
matrices stored in h5ad files *without* loading the entire object into
memory.  All public functions accept file paths (or open h5py handles)
so callers never need to touch anndata or scanpy.
"""

from __future__ import annotations

import warnings
from typing import Literal

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_stringlike(values):
    """Decode h5py byte/object arrays to Python str arrays.

    Works for scalars, 0-d arrays, and n-d arrays.  Returns the input
    unchanged when no decoding is needed.
    """
    arr = np.asarray(values)
    if not hasattr(arr, "dtype") or arr.dtype.kind not in ("S", "O", "U"):
        return values

    def _one(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        return str(value)

    if arr.ndim == 0:
        return _one(arr.item())

    return np.array([_one(v) for v in arr.flat], dtype=object).reshape(arr.shape)


def _read_h5_group_to_dataframe(grp: h5py.Group) -> pd.DataFrame:
    """Read an h5ad obs or var group into a :class:`~pandas.DataFrame`.

    Handles:
    * Legacy ``__categories`` format (older anndata versions)
    * Modern categorical / string-array encoding-type attributes
    * Scalar categories (single element stored as scalar)
    * Byte-string decoding
    """
    col_dict: dict[str, object] = {}

    # --- legacy format --------------------------------------------------
    if "__categories" in grp:
        index_key = str(grp.attrs.get("_index", "index"))
        raw_index = grp[index_key][:]
        if raw_index.dtype.kind in ("S", "O"):
            raw_index = _decode_stringlike(raw_index)

        for key in grp.keys():
            if key in ("__categories", index_key):
                continue
            col_data = grp[key][:]
            if col_data.dtype.kind in ("S", "O"):
                col_data = _decode_stringlike(col_data)
            col_dict[key] = col_data

        df = pd.DataFrame(col_dict)
        df.index = raw_index
        return df

    # --- modern format --------------------------------------------------
    index_key = str(grp.attrs.get("_index", "_index"))
    raw_index = None
    if index_key in grp:
        idx_ds = grp[index_key]
        raw_index = idx_ds[:]
        if hasattr(raw_index, "dtype") and raw_index.dtype.kind in ("S", "O"):
            raw_index = _decode_stringlike(raw_index)

    for key in grp.keys():
        if key == index_key:
            continue
        dataset = grp[key]
        enc_type = dataset.attrs.get("encoding-type", "")

        if enc_type == "categorical":
            codes = dataset["codes"][:]
            categories = dataset["categories"][:]
            if categories.dtype.kind in ("S", "O"):
                categories = _decode_stringlike(categories)
            if categories.ndim == 0:
                categories = np.array([str(categories)])
            col_dict[key] = pd.Categorical.from_codes(codes, categories=categories)
        elif enc_type == "string-array":
            col_data = dataset[:]
            if col_data.dtype.kind in ("S", "O"):
                col_data = _decode_stringlike(col_data)
            col_dict[key] = col_data
        else:
            col_data = dataset[:]
            if col_data.dtype.kind in ("S", "O"):
                col_data = _decode_stringlike(col_data)
            col_dict[key] = col_data

    df = pd.DataFrame(col_dict)
    if raw_index is not None:
        df.index = raw_index
    return df


# ---------------------------------------------------------------------------
# Public: obs / var readers
# ---------------------------------------------------------------------------

def read_obs(h5ad_path: str | object) -> pd.DataFrame:
    """Read the ``obs`` DataFrame from an h5ad file via h5py.

    Parameters
    ----------
    h5ad_path : str or path-like
        Path to the h5ad file.

    Returns
    -------
    pandas.DataFrame
    """
    with h5py.File(str(h5ad_path), "r") as f:
        return _read_h5_group_to_dataframe(f["obs"])


def read_var(h5ad_path: str | object, group: str = "var") -> pd.DataFrame:
    """Read a ``var`` DataFrame from an h5ad file via h5py.

    Parameters
    ----------
    h5ad_path : str or path-like
        Path to the h5ad file.
    group : str, default ``"var"``
        HDF5 group path.  Use ``"raw/var"`` to read the var from the raw
        layer.

    Returns
    -------
    pandas.DataFrame
    """
    with h5py.File(str(h5ad_path), "r") as f:
        return _read_h5_group_to_dataframe(f[group])


# ---------------------------------------------------------------------------
# Public: matrix helpers
# ---------------------------------------------------------------------------

def locate_matrix(
    f: h5py.File,
    source: str,
) -> tuple:
    """Resolve a matrix source string inside an open h5ad file.

    Parameters
    ----------
    f : h5py.File
        An open h5py File handle.
    source : str
        One of ``"raw.X"``, ``"X"``, or ``"layers/<name>"``.

    Returns
    -------
    (matrix_obj, var_group, label) : tuple
        *matrix_obj* is either an h5py.Group (sparse) or h5py.Dataset
        (dense).  *var_group* is the corresponding ``var`` or ``raw/var``
        group.  *label* is a human-readable string.
    """
    if source == "raw.X":
        if "raw" not in f or "X" not in f["raw"]:
            raise ValueError("raw.X not found in file")
        mat = f["raw/X"]
        var_grp = f["raw/var"] if "raw/var" in f else f["var"]
        return mat, var_grp, "raw.X"

    if source == "X":
        if "X" not in f:
            raise ValueError("X not found in file")
        return f["X"], f["var"], "X"

    if source.startswith("layers/"):
        layer_name = source[len("layers/"):]
        if "layers" not in f or layer_name not in f["layers"]:
            raise ValueError(f"layers/{layer_name} not found in file")
        return f["layers"][layer_name], f["var"], f"layers/{layer_name}"

    raise ValueError(
        f"Unknown matrix source '{source}'. "
        "Expected 'raw.X', 'X', or 'layers/<name>'."
    )


def get_matrix_shape(matrix_obj) -> tuple[int, int]:
    """Return ``(n_obs, n_vars)`` for a sparse group or dense dataset.

    Parameters
    ----------
    matrix_obj : h5py.Group or h5py.Dataset
        The matrix object returned by :func:`locate_matrix`.
    """
    if isinstance(matrix_obj, h5py.Group):
        # CSR / CSC sparse — shape stored as attribute
        shape = matrix_obj.attrs.get("shape")
        if shape is not None:
            return tuple(int(s) for s in shape)
        # Fallback: infer from indptr length
        n_obs = len(matrix_obj["indptr"]) - 1
        n_vars = int(matrix_obj["indices"][:].max()) + 1
        return n_obs, n_vars
    else:
        return matrix_obj.shape


def read_matrix_rows(
    matrix_obj,
    row_indices: np.ndarray,
) -> csr_matrix:
    """Read arbitrary rows from a sparse group or dense dataset.

    Parameters
    ----------
    matrix_obj : h5py.Group or h5py.Dataset
        Sparse group or dense dataset.
    row_indices : numpy.ndarray
        1-D integer array of row indices to read.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(len(row_indices), n_vars)``.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)
    n_rows = len(row_indices)

    if isinstance(matrix_obj, h5py.Group):
        # Sparse (CSR) group
        n_obs, n_vars = get_matrix_shape(matrix_obj)
        indptr_full = matrix_obj["indptr"][:]
        all_data = matrix_obj["data"]
        all_indices = matrix_obj["indices"]

        new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
        data_parts = []
        idx_parts = []

        for i, row in enumerate(row_indices):
            start, end = int(indptr_full[row]), int(indptr_full[row + 1])
            row_len = end - start
            new_indptr[i + 1] = new_indptr[i] + row_len
            if row_len > 0:
                data_parts.append(all_data[start:end])
                idx_parts.append(all_indices[start:end])

        if data_parts:
            data = np.concatenate(data_parts)
            indices = np.concatenate(idx_parts)
        else:
            data = np.array([], dtype=np.float32)
            indices = np.array([], dtype=np.int64)

        return csr_matrix((data, indices, new_indptr), shape=(n_rows, n_vars))

    else:
        # Dense dataset – fancy-index then convert
        n_vars = matrix_obj.shape[1]
        # h5py fancy indexing requires sorted indices
        sort_order = np.argsort(row_indices)
        sorted_rows = row_indices[sort_order]
        dense_block = matrix_obj[sorted_rows, :]
        # Unsort to match caller's order
        unsort = np.argsort(sort_order)
        dense_block = dense_block[unsort]
        return csr_matrix(dense_block)


def read_sparse_chunk(
    matrix_obj,
    start: int,
    end: int,
    n_vars: int,
) -> csr_matrix:
    """Read a contiguous row range ``[start, end)`` as CSR.

    Optimised for sequential chunk reading.

    Parameters
    ----------
    matrix_obj : h5py.Group or h5py.Dataset
        Sparse group or dense dataset.
    start, end : int
        Row range (exclusive end).
    n_vars : int
        Number of columns (genes).

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    n_rows = end - start

    if isinstance(matrix_obj, h5py.Group):
        indptr_slice = matrix_obj["indptr"][start : end + 1]
        data_start = int(indptr_slice[0])
        data_end = int(indptr_slice[-1])
        data = matrix_obj["data"][data_start:data_end]
        indices = matrix_obj["indices"][data_start:data_end]
        indptr = (indptr_slice - indptr_slice[0]).astype(np.int64)
        return csr_matrix((data, indices, indptr), shape=(n_rows, n_vars))
    else:
        dense_block = matrix_obj[start:end, :]
        return csr_matrix(dense_block)


def validate_matrix_values(
    matrix_obj,
    expected_type: Literal["counts", "normalized"],
    sample_n: int = 200_000,
    strict: bool = False,
) -> bool:
    """Check that sampled matrix values match the declared type.

    Parameters
    ----------
    matrix_obj : h5py.Group or h5py.Dataset
        Sparse group or dense dataset.
    expected_type : ``"counts"`` or ``"normalized"``
        What the values should look like.
    sample_n : int
        Number of non-zero values to sample.
    strict : bool
        If *True*, raise :class:`ValueError` on mismatch instead of
        returning *False*.

    Returns
    -------
    bool
        *True* when the sample matches the expected type.
    """
    # Collect a sample of values
    if isinstance(matrix_obj, h5py.Group):
        data_ds = matrix_obj["data"]
        total = data_ds.shape[0]
        if total == 0:
            return True
        n = min(sample_n, total)
        if n == total:
            sample = data_ds[:]
        else:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(total, size=n, replace=False))
            sample = data_ds[idx]
    else:
        flat = matrix_obj[:].ravel()
        nonzero = flat[flat != 0]
        if len(nonzero) == 0:
            return True
        n = min(sample_n, len(nonzero))
        if n == len(nonzero):
            sample = nonzero
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(nonzero), size=n, replace=False)
            sample = nonzero[idx]

    sample = sample.astype(np.float64)

    if expected_type == "counts":
        # Counts should be non-negative integers
        is_int = np.allclose(sample, np.round(sample))
        is_nonneg = (sample >= 0).all()
        ok = is_int and is_nonneg
    elif expected_type == "normalized":
        # Normalized data typically has non-integer values
        is_int = np.allclose(sample, np.round(sample))
        ok = not is_int  # at least some fractional values expected
    else:
        raise ValueError(f"expected_type must be 'counts' or 'normalized', got '{expected_type}'")

    if not ok and strict:
        raise ValueError(
            f"Matrix values do not match expected type '{expected_type}'. "
            f"Sample stats: min={sample.min():.4f}, max={sample.max():.4f}, "
            f"frac_integer={np.isclose(sample, np.round(sample)).mean():.3f}"
        )
    return ok
