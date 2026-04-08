"""Build standardized per-split h5ad files from a source h5ad.

Reads a source h5ad, splits cells by a pre-computed split assignment,
normalizes, and writes per-split h5ad files.  Two strategies are provided:

* **accumulate** (default) — loads chunks into memory, vstacks, then writes
  via anndata.
* **streaming** — writes CSR components directly to h5py, one pass per
  split, for datasets too large to fit in memory.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sp_vstack

from .h5ad_io import locate_matrix, get_matrix_shape, read_sparse_chunk


# ---------------------------------------------------------------------------
# Var builder
# ---------------------------------------------------------------------------

def _build_var(all_var: pd.DataFrame, hvg_mask: np.ndarray) -> pd.DataFrame:
    """Build the output var DataFrame.

    Copies *all_var*, adds a ``highly_variable`` boolean column.
    If a ``feature_name`` column exists and >50 % of values are valid
    strings, those are used as the index.
    """
    var = all_var.copy()
    var["highly_variable"] = hvg_mask.astype(bool)

    if "feature_name" in var.columns:
        names = var["feature_name"]
        valid = names.notna() & (names.astype(str).str.strip() != "")
        if valid.mean() > 0.5:
            var.index = names.astype(str)

    return var


# ---------------------------------------------------------------------------
# Obs builder (public helper)
# ---------------------------------------------------------------------------

def build_standardized_obs(
    obs: pd.DataFrame,
    cell_indices: np.ndarray,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    library_size: np.ndarray,
    extra_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Produce a standardized obs DataFrame for one split.

    Columns: cell_type, batch, tissue, dataset, library_size (+ extras).
    If the source *obs* has no ``tissue`` column the value is filled with
    *dataset_name*.
    """
    sub = obs.iloc[cell_indices]
    out: dict[str, object] = {
        "cell_type": sub[cell_type_col].values,
        "batch": sub[batch_key].values,
        "tissue": sub["tissue"].values if "tissue" in sub.columns else np.full(len(sub), dataset_name, dtype=object),
        "dataset": np.full(len(sub), dataset_name, dtype=object),
        "library_size": library_size,
    }

    if extra_columns:
        for dst, src in extra_columns.items():
            out[dst] = sub[src].values

    return pd.DataFrame(out, index=sub.index)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_csr(X: csr_matrix, target_sum: float) -> csr_matrix:
    """Return ``log1p(target_sum * X / row_sums)`` as float32 CSR."""
    X = X.astype(np.float32).copy()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    # Multiply each row in-place
    indptr = X.indptr
    for i in range(X.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if s < e:
            X.data[s:e] = np.log1p(target_sum * X.data[s:e] / row_sums[i])
    return X


# ---------------------------------------------------------------------------
# h5py helpers for streaming mode
# ---------------------------------------------------------------------------

def _write_csr_to_h5(grp: h5py.Group, csr_mat: csr_matrix) -> None:
    """Write a CSR matrix into an h5py group with anndata conventions."""
    csr_mat = csr_mat.astype(np.float32)
    csr_mat.sort_indices()
    grp.create_dataset("data", data=csr_mat.data)
    grp.create_dataset("indices", data=csr_mat.indices.astype(np.int32))
    grp.create_dataset("indptr", data=csr_mat.indptr.astype(np.int64))
    grp.attrs["encoding-type"] = "csr_matrix"
    grp.attrs["encoding-version"] = "0.1.0"
    grp.attrs["shape"] = list(csr_mat.shape)


def _write_obs_to_h5(grp: h5py.Group, std_obs: pd.DataFrame) -> None:
    """Write a DataFrame to an h5py group, encoding categoricals."""
    grp.attrs["_index"] = "_index"
    grp.attrs["encoding-type"] = "dataframe"
    grp.attrs["encoding-version"] = "0.2.0"
    columns = list(std_obs.columns)
    grp.attrs["column-order"] = columns

    # Index
    idx_vals = np.array(std_obs.index.astype(str), dtype="S")
    grp.create_dataset("_index", data=idx_vals)

    for col in columns:
        series = std_obs[col]
        if hasattr(series, "cat") or isinstance(series.dtype, pd.CategoricalDtype):
            cat = series.astype("category")
            g = grp.create_group(col)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            g.create_dataset("codes", data=cat.cat.codes.values.astype(np.int32))
            cats = np.array(cat.cat.categories.astype(str), dtype="S")
            g.create_dataset("categories", data=cats)
        elif np.issubdtype(series.dtype, np.floating) or np.issubdtype(series.dtype, np.integer):
            grp.create_dataset(col, data=series.values)
        else:
            vals = np.array(series.astype(str), dtype="S")
            ds = grp.create_dataset(col, data=vals)
            ds.attrs["encoding-type"] = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"


def _write_var_to_h5(grp: h5py.Group, var: pd.DataFrame) -> None:
    """Write var DataFrame to h5py group."""
    _write_obs_to_h5(grp, var)  # same encoding logic


# ---------------------------------------------------------------------------
# Accumulate mode
# ---------------------------------------------------------------------------

def _write_accumulate(
    h5ad_path: str | Path,
    output_dir: Path,
    split_col: np.ndarray,
    hvg_mask: np.ndarray,
    all_var: pd.DataFrame,
    obs: pd.DataFrame,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    matrix_source: str,
    chunk_size: int,
    target_sum: float,
    extra_obs_columns: dict[str, str] | None,
) -> dict:
    splits = ["train", "val", "test"]
    accumulators: dict[str, list[csr_matrix]] = {s: [] for s in splits}
    lib_sizes: dict[str, list[np.ndarray]] = {s: [] for s in splits}
    cell_idx: dict[str, list[np.ndarray]] = {s: [] for s in splits}

    # Read obsm arrays from source for carry-over
    obsm_keys: list[str] = []
    obsm_data: dict[str, np.ndarray] = {}
    with h5py.File(str(h5ad_path), "r") as f:
        if "obsm" in f:
            obsm_keys = list(f["obsm"].keys())
            for k in obsm_keys:
                obsm_data[k] = f["obsm"][k][:]

    # Single pass through source
    with h5py.File(str(h5ad_path), "r") as f:
        mat_obj, _, _ = locate_matrix(f, matrix_source)
        n_obs, n_vars = get_matrix_shape(mat_obj)

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            chunk = read_sparse_chunk(mat_obj, start, end, n_vars)
            chunk = chunk.astype(np.float32)
            row_sums = np.asarray(chunk.sum(axis=1)).ravel()

            for s in splits:
                mask = split_col[start:end] == s
                if not mask.any():
                    continue
                accumulators[s].append(chunk[mask])
                lib_sizes[s].append(row_sums[mask])
                cell_idx[s].append(np.arange(start, end)[mask])

    stats: dict = {}
    var_out = _build_var(all_var, hvg_mask)

    for s in splits:
        if not accumulators[s]:
            stats[s] = {"n_cells": 0, "nnz_counts": 0, "nnz_normalized": 0,
                        "library_size_mean": 0.0, "library_size_median": 0.0, "file_size_mb": 0.0}
            continue

        X_raw = sp_vstack(accumulators[s], format="csr").astype(np.float32)
        ls = np.concatenate(lib_sizes[s])
        indices = np.concatenate(cell_idx[s])

        del accumulators[s]
        gc.collect()

        X_norm = _normalize_csr(X_raw.copy(), target_sum)
        std_obs = build_standardized_obs(obs, indices, cell_type_col, batch_key, dataset_name, ls, extra_obs_columns)

        adata = ad.AnnData(
            X=X_raw,
            obs=std_obs,
            var=var_out.copy(),
            layers={"normalized": X_norm},
        )

        # Carry over obsm
        for k in obsm_keys:
            adata.obsm[k] = obsm_data[k][indices]

        out_path = output_dir / f"{s}.h5ad"
        adata.write_h5ad(out_path, compression=None)

        file_size = os.path.getsize(out_path) / (1024 * 1024)
        stats[s] = {
            "n_cells": int(X_raw.shape[0]),
            "nnz_counts": int(X_raw.nnz),
            "nnz_normalized": int(X_norm.nnz),
            "library_size_mean": float(np.mean(ls)),
            "library_size_median": float(np.median(ls)),
            "file_size_mb": float(file_size),
        }

        del X_raw, X_norm, adata
        gc.collect()

    return stats


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------

def _write_streaming(
    h5ad_path: str | Path,
    output_dir: Path,
    split_col: np.ndarray,
    hvg_mask: np.ndarray,
    all_var: pd.DataFrame,
    obs: pd.DataFrame,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    matrix_source: str,
    chunk_size: int,
    target_sum: float,
    extra_obs_columns: dict[str, str] | None,
) -> dict:
    splits = ["train", "val", "test"]
    var_out = _build_var(all_var, hvg_mask)

    # Read obsm from source
    obsm_keys: list[str] = []
    obsm_data: dict[str, np.ndarray] = {}
    with h5py.File(str(h5ad_path), "r") as f:
        if "obsm" in f:
            obsm_keys = list(f["obsm"].keys())
            for k in obsm_keys:
                obsm_data[k] = f["obsm"][k][:]

    stats: dict = {}

    for s in splits:
        out_path = output_dir / f"{s}.h5ad"
        split_mask_global = split_col == s
        n_split = int(split_mask_global.sum())

        if n_split == 0:
            stats[s] = {"n_cells": 0, "nnz_counts": 0, "nnz_normalized": 0,
                        "library_size_mean": 0.0, "library_size_median": 0.0, "file_size_mb": 0.0}
            continue

        # Collect raw CSR chunks, library sizes, and cell indices
        raw_chunks: list[csr_matrix] = []
        norm_chunks: list[csr_matrix] = []
        lib_sizes: list[np.ndarray] = []
        indices_list: list[np.ndarray] = []

        with h5py.File(str(h5ad_path), "r") as f:
            mat_obj, _, _ = locate_matrix(f, matrix_source)
            n_obs, n_vars = get_matrix_shape(mat_obj)

            for start in range(0, n_obs, chunk_size):
                end = min(start + chunk_size, n_obs)
                local_mask = split_mask_global[start:end]
                if not local_mask.any():
                    continue

                chunk = read_sparse_chunk(mat_obj, start, end, n_vars)
                chunk = chunk[local_mask].astype(np.float32)
                row_sums = np.asarray(chunk.sum(axis=1)).ravel()

                raw_chunks.append(chunk)
                norm_chunks.append(_normalize_csr(chunk.copy(), target_sum))
                lib_sizes.append(row_sums)
                indices_list.append(np.arange(start, end)[local_mask])

        # Stack
        X_raw = sp_vstack(raw_chunks, format="csr").astype(np.float32)
        X_norm = sp_vstack(norm_chunks, format="csr").astype(np.float32)
        ls = np.concatenate(lib_sizes)
        all_indices = np.concatenate(indices_list)

        del raw_chunks, norm_chunks
        gc.collect()

        std_obs = build_standardized_obs(obs, all_indices, cell_type_col, batch_key, dataset_name, ls, extra_obs_columns)

        # Write directly via h5py
        with h5py.File(str(out_path), "w") as hf:
            # X (raw counts)
            x_grp = hf.create_group("X")
            _write_csr_to_h5(x_grp, X_raw)

            # layers/normalized
            layers_grp = hf.create_group("layers")
            norm_grp = layers_grp.create_group("normalized")
            _write_csr_to_h5(norm_grp, X_norm)

            # obs
            obs_grp = hf.create_group("obs")
            _write_obs_to_h5(obs_grp, std_obs)

            # var
            var_grp = hf.create_group("var")
            _write_var_to_h5(var_grp, var_out)

            # obsm
            if obsm_keys:
                obsm_grp = hf.create_group("obsm")
                for k in obsm_keys:
                    obsm_grp.create_dataset(k, data=obsm_data[k][all_indices])

        file_size = os.path.getsize(out_path) / (1024 * 1024)
        stats[s] = {
            "n_cells": int(X_raw.shape[0]),
            "nnz_counts": int(X_raw.nnz),
            "nnz_normalized": int(X_norm.nnz),
            "library_size_mean": float(np.mean(ls)),
            "library_size_median": float(np.median(ls)),
            "file_size_mb": float(file_size),
        }

        del X_raw, X_norm
        gc.collect()

    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_standardized_h5ads(
    h5ad_path: str | Path,
    output_dir: str | Path,
    split_col: np.ndarray,
    hvg_mask: np.ndarray,
    all_var: pd.DataFrame,
    obs: pd.DataFrame,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    matrix_source: str = "raw.X",
    chunk_size: int = 50_000,
    target_sum: float = 1e4,
    extra_obs_columns: dict[str, str] | None = None,
    streaming: bool = False,
) -> dict:
    """Build standardized per-split h5ad files from a source h5ad.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the source h5ad file.
    output_dir : str or Path
        Directory to write per-split h5ad files.
    split_col : numpy.ndarray
        Array of ``"train"``/``"val"``/``"test"`` per cell.
    hvg_mask : numpy.ndarray
        Boolean mask over all genes.
    all_var : pandas.DataFrame
        Full var DataFrame.
    obs : pandas.DataFrame
        Full obs DataFrame.
    cell_type_col : str
        Column in *obs* holding cell-type labels.
    batch_key : str
        Column in *obs* holding batch labels.
    dataset_name : str
        Name of the dataset (used for the ``dataset`` column in obs).
    matrix_source : str
        Source of the count matrix (``"raw.X"``, ``"X"``, etc.).
    chunk_size : int
        Number of rows to read per chunk.
    target_sum : float
        Target total count for normalization.
    extra_obs_columns : dict or None
        Mapping of ``{output_col: source_col}`` for extra obs columns.
    streaming : bool
        If *True*, use the streaming write strategy.

    Returns
    -------
    dict
        Per-split statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        h5ad_path=h5ad_path,
        output_dir=output_dir,
        split_col=split_col,
        hvg_mask=hvg_mask,
        all_var=all_var,
        obs=obs,
        cell_type_col=cell_type_col,
        batch_key=batch_key,
        dataset_name=dataset_name,
        matrix_source=matrix_source,
        chunk_size=chunk_size,
        target_sum=target_sum,
        extra_obs_columns=extra_obs_columns,
    )

    if streaming:
        return _write_streaming(**common)
    else:
        return _write_accumulate(**common)
