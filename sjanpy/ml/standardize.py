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

def _build_var(all_var: pd.DataFrame, hvg_mask: np.ndarray, use_feature_name_as_index: bool = True) -> pd.DataFrame:
    """Build the output var DataFrame.

    Copies *all_var*, adds a ``highly_variable`` boolean column.
    If *use_feature_name_as_index* is True and a ``feature_name`` column
    exists with >50 % valid strings, those are used as the index.
    """
    var = all_var.copy()
    var["highly_variable"] = hvg_mask.astype(bool)

    if use_feature_name_as_index and "feature_name" in var.columns:
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
        for src, dst in extra_columns.items():
            if src in sub.columns:
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
    # Vectorized: repeat each row_sum for the number of nonzeros in that row
    row_counts = np.diff(X.indptr)
    divisors = np.repeat(row_sums, row_counts)
    X.data = np.log1p(target_sum * X.data / divisors).astype(np.float32)
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
    idx_ds = grp.create_dataset("_index", data=idx_vals)
    idx_ds.attrs["encoding-type"] = "string-array"
    idx_ds.attrs["encoding-version"] = "0.2.0"

    for col in columns:
        series = std_obs[col]
        if hasattr(series, "cat") or isinstance(series.dtype, pd.CategoricalDtype):
            cat = series.astype("category")
            g = grp.create_group(col)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            codes_ds = g.create_dataset("codes", data=cat.cat.codes.values.astype(np.int32))
            codes_ds.attrs["encoding-type"] = "array"
            codes_ds.attrs["encoding-version"] = "0.2.0"
            cats = np.array(cat.cat.categories.astype(str), dtype="S")
            cats_ds = g.create_dataset("categories", data=cats)
            cats_ds.attrs["encoding-type"] = "string-array"
            cats_ds.attrs["encoding-version"] = "0.2.0"
        elif series.dtype == np.bool_ or np.issubdtype(series.dtype, np.floating) or np.issubdtype(series.dtype, np.integer):
            ds = grp.create_dataset(col, data=series.values)
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"
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
    obsm_data: dict[str, np.ndarray] = {}
    with h5py.File(str(h5ad_path), "r") as f:
        if "obsm" in f:
            for k in f["obsm"]:
                obsm_data[k] = f["obsm"][k][:]

    stats: dict = {}

    for s in splits:
        split_mask_full = split_col == s
        n_cells = int(split_mask_full.sum())

        if n_cells == 0:
            stats[s] = {"n_cells": 0, "nnz_counts": 0, "nnz_normalized": 0,
                        "library_size_mean": 0.0, "library_size_median": 0.0, "file_size_mb": 0.0}
            continue

        out_path = output_dir / f"{s}.h5ad"
        obs_indices = np.where(split_mask_full)[0]

        with h5py.File(str(h5ad_path), "r") as src_f:
            mat_obj, _, _ = locate_matrix(src_f, matrix_source)
            n_obs, n_vars = get_matrix_shape(mat_obj)

            with h5py.File(str(out_path), "w") as out_f:
                # Create resizable datasets for raw X
                x_grp = out_f.create_group("X")
                x_data = x_grp.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32)
                x_indices = x_grp.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32)
                x_indptr: list[np.int64] = [np.int64(0)]

                # Create resizable datasets for normalized layer
                layers_grp = out_f.create_group("layers")
                n_grp = layers_grp.create_group("normalized")
                n_data = n_grp.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32)
                n_indices = n_grp.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32)
                n_indptr: list[np.int64] = [np.int64(0)]

                lib_sizes: list[np.ndarray] = []
                total_nnz_x = 0
                total_nnz_n = 0

                for start in range(0, n_obs, chunk_size):
                    end = min(start + chunk_size, n_obs)
                    local_mask = split_mask_full[start:end]
                    if not local_mask.any():
                        continue

                    chunk = read_sparse_chunk(mat_obj, start, end, n_vars)
                    chunk = chunk.astype(np.float32)
                    row_sums = np.asarray(chunk.sum(axis=1)).ravel()

                    split_chunk = chunk[local_mask]
                    split_lib = row_sums[local_mask]
                    del chunk, row_sums

                    if not isinstance(split_chunk, csr_matrix):
                        split_chunk = split_chunk.tocsr()

                    # Normalize this chunk
                    norm_chunk = _normalize_csr(split_chunk.copy(), target_sum)

                    # Append raw CSR components
                    nnz_x = split_chunk.nnz
                    if nnz_x > 0:
                        x_data.resize(total_nnz_x + nnz_x, axis=0)
                        x_data[total_nnz_x:] = split_chunk.data
                        x_indices.resize(total_nnz_x + nnz_x, axis=0)
                        x_indices[total_nnz_x:] = split_chunk.indices.astype(np.int32)
                    for row_nnz in np.diff(split_chunk.indptr):
                        x_indptr.append(x_indptr[-1] + row_nnz)
                    total_nnz_x += nnz_x

                    # Append normalized CSR components
                    nnz_n = norm_chunk.nnz
                    if nnz_n > 0:
                        n_data.resize(total_nnz_n + nnz_n, axis=0)
                        n_data[total_nnz_n:] = norm_chunk.data
                        n_indices.resize(total_nnz_n + nnz_n, axis=0)
                        n_indices[total_nnz_n:] = norm_chunk.indices.astype(np.int32)
                    for row_nnz in np.diff(norm_chunk.indptr):
                        n_indptr.append(n_indptr[-1] + row_nnz)
                    total_nnz_n += nnz_n

                    lib_sizes.append(split_lib)
                    del split_chunk, norm_chunk
                    gc.collect()

                # Write indptr arrays and CSR metadata
                x_grp.create_dataset("indptr", data=np.array(x_indptr, dtype=np.int64))
                x_grp.attrs["encoding-type"] = "csr_matrix"
                x_grp.attrs["encoding-version"] = "0.1.0"
                x_grp.attrs["shape"] = [n_cells, n_vars]

                n_grp.create_dataset("indptr", data=np.array(n_indptr, dtype=np.int64))
                n_grp.attrs["encoding-type"] = "csr_matrix"
                n_grp.attrs["encoding-version"] = "0.1.0"
                n_grp.attrs["shape"] = [n_cells, n_vars]

                layers_grp.attrs["encoding-type"] = "dict"
                layers_grp.attrs["encoding-version"] = "0.1.0"

                # obs
                library_size = np.concatenate(lib_sizes)
                std_obs = build_standardized_obs(
                    obs, obs_indices, cell_type_col, batch_key,
                    dataset_name, library_size, extra_obs_columns,
                )
                _write_obs_to_h5(out_f.create_group("obs"), std_obs)

                # var
                _write_var_to_h5(out_f.create_group("var"), var_out)

                # obsm
                if obsm_data:
                    obsm_grp = out_f.create_group("obsm")
                    obsm_grp.attrs["encoding-type"] = "dict"
                    obsm_grp.attrs["encoding-version"] = "0.1.0"
                    for k, arr in obsm_data.items():
                        ds = obsm_grp.create_dataset(k, data=arr[obs_indices])
                        ds.attrs["encoding-type"] = "array"
                        ds.attrs["encoding-version"] = "0.2.0"

                # anndata root attributes
                out_f.attrs["encoding-type"] = "anndata"
                out_f.attrs["encoding-version"] = "0.1.0"

        file_size = os.path.getsize(out_path) / (1024 * 1024)
        stats[s] = {
            "n_cells": n_cells,
            "nnz_counts": total_nnz_x,
            "nnz_normalized": total_nnz_n,
            "library_size_mean": float(np.mean(library_size)),
            "library_size_median": float(np.median(library_size)),
            "file_size_mb": float(file_size),
        }
        del obs_indices, library_size, std_obs
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
