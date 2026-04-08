"""Highly-variable gene selection with stratified sampling support."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from ..ml.h5ad_io import (
    locate_matrix,
    read_matrix_rows,
    read_h5_group,
)


def prepare_hvg_sample(
    obs: pd.DataFrame,
    train_indices: np.ndarray,
    stratify_col: str,
    target_size: int = 300_000,
    min_cells: int = 100,
    seed: int = 42,
) -> np.ndarray | None:
    """Stratified subsample of training cells for HVG computation.

    Parameters
    ----------
    obs : pandas.DataFrame
        Full observation metadata (all cells).
    train_indices : numpy.ndarray
        1-D integer array of global row indices for training cells.
    stratify_col : str
        Column in *obs* used for stratification (e.g. ``"cell_type"``).
    target_size : int
        Desired number of cells in the subsample.
    min_cells : int
        Categories with <= this many cells are kept in full.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray or None
        Sorted array of global indices, or *None* when no sampling is needed.
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    n_train = len(train_indices)

    if n_train <= target_size:
        return None

    rng = np.random.default_rng(seed)
    obs_train = obs.iloc[train_indices]
    cell_types = obs_train[stratify_col].astype(str).fillna("NA")

    # Group local indices by category
    by_type: dict[str, list[int]] = {}
    for local_idx, ct in enumerate(cell_types.values):
        by_type.setdefault(ct, []).append(local_idx)

    small_kept: list[int] = []
    large_groups: dict[str, np.ndarray] = {}
    for ct, local_indices in by_type.items():
        if len(local_indices) <= min_cells:
            small_kept.extend(local_indices)
        else:
            large_groups[ct] = np.asarray(local_indices, dtype=np.int64)

    fixed_n = len(small_kept)
    if fixed_n >= target_size:
        sampled_local = np.asarray(small_kept, dtype=np.int64)
        sampled_global = train_indices[sampled_local]
        return np.sort(sampled_global)

    remain_budget = target_size - fixed_n
    large_total = sum(len(v) for v in large_groups.values())
    alloc: dict[str, int] = {}

    if large_total > 0 and remain_budget > 0:
        frac_parts: list[tuple[float, str]] = []
        total_assigned = 0
        for ct, arr in large_groups.items():
            ideal = remain_budget * (len(arr) / large_total)
            take = int(math.floor(ideal))
            take = min(take, len(arr))
            alloc[ct] = take
            frac_parts.append((ideal - take, ct))
            total_assigned += take

        leftover = remain_budget - total_assigned
        if leftover > 0:
            frac_parts.sort(reverse=True)
            for _, ct in frac_parts:
                if leftover <= 0:
                    break
                can_add = len(large_groups[ct]) - alloc[ct]
                if can_add > 0:
                    alloc[ct] += 1
                    leftover -= 1

    sampled_local = list(small_kept)
    for ct, arr in large_groups.items():
        n_take = alloc.get(ct, 0)
        if n_take <= 0:
            continue
        chosen = rng.choice(arr, size=n_take, replace=False)
        sampled_local.extend(chosen.tolist())

    sampled_local_arr = np.asarray(sampled_local, dtype=np.int64)
    sampled_global = train_indices[sampled_local_arr]
    return np.sort(sampled_global)


def _make_var_names_unique(var_df: pd.DataFrame) -> pd.DataFrame:
    """Append suffixes to duplicate index entries to make them unique."""
    idx = var_df.index.tolist()
    seen: dict[str, int] = {}
    new_idx: list[str] = []
    for name in idx:
        if name in seen:
            seen[name] += 1
            new_idx.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            new_idx.append(name)
    var_df = var_df.copy()
    var_df.index = new_idx
    return var_df


def compute_hvg(
    h5ad_path: str | Path,
    matrix_source: str,
    cell_indices: np.ndarray,
    batch_key: str,
    matrix_value_type: str = "counts",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
) -> tuple[list[str], np.ndarray]:
    """Compute highly-variable genes from a subset of cells.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the h5ad file.
    matrix_source : str
        One of ``"raw.X"``, ``"X"``, or ``"layers/<name>"``.
    cell_indices : numpy.ndarray
        1-D integer array of cell (row) indices to use.
    batch_key : str
        Column in obs for batch correction.
    matrix_value_type : str
        ``"counts"`` or ``"normalized"``.
    min_mean, max_mean, min_disp : float
        Thresholds passed to :func:`scanpy.pp.highly_variable_genes`.

    Returns
    -------
    (hvg_genes, hvg_mask) : tuple[list[str], numpy.ndarray]
        List of HVG gene names and boolean mask over all genes.
    """
    cell_indices = np.asarray(cell_indices, dtype=np.int64)

    # Read matrix, var, and obs from a single file open
    with h5py.File(str(h5ad_path), "r") as f:
        matrix_obj, var_grp, _ = locate_matrix(f, matrix_source)
        X_sub = read_matrix_rows(matrix_obj, cell_indices)
        var_df = read_h5_group(var_grp)
        obs_full = read_h5_group(f["obs"])

    # Make gene names unique
    var_df = _make_var_names_unique(var_df)
    obs_sub = obs_full.iloc[cell_indices].copy()

    # Build temporary AnnData
    adata = sc.AnnData(X=X_sub, obs=obs_sub, var=var_df)
    adata.var_names_make_unique()

    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in obs")

    # Normalize if counts
    if matrix_value_type == "counts":
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # Run HVG
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        batch_key=batch_key,
    )

    hvg_mask = adata.var["highly_variable"].values.astype(bool)
    hvg_genes = list(adata.var_names[hvg_mask])

    return hvg_genes, hvg_mask
