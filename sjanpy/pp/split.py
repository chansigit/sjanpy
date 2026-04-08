"""Stratified train / val / test splitting for single-cell obs DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    obs: pd.DataFrame,
    stratify_col: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Two-stage stratified split into train / val / test.

    Parameters
    ----------
    obs
        Cell-level metadata (one row per cell).
    stratify_col
        Column in *obs* used for stratification (e.g. ``"cell_type"``).
    val_ratio, test_ratio
        Fraction of total cells for validation and test sets.
    seed
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Two columns: ``cell_index`` (int position) and ``split``
        (one of ``"train"``, ``"val"``, ``"test"``).
    """
    held_out_ratio = val_ratio + test_ratio
    n = len(obs)
    indices = np.arange(n)
    labels = obs[stratify_col].values

    # Identify rare categories (count < 2) — always go to train.
    unique, counts = np.unique(labels, return_counts=True)
    rare_cats = set(unique[counts < 2])

    rare_mask = np.array([l in rare_cats for l in labels])
    rare_idx = indices[rare_mask]
    normal_idx = indices[~rare_mask]

    # If all cells are rare, everything goes to train.
    if len(normal_idx) == 0:
        return pd.DataFrame(
            {"cell_index": indices, "split": "train"}
        )

    normal_labels = labels[~rare_mask]

    # Stage 1: split normal indices into train vs held-out.
    train_idx, held_idx = train_test_split(
        normal_idx,
        test_size=held_out_ratio,
        stratify=normal_labels,
        random_state=seed,
    )

    # Stage 2: split held-out into val vs test.
    held_labels = labels[held_idx]

    # Categories with < 2 members in held-out go to val.
    held_unique, held_counts = np.unique(held_labels, return_counts=True)
    held_rare_cats = set(held_unique[held_counts < 2])

    held_rare_mask = np.array([l in held_rare_cats for l in held_labels])
    forced_val_idx = held_idx[held_rare_mask]
    splittable_held_idx = held_idx[~held_rare_mask]

    if len(splittable_held_idx) == 0:
        # All held-out cells are rare in held-out — send to val.
        val_idx = held_idx
        test_idx = np.array([], dtype=int)
    else:
        splittable_labels = held_labels[~held_rare_mask]
        # Proportion of test within the held-out portion.
        test_in_held = test_ratio / held_out_ratio
        val_split, test_idx = train_test_split(
            splittable_held_idx,
            test_size=test_in_held,
            stratify=splittable_labels,
            random_state=seed,
        )
        val_idx = np.concatenate([forced_val_idx, val_split])

    # Combine rare indices into train.
    train_idx = np.concatenate([train_idx, rare_idx])

    # Build result DataFrame.
    split_labels = np.empty(n, dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    return pd.DataFrame(
        {"cell_index": indices, "split": split_labels}
    )
