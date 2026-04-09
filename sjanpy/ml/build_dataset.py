#!/usr/bin/env python3
"""
Build PyTorch-ready datasets from h5ad files.

Streams a single h5ad file via h5py (never loads full matrix), subsets to
selected gene columns, builds condition vectors, and writes output tensors.

Two output formats:

- **safetensors** (default): one `{split}.safetensors` file per split.
  Enables fast DMA-based GPU loading via ``safetensors.torch.load_file(path, device="cuda")``.
- **pt_chunks** (legacy): multiple ``chunk_NNNN.pt`` files per split.
  For backward compatibility or streaming on very low-memory machines.

Three ways to use
-----------------

1. As a library::

    from sjanpy.ml import build_dataset
    import torch

    build_dataset(
        input_path="data/train.h5ad",
        output_dir="pt_chunks/",
        gene_list=None, gene_col="highly_variable",
        label_col="cell_type",
        numerical_specs=[], cat_specs=[{"source": "batch", "encoding": "onehot"}],
        chunk_size=50000,
        save_schema_path="pt_chunks/condition-schema.json",
        load_schema_path=None,
        output_format="safetensors",    # or "pt_chunks"
        counts_dtype=torch.bfloat16,    # or None for fp32
    )

2. As a CLI module::

    # safetensors + bf16 (recommended)
    python -m sjanpy.ml.build_dataset \
        --h5ad-input data/train.h5ad --output pt_chunks/ \
        --gene-col highly_variable --cell-type-label-col cell_type \
        --cat-cond "batch:onehot" --format safetensors --counts-dtype bf16 \
        --save-schema pt_chunks/condition-schema.json

    # val/test: reuse train's schema
    python -m sjanpy.ml.build_dataset \
        --h5ad-input data/val.h5ad --output pt_chunks/ \
        --gene-col highly_variable --cell-type-label-col cell_type \
        --load-schema pt_chunks/condition-schema.json \
        --format safetensors --counts-dtype bf16

    # legacy pt_chunks format
    python -m sjanpy.ml.build_dataset \
        --h5ad-input data/train.h5ad --output pt_chunks/train/ \
        --format pt_chunks --counts-dtype fp32 ...

CLI flags
---------

--h5ad-input          Path to a single h5ad file.
--output              Output directory. safetensors writes {stem}.safetensors here;
                      pt_chunks writes chunk_NNNN.pt files here.
--gene-list           Gene list file (txt one-per-line, or JSON with 'genes'/'hvg_genes' key).
--gene-col            Boolean column in var to filter genes (e.g. 'highly_variable').
--cell-type-label-col Obs column for cell type labels. Omit to skip labels.
--cat-cond            Categorical condition spec. Repeatable. E.g. 'batch:onehot'.
--numerical-cond      Numerical condition spec. Repeatable. E.g. 'library_size:log1p:zscore'.
--save-schema         Save condition schema + label mapping to JSON (use on train).
--load-schema         Load schema from JSON (use on val/test to reuse train stats).
--format              Output format: 'safetensors' (default) or 'pt_chunks' (legacy).
--counts-dtype        Dtype for counts: 'fp32' (default), 'bf16', or 'fp16'.
                      bf16 halves disk/memory with negligible precision loss for UMI counts.
--chunk-size          Cells per read-chunk when streaming h5ad (default 50000).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from .h5ad_io import read_obs as read_obs_h5py, read_var as read_var_h5py


# ---------------------------------------------------------------------------
# Gene filtering
# ---------------------------------------------------------------------------

def load_gene_list(path: str) -> list[str]:
    """Load a gene list from a text file (one gene per line) or JSON.

    JSON format: either a plain list ["gene1", "gene2", ...] or an object
    with a "genes" or "hvg_genes" key.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(g) for g in data]
        for key in ("genes", "hvg_genes"):
            if key in data:
                return [str(g) for g in data[key]]
        raise ValueError(f"JSON gene list must be a list or have a 'genes'/'hvg_genes' key: {path}")
    else:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def resolve_gene_indices(var_df: pd.DataFrame, gene_list: list[str] | None,
                         gene_col: str | None) -> tuple[np.ndarray, list[str]]:
    """Determine which gene columns to keep and their indices.

    Args:
        var_df: var DataFrame from the h5ad file
        gene_list: explicit list of gene names to keep (from --gene-list)
        gene_col: boolean column in var to filter on (from --gene-col)

    Returns:
        (gene_indices, gene_names) where gene_indices is a sorted int array
        of column positions and gene_names is the corresponding gene names.
    """
    all_gene_names = var_df.index.tolist()

    if gene_list is not None and gene_col is not None:
        raise ValueError("Cannot specify both --gene-list and --gene-col")

    if gene_list is not None:
        gene_set = set(gene_list)
        missing = gene_set - set(all_gene_names)
        if missing:
            raise ValueError(
                f"{len(missing)} genes from --gene-list not found in h5ad var index. "
                f"First 5: {sorted(missing)[:5]}"
            )
        name_to_idx = {name: i for i, name in enumerate(all_gene_names)}
        indices = sorted(name_to_idx[g] for g in gene_list)
        names = [all_gene_names[i] for i in indices]
        return np.array(indices, dtype=np.int64), names

    if gene_col is not None:
        if gene_col not in var_df.columns:
            raise ValueError(
                f"Gene column '{gene_col}' not found in var. "
                f"Available: {list(var_df.columns)}"
            )
        mask = var_df[gene_col].values.astype(bool)
        indices = np.where(mask)[0]
        names = var_df.index[mask].tolist()
        return indices, names

    # No filtering: use all genes
    return np.arange(len(var_df), dtype=np.int64), all_gene_names


# ---------------------------------------------------------------------------
# Condition vector logic
# ---------------------------------------------------------------------------

def parse_numerical_spec(spec_str: str) -> dict:
    """Parse 'library_size:log1p:zscore' -> {"source": "library_size", "transforms": ["log1p", "zscore"]}"""
    spec_str = spec_str.strip()
    parts = spec_str.split(":")
    source = parts[0]
    transforms = parts[1:] if len(parts) > 1 else []
    # validate transforms
    for t in transforms:
        if t not in ("log1p", "zscore"):
            raise ValueError(f"Unknown transform '{t}'. Supported: log1p, zscore")
    # validate order: log1p must come before zscore
    if "log1p" in transforms and "zscore" in transforms:
        if transforms.index("log1p") > transforms.index("zscore"):
            raise ValueError("log1p must come before zscore in transform chain")
    return {"source": source, "transforms": transforms}


def parse_cat_spec(spec_str: str) -> list[dict]:
    """Parse categorical spec. Returns list (groupmean with multiple sources expands).

    'batch:onehot' -> [{"source": "batch", "encoding": "onehot"}]
    'batch:groupmean(n_genes:log1p, n_counts)' -> [
        {"source": "batch", "encoding": "groupmean", "value_col": "n_genes", "value_transforms": ["log1p"]},
        {"source": "batch", "encoding": "groupmean", "value_col": "n_counts", "value_transforms": []},
    ]
    """
    spec_str = spec_str.strip()
    # Split on first ':'
    colon_idx = spec_str.index(":")
    source = spec_str[:colon_idx]
    rest = spec_str[colon_idx+1:]

    if rest == "onehot":
        return [{"source": source, "encoding": "onehot"}]
    elif rest.startswith("groupmean(") and rest.endswith(")"):
        inner = rest[len("groupmean("):-1]
        # Split on ',' but respect nested content
        source_specs = [s.strip() for s in inner.split(",")]
        results = []
        for ss in source_specs:
            parts = ss.split(":")
            value_col = parts[0]
            value_transforms = parts[1:] if len(parts) > 1 else []
            for t in value_transforms:
                if t not in ("log1p", "zscore"):
                    raise ValueError(f"Unknown transform '{t}' in groupmean source spec")
            results.append({
                "source": source,
                "encoding": "groupmean",
                "value_col": value_col,
                "value_transforms": value_transforms,
            })
        return results
    else:
        raise ValueError(f"Unknown categorical encoding '{rest}'. Supported: onehot, groupmean(...)")


def apply_transforms(values, transforms, stats=None, fit=False):
    """Apply transform chain. If fit=True, compute and return stats for zscore."""
    computed_stats = {}
    for t in transforms:
        if t == "log1p":
            if (values < 0).any():
                raise ValueError(f"log1p requires non-negative values, found min={values.min()}")
            values = np.log1p(values)
        elif t == "zscore":
            if fit:
                mean = float(values.mean())
                std = float(values.std())
                if std == 0:
                    raise ValueError("zscore: std=0 (constant column)")
                computed_stats = {"mean": mean, "std": std}
            else:
                mean = stats["mean"]
                std = stats["std"]
            values = (values - mean) / std
    return values, computed_stats


def build_condition_schema(obs_df, numerical_specs, cat_specs):
    """Build condition columns with stats from the given obs DataFrame."""
    condition_columns = []
    offset = 0

    # Numerical columns
    for spec in numerical_specs:
        source = spec["source"]
        transforms = spec["transforms"]
        if source not in obs_df.columns:
            raise ValueError(f"Column '{source}' not found in obs. Available: {list(obs_df.columns)}")

        values = obs_df[source].values.astype(np.float32)
        _, stats = apply_transforms(values, transforms, fit=True)

        name_parts = [source] + transforms
        condition_columns.append({
            "name": "_".join(name_parts),
            "kind": "numerical",
            "source": source,
            "transforms": transforms,
            "stats": stats,
            "dim": 1,
            "offset": offset,
        })
        offset += 1

    # Categorical columns
    for spec in cat_specs:
        source = spec["source"]
        if source not in obs_df.columns:
            raise ValueError(f"Column '{source}' not found in obs. Available: {list(obs_df.columns)}")

        if spec["encoding"] == "onehot":
            col_vals = obs_df[source]
            if hasattr(col_vals, "cat"):
                categories = sorted(col_vals.cat.categories.tolist())
            else:
                categories = sorted(col_vals.unique().tolist())
            categories = [str(c) for c in categories]
            dim = len(categories)

            condition_columns.append({
                "name": f"{source}_onehot",
                "kind": "categorical",
                "encoding": "onehot",
                "source": source,
                "categories": categories,
                "dim": dim,
                "offset": offset,
            })
            offset += dim

        elif spec["encoding"] == "groupmean":
            value_col = spec["value_col"]
            value_transforms = spec["value_transforms"]

            if value_col not in obs_df.columns:
                raise ValueError(f"Value column '{value_col}' not found in obs. Available: {list(obs_df.columns)}")

            # Compute group means
            values = obs_df[value_col].values.astype(np.float32)
            transformed_values, _ = apply_transforms(values, value_transforms, fit=True)

            groups = obs_df[source]
            if hasattr(groups, "cat"):
                groups = groups.astype(str)

            # Build mean table
            df_tmp = pd.DataFrame({"group": groups.values, "value": transformed_values})
            mean_table = df_tmp.groupby("group")["value"].mean().to_dict()
            mean_table = {str(k): float(v) for k, v in mean_table.items()}

            name = f"{source}_groupmean_{value_col}"
            condition_columns.append({
                "name": name,
                "kind": "categorical",
                "encoding": "groupmean",
                "source": source,
                "value_col": value_col,
                "value_transforms": value_transforms,
                "mean_table": mean_table,
                "dim": 1,
                "offset": offset,
            })
            offset += 1

    return condition_columns, offset


def build_condition_tensor(obs_chunk, condition_columns, n_cond):
    """Convert obs rows to a condition tensor.

    Args:
        obs_chunk: pd.DataFrame with obs columns for this chunk
        condition_columns: list of condition column metadata dicts
        n_cond: total condition vector dimensionality

    Returns:
        torch.FloatTensor of shape (n_cells, n_cond)
    """
    n_cells = len(obs_chunk)
    if n_cond == 0:
        return torch.zeros(n_cells, 0, dtype=torch.float32)

    cond = np.zeros((n_cells, n_cond), dtype=np.float32)

    for col_info in condition_columns:
        offset = col_info["offset"]

        if col_info["kind"] == "numerical":
            values = obs_chunk[col_info["source"]].values.astype(np.float32)
            values, _ = apply_transforms(values, col_info["transforms"], stats=col_info.get("stats"))
            cond[:, offset] = values

        elif col_info["kind"] == "categorical":
            if col_info["encoding"] == "onehot":
                cat_to_idx = {c: i for i, c in enumerate(col_info["categories"])}
                source_vals = obs_chunk[col_info["source"]]
                if hasattr(source_vals, "cat"):
                    source_vals = source_vals.astype(str)
                indices = np.array([cat_to_idx.get(str(v), -1) for v in source_vals])
                if (indices == -1).any():
                    unknown = set(str(v) for v, idx in zip(source_vals, indices) if idx == -1)
                    raise ValueError(f"Unknown categories in '{col_info['source']}': {unknown}")
                cond[np.arange(n_cells), offset + indices] = 1.0

            elif col_info["encoding"] == "groupmean":
                mean_table = col_info["mean_table"]
                source_vals = obs_chunk[col_info["source"]]
                if hasattr(source_vals, "cat"):
                    source_vals = source_vals.astype(str)
                mapped = source_vals.astype(str).map(mean_table)
                if mapped.isna().any():
                    unknown = set(source_vals.astype(str)[mapped.isna()])
                    raise ValueError(f"Unknown categories in '{col_info['source']}' for groupmean: {unknown}")
                cond[:, offset] = mapped.values

    return torch.from_numpy(cond)


# ---------------------------------------------------------------------------
# Condition schema I/O
# ---------------------------------------------------------------------------

def save_condition_schema(path: str, condition_columns: list[dict], n_cond: int,
                          label_to_idx: dict | None, n_labels: int):
    """Save condition schema + label mapping to JSON."""
    # Make categories JSON-safe
    cols_serializable = []
    for cc in condition_columns:
        cc_copy = dict(cc)
        if "categories" in cc_copy:
            cc_copy["categories"] = [str(c) for c in cc_copy["categories"]]
        cols_serializable.append(cc_copy)

    schema = {
        "condition_columns": cols_serializable,
        "n_cond": n_cond,
        "label_to_idx": label_to_idx,
        "n_labels": n_labels,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved to {path}")


def load_condition_schema(path: str) -> dict:
    """Load condition schema + label mapping from JSON."""
    with open(path) as f:
        schema = json.load(f)
    required = {"condition_columns", "n_cond"}
    missing = required - set(schema.keys())
    if missing:
        raise ValueError(f"condition-schema.json missing keys: {missing}")
    return schema


# ---------------------------------------------------------------------------
# Streaming h5ad → tensor conversion (shared by both output formats)
# ---------------------------------------------------------------------------

def _stream_h5ad_to_tensors(h5ad_path, gene_indices, n_total_genes,
                            condition_columns, n_cond, label_col, label_to_idx,
                            obs_df, chunk_size, counts_dtype=None):
    """Stream h5ad in row chunks, yield (counts, condition, labels) tensors per chunk.

    Args:
        counts_dtype: optional torch dtype for counts (e.g. torch.bfloat16).
    """
    n_genes = len(gene_indices)
    gene_membership = np.zeros(n_total_genes, dtype=bool)
    gene_membership[gene_indices] = True
    remap_vec = np.full(n_total_genes, -1, dtype=np.int64)
    remap_vec[gene_indices] = np.arange(n_genes)

    with h5py.File(str(h5ad_path), "r") as f:
        x_group = f["X"]
        indptr = x_group["indptr"][:]
        n_rows = len(indptr) - 1
        h5_data = x_group["data"]
        h5_indices = x_group["indices"]

        n_chunks = math.ceil(n_rows / chunk_size)
        print(f"  {n_rows} cells, {n_chunks} read-chunks, {n_genes} genes")

        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, n_rows)
            chunk_n = row_end - row_start

            ptr_start = int(indptr[row_start])
            ptr_end = int(indptr[row_end])

            if ptr_end == ptr_start:
                counts_dense = np.zeros((chunk_n, n_genes), dtype=np.float32)
            else:
                chunk_data = h5_data[ptr_start:ptr_end]
                chunk_indices = h5_indices[ptr_start:ptr_end]

                keep = gene_membership[chunk_indices]
                filtered_data = chunk_data[keep]
                filtered_indices = remap_vec[chunk_indices[keep]]

                chunk_indptr_orig = indptr[row_start:row_end + 1] - ptr_start
                cumsum_keep = np.cumsum(keep)
                new_indptr = np.empty(chunk_n + 1, dtype=np.int64)
                new_indptr[0] = 0
                ends = chunk_indptr_orig[1:].astype(np.int64)
                new_indptr[1:] = np.where(ends > 0, cumsum_keep[ends - 1], 0)

                chunk_csr = csr_matrix(
                    (filtered_data, filtered_indices, new_indptr),
                    shape=(chunk_n, n_genes),
                )
                counts_dense = chunk_csr.toarray().astype(np.float32)

            counts_tensor = torch.from_numpy(counts_dense)
            if counts_dtype is not None:
                counts_tensor = counts_tensor.to(counts_dtype)

            obs_chunk = obs_df.iloc[row_start:row_end]
            cond_tensor = build_condition_tensor(obs_chunk, condition_columns, n_cond)

            if label_col is not None and label_to_idx is not None:
                label_vals = obs_chunk[label_col]
                if hasattr(label_vals, "cat"):
                    label_vals = label_vals.astype(str)
                labels_tensor = torch.tensor(
                    label_vals.astype(str).map(label_to_idx).values, dtype=torch.long,
                )
            else:
                labels_tensor = torch.zeros(chunk_n, dtype=torch.long)

            yield counts_tensor, cond_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Output format: pt_chunks (legacy, multiple .pt files)
# ---------------------------------------------------------------------------

def process_file(h5ad_path, output_dir, gene_indices, n_total_genes,
                 condition_columns, n_cond, label_col, label_to_idx,
                 obs_df, chunk_size, counts_dtype=None):
    """Stream h5ad → write multiple .pt chunk files (legacy format)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_sizes = []
    for chunk_idx, (counts, cond, labels) in enumerate(
        _stream_h5ad_to_tensors(
            h5ad_path, gene_indices, n_total_genes,
            condition_columns, n_cond, label_col, label_to_idx,
            obs_df, chunk_size, counts_dtype,
        )
    ):
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save({"counts": counts, "condition": cond, "labels": labels}, chunk_path)
        chunk_sizes.append(len(counts))
        print(f"    chunk {chunk_idx}: {len(counts)} cells -> {chunk_path.name}")

    n_cells = sum(chunk_sizes)
    return {"n_cells": n_cells, "n_chunks": len(chunk_sizes), "chunk_sizes": chunk_sizes}


# ---------------------------------------------------------------------------
# Output format: safetensors (single file per split)
# ---------------------------------------------------------------------------

def process_file_safetensors(h5ad_path, output_path, gene_indices, n_total_genes,
                             condition_columns, n_cond, label_col, label_to_idx,
                             obs_df, chunk_size, counts_dtype=None):
    """Stream h5ad → pre-allocated tensors → write a single .safetensors file.

    Pre-allocates output tensors based on the known total cell count, then
    fills them chunk-by-chunk.  Peak memory ≈ 1× the final tensor size
    (no temporary list of chunks or torch.cat copy).

    Args:
        output_path: path to the .safetensors file (e.g. "train/train.safetensors")
    """
    from safetensors.torch import save_file

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = len(obs_df)
    n_genes = len(gene_indices)
    ct_dtype = counts_dtype if counts_dtype is not None else torch.float32

    # Pre-allocate contiguous tensors (no accumulation, no torch.cat)
    counts_out = torch.empty(n_cells, n_genes, dtype=ct_dtype)
    cond_out = torch.empty(n_cells, n_cond, dtype=torch.float32)
    labels_out = torch.empty(n_cells, dtype=torch.long)

    cursor = 0
    for counts, cond, labels in _stream_h5ad_to_tensors(
        h5ad_path, gene_indices, n_total_genes,
        condition_columns, n_cond, label_col, label_to_idx,
        obs_df, chunk_size, counts_dtype,
    ):
        n = len(counts)
        counts_out[cursor:cursor + n] = counts
        cond_out[cursor:cursor + n] = cond
        labels_out[cursor:cursor + n] = labels
        cursor += n

    print(f"  Writing {output_path} ({n_cells:,} cells, "
          f"counts {counts_out.dtype}, {counts_out.nelement() * counts_out.element_size() / 1e9:.1f}GB)")

    save_file({"counts": counts_out, "condition": cond_out, "labels": labels_out}, str(output_path))

    return {"n_cells": n_cells}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(input_path, output_dir, gene_list, gene_col, label_col,
                  numerical_specs, cat_specs, chunk_size,
                  save_schema_path, load_schema_path,
                  output_format="safetensors", counts_dtype=None):
    """Build dataset from a single h5ad file.

    Args:
        output_format: "safetensors" (single file, fast loading) or "pt_chunks" (legacy).
        counts_dtype: torch dtype for counts tensor (e.g. torch.bfloat16). None = fp32.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return False

    print(f"\n{'='*60}")
    print(f"  Building dataset: {input_path.name}")
    print(f"{'='*60}")

    # --- Determine gene subset ---
    var_df = read_var_h5py(str(input_path))
    n_total_genes = len(var_df)

    gene_indices, gene_names = resolve_gene_indices(var_df, gene_list, gene_col)
    n_genes = len(gene_indices)
    if gene_list is not None or gene_col is not None:
        print(f"  Genes: {n_genes} selected (of {n_total_genes} total)")
    else:
        print(f"  Genes: {n_genes} (all genes)")

    # --- Read obs ---
    print("  Reading obs...")
    obs_df = read_obs_h5py(str(input_path))
    print(f"  Cells: {len(obs_df)}")

    # --- Condition schema ---
    if load_schema_path:
        print(f"  Loading schema from {load_schema_path}")
        schema = load_condition_schema(load_schema_path)
        condition_columns = schema["condition_columns"]
        n_cond = schema["n_cond"]
        label_to_idx = schema.get("label_to_idx")
        n_labels = schema.get("n_labels", 0)
        # Override label_col from schema presence
        if label_to_idx is not None and label_col is None:
            # Schema has labels but caller didn't specify --label-col.
            # We need to know which column to read. Check if schema recorded it.
            # For now, require --label-col even when loading schema, unless
            # the schema was saved without labels.
            pass
    else:
        # Build schema from this file's obs
        condition_columns, n_cond = build_condition_schema(obs_df, numerical_specs, cat_specs)

        # Build label mapping
        if label_col is not None:
            if label_col not in obs_df.columns:
                raise ValueError(
                    f"Label column '{label_col}' not found in obs. "
                    f"Available: {list(obs_df.columns)}"
                )
            label_vals = obs_df[label_col]
            if hasattr(label_vals, "cat"):
                all_labels = sorted(label_vals.cat.categories.tolist())
            else:
                all_labels = sorted(label_vals.unique().tolist())
            label_to_idx = {str(label): i for i, label in enumerate(all_labels)}
            n_labels = len(label_to_idx)
        else:
            label_to_idx = None
            n_labels = 0

    print(f"  Conditions: {n_cond} dims from {len(condition_columns)} columns")
    for cc in condition_columns:
        kind_label = cc["kind"]
        if cc["kind"] == "categorical":
            kind_label = f"{cc['kind']}/{cc['encoding']}"
        print(f"    {cc['name']}: {kind_label}, dim={cc['dim']}")

    if label_to_idx is not None:
        print(f"  Labels: {n_labels} classes (from '{label_col}')")
    else:
        print(f"  Labels: none")

    # --- Save schema if requested ---
    if save_schema_path:
        save_condition_schema(save_schema_path, condition_columns, n_cond,
                              label_to_idx, n_labels)

    # --- Process file ---
    dtype_str = str(counts_dtype).replace("torch.", "") if counts_dtype else "fp32"
    print(f"  Format: {output_format}, counts dtype: {dtype_str}")

    if output_format == "safetensors":
        st_path = output_dir / f"{input_path.stem}.safetensors"
        file_info = process_file_safetensors(
            input_path, st_path, gene_indices, n_total_genes,
            condition_columns, n_cond, label_col, label_to_idx,
            obs_df, chunk_size, counts_dtype,
        )
    else:
        file_info = process_file(
            input_path, output_dir, gene_indices, n_total_genes,
            condition_columns, n_cond, label_col, label_to_idx,
            obs_df, chunk_size, counts_dtype,
        )

    # --- Write metadata ---
    cond_cols_serializable = []
    for cc in condition_columns:
        cc_copy = dict(cc)
        if "categories" in cc_copy:
            cc_copy["categories"] = [str(c) for c in cc_copy["categories"]]
        cond_cols_serializable.append(cc_copy)

    metadata = {
        "format": output_format,
        "counts_dtype": dtype_str,
        "input_file": str(input_path),
        "n_genes": n_genes,
        "gene_names": gene_names,
        "n_cond": n_cond,
        "condition_columns": cond_cols_serializable,
        "label_col": label_col,
        "label_to_idx": label_to_idx,
        "n_labels": n_labels,
        "n_cells": file_info["n_cells"],
    }
    if output_format == "pt_chunks":
        metadata["chunk_size"] = chunk_size
        metadata["n_chunks"] = file_info["n_chunks"]
        metadata["chunk_sizes"] = file_info["chunk_sizes"]

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata written to {metadata_path}")
    print(f"  Done: {input_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build PyTorch-ready datasets from h5ad files (safetensors or pt_chunks)",
        epilog=(
            "Three ways to use this tool:\n"
            "  1. Library:  from sjanpy.ml import build_dataset\n"
            "  2. Module:   python -m sjanpy.ml.build_dataset --h5ad-input ... --output ...\n"
            "  3. Wrapper:  python 0_build_dataset.py --h5ad-input ... --output ...\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--h5ad-input", required=True,
        help="Path to h5ad file",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory. safetensors: writes {stem}.safetensors + metadata.json here. "
             "pt_chunks: writes chunk_NNNN.pt + metadata.json here.",
    )

    # Gene filtering (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group()
    gene_group.add_argument(
        "--gene-list",
        help="Path to gene list file (txt: one per line, or JSON with 'genes'/'hvg_genes' key). "
             "Only these genes are kept.",
    )
    gene_group.add_argument(
        "--gene-col",
        help="Boolean column name in var to filter genes (e.g. 'highly_variable'). "
             "Only genes where this column is True are kept.",
    )

    # Labels
    parser.add_argument(
        "--cell-type-label-col",
        help="Obs column name for cell type labels (e.g. 'cell_type'). "
             "If omitted, no labels are generated.",
    )

    # Conditions
    parser.add_argument(
        "--numerical-cond", action="append", default=[],
        help="Numerical condition spec, e.g. 'library_size:log1p:zscore'. Repeatable.",
    )
    parser.add_argument(
        "--cat-cond", action="append", default=[],
        help="Categorical condition spec, e.g. 'batch:onehot' or "
             "'batch:groupmean(n_genes:log1p,n_counts)'. Repeatable.",
    )

    # Schema I/O
    parser.add_argument(
        "--save-schema",
        help="Save condition schema + label mapping to this JSON file.",
    )
    parser.add_argument(
        "--load-schema",
        help="Load condition schema + label mapping from this JSON file. "
             "When set, --cat-cond and --numerical-cond are ignored.",
    )

    # Output format
    parser.add_argument(
        "--format", choices=["safetensors", "pt_chunks"], default="safetensors",
        help="Output format: 'safetensors' (single file, fast) or 'pt_chunks' (legacy). Default: safetensors.",
    )
    parser.add_argument(
        "--counts-dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
        help="Dtype for counts tensor. bf16 halves disk/memory. Default: fp32.",
    )

    # Chunking (used for h5ad read batching, and for pt_chunks output)
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Cells per read-chunk when streaming h5ad (default: 50000). "
             "Also the chunk size for pt_chunks output format.",
    )

    args = parser.parse_args()

    # Parse condition specs
    if args.load_schema:
        if args.numerical_cond or args.cat_cond:
            print("WARNING: --load-schema is set, ignoring --numerical-cond and --cat-cond")
        numerical_specs = []
        cat_specs = []
    else:
        numerical_specs = [parse_numerical_spec(s) for s in args.numerical_cond]
        cat_specs = []
        for s in args.cat_cond:
            cat_specs.extend(parse_cat_spec(s))

    # Load gene list if provided
    gene_list = None
    if args.gene_list:
        gene_list = load_gene_list(args.gene_list)
        print(f"Gene list: {len(gene_list)} genes from {args.gene_list}")

    print(f"Input: {args.h5ad_input}")
    print(f"Output: {args.output}")
    if args.gene_col:
        print(f"Gene filter: var['{args.gene_col}'] == True")
    elif gene_list:
        print(f"Gene filter: {len(gene_list)} genes from file")
    else:
        print(f"Gene filter: all genes")
    # Resolve counts dtype
    _dtype_map = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
    counts_dtype = _dtype_map[args.counts_dtype]

    print(f"Label column: {args.cell_type_label_col or '(none)'}")
    print(f"Format: {args.format}, counts dtype: {args.counts_dtype}")
    print(f"Chunk size: {args.chunk_size}")

    ok = build_dataset(
        input_path=args.h5ad_input,
        output_dir=args.output,
        gene_list=gene_list,
        gene_col=args.gene_col,
        label_col=args.cell_type_label_col,
        numerical_specs=numerical_specs,
        cat_specs=cat_specs,
        chunk_size=args.chunk_size,
        save_schema_path=args.save_schema,
        load_schema_path=args.load_schema,
        output_format=args.format,
        counts_dtype=counts_dtype,
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
