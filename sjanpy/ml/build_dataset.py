#!/usr/bin/env python3
"""
Build chunked .pt datasets from h5ad files.

Streams a single h5ad file via h5py (never loads full matrix), subsets to
selected gene columns, builds condition vectors, and writes .pt chunk files
+ metadata.json.

Three ways to use
-----------------

1. As a library::

    from sjanpy.ml import build_dataset, read_obs_h5py, build_condition_schema

    build_dataset(
        input_path="data/train.h5ad",
        output_dir="chunks/train/",
        gene_list=None, gene_col="highly_variable",
        label_col="cell_type",
        numerical_specs=[], cat_specs=[{"source": "batch", "encoding": "onehot"}],
        chunk_size=50000,
        save_schema_path="condition-schema.json",
        load_schema_path=None,
    )

2. As a CLI module::

    python -m sjanpy.ml.build_dataset \
        --h5ad-input data/train.h5ad --output chunks/train/ \
        --gene-col highly_variable --cell-type-label-col cell_type \
        --cat-cond "batch:onehot" --save-schema condition-schema.json

3. Via the experiment wrapper script::

    python 0_build_dataset.py \
        --h5ad-input data/train.h5ad --output chunks/train/ \
        --gene-col highly_variable --cell-type-label-col cell_type \
        --cat-cond "batch:onehot" --save-schema condition-schema.json

CLI flags
---------

--h5ad-input          Path to a single h5ad file.
--output              Output directory for .pt chunks and metadata.json.
--gene-list           Gene list file (txt one-per-line, or JSON with 'genes'/'hvg_genes' key).
--gene-col            Boolean column in var to filter genes (e.g. 'highly_variable').
--cell-type-label-col Obs column for cell type labels. Omit to skip labels.
--cat-cond            Categorical condition spec. Repeatable. E.g. 'batch:onehot'.
--numerical-cond      Numerical condition spec. Repeatable. E.g. 'library_size:log1p:zscore'.
--save-schema         Save condition schema + label mapping to JSON (use on train).
--load-schema         Load schema from JSON (use on val/test to reuse train stats).
--chunk-size          Cells per .pt chunk (default 50000).
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
# Chunked h5ad streaming and .pt writing
# ---------------------------------------------------------------------------

def process_file(h5ad_path, output_dir, gene_indices, n_total_genes,
                 condition_columns, n_cond, label_col, label_to_idx,
                 obs_df, chunk_size):
    """Stream one h5ad file in row chunks, write .pt files.

    Args:
        h5ad_path: path to h5ad file
        output_dir: output directory for .pt files
        gene_indices: numpy array of gene column indices to keep (sorted)
        n_total_genes: total number of genes in the h5ad (for boolean lookup sizing)
        condition_columns: condition column metadata
        n_cond: total condition vector dim
        label_col: obs column name for labels, or None to skip labels
        label_to_idx: dict mapping label -> int index, or None
        obs_df: preloaded obs DataFrame
        chunk_size: rows per chunk

    Returns:
        dict with n_cells, n_chunks, chunk_sizes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_genes = len(gene_indices)
    # Boolean lookup for O(1) gene membership test (replaces np.isin per chunk)
    gene_membership = np.zeros(n_total_genes, dtype=bool)
    gene_membership[gene_indices] = True
    # Vectorized column remap: remap_vec[original_col] = new_col_index
    remap_vec = np.full(n_total_genes, -1, dtype=np.int64)
    remap_vec[gene_indices] = np.arange(n_genes)

    with h5py.File(str(h5ad_path), "r") as f:
        x_group = f["X"]
        indptr = x_group["indptr"][:]
        n_rows = len(indptr) - 1
        h5_data = x_group["data"]
        h5_indices = x_group["indices"]

        n_chunks = math.ceil(n_rows / chunk_size)
        print(f"  {n_rows} cells, {n_chunks} chunks, {n_genes} genes")
        chunk_sizes = []

        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, n_rows)
            chunk_n = row_end - row_start

            # --- Read sparse rows and subset to selected gene columns ---
            ptr_start = int(indptr[row_start])
            ptr_end = int(indptr[row_end])

            if ptr_end == ptr_start:
                counts_dense = np.zeros((chunk_n, n_genes), dtype=np.float32)
            else:
                chunk_data = h5_data[ptr_start:ptr_end]
                chunk_indices = h5_indices[ptr_start:ptr_end]

                # Filter to selected gene columns via boolean lookup (O(1) per element)
                keep = gene_membership[chunk_indices]
                filtered_data = chunk_data[keep]
                filtered_indices = remap_vec[chunk_indices[keep]]

                # Rebuild indptr for the gene-filtered sparse rows.
                # chunk_indptr_orig gives per-row nnz boundaries (shifted to 0).
                # cumsum_keep[i] = number of kept elements in positions 0..i.
                # For row r, its nnz-end in the original chunk is ends[r]; the
                # corresponding end in the filtered array is cumsum_keep[ends[r]-1].
                # When ends[r]==0 the row had no nonzeros at all, so its filtered
                # end is also 0 — hence the np.where guard.
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

            # --- Build condition vector ---
            obs_chunk = obs_df.iloc[row_start:row_end]
            cond_tensor = build_condition_tensor(obs_chunk, condition_columns, n_cond)

            # --- Build labels ---
            if label_col is not None and label_to_idx is not None:
                label_vals = obs_chunk[label_col]
                if hasattr(label_vals, "cat"):
                    label_vals = label_vals.astype(str)
                labels_tensor = torch.tensor(
                    label_vals.astype(str).map(label_to_idx).values, dtype=torch.long,
                )
            else:
                labels_tensor = torch.zeros(chunk_n, dtype=torch.long)

            # --- Save chunk ---
            chunk_path = output_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save({
                "counts": counts_tensor,
                "condition": cond_tensor,
                "labels": labels_tensor,
            }, chunk_path)

            chunk_sizes.append(chunk_n)
            print(f"    chunk {chunk_idx}: {chunk_n} cells -> {chunk_path.name}")

    return {"n_cells": n_rows, "n_chunks": n_chunks, "chunk_sizes": chunk_sizes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(input_path, output_dir, gene_list, gene_col, label_col,
                  numerical_specs, cat_specs, chunk_size,
                  save_schema_path, load_schema_path):
    """Build chunked .pt dataset from a single h5ad file."""
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
    file_info = process_file(
        input_path, output_dir, gene_indices, n_total_genes,
        condition_columns, n_cond, label_col, label_to_idx,
        obs_df, chunk_size,
    )

    # --- Write metadata ---
    cond_cols_serializable = []
    for cc in condition_columns:
        cc_copy = dict(cc)
        if "categories" in cc_copy:
            cc_copy["categories"] = [str(c) for c in cc_copy["categories"]]
        cond_cols_serializable.append(cc_copy)

    metadata = {
        "input_file": str(input_path),
        "n_genes": n_genes,
        "gene_names": gene_names,
        "n_cond": n_cond,
        "condition_columns": cond_cols_serializable,
        "label_col": label_col,
        "label_to_idx": label_to_idx,
        "n_labels": n_labels,
        "chunk_size": chunk_size,
        "n_cells": file_info["n_cells"],
        "n_chunks": file_info["n_chunks"],
        "chunk_sizes": file_info["chunk_sizes"],
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata written to {metadata_path}")
    print(f"  Done: {input_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build chunked .pt datasets from h5ad files",
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
        help="Output directory for .pt chunks and metadata.json",
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

    # Chunking
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Number of cells per .pt chunk (default: 50000)",
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
    print(f"Label column: {args.cell_type_label_col or '(none)'}")
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
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
