# Extract Reusable scRNA-seq Standardization Pipeline into sjanpy

**Date:** 2026-04-07
**Source:** `genofoundation/data/scgraph/` — `stratified_split.py` and `build_standardized_h5ad.py`

## Problem

The scgraph dataset pipeline contains ~1200 lines of well-tested, generic scRNA-seq preprocessing code mixed with dataset-specific configuration. The generic parts (stratified splitting, HVG computation, h5ad I/O, standardized h5ad building) are reusable across any scRNA-seq project but currently live in a project-specific directory.

## Goal

Extract the generic engine into `sjanpy` so that:
1. Future scRNA-seq projects reuse the same splitting/standardization code
2. The scgraph pipeline shrinks to a thin configuration-driven script
3. Overlapping h5py reader code in `sjanpy.ml.build_dataset` is consolidated

## New Modules

### 1. `sjanpy.pp.split` — Stratified Train/Val/Test Splitting

**Source:** `stratified_split.py` lines 70–163

**Public API:**
```python
def stratified_split(
    obs: pd.DataFrame,
    stratify_col: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Two-stage stratified split. Returns DataFrame with columns: cell_index, split.
    Rare cell types (<2 cells) always placed in train."""
```

One function. No dataset-specific logic. The UMAP visualization stays in scgraph (it's a QC convenience, and `sjanpy.pl` already has embedding plotting).

### 2. `sjanpy.pp.hvg` — Train-Only HVG Computation

**Source:** `build_standardized_h5ad.py` lines 358–514

**Public API:**
```python
def prepare_hvg_sample(
    obs: pd.DataFrame,
    train_indices: np.ndarray,
    stratify_col: str,
    target_size: int = 300_000,
    min_cells: int = 100,
    seed: int = 42,
) -> np.ndarray | None:
    """Stratified subsample of training cells for HVG computation.
    Returns sampled global indices, or None if train set is small enough."""

def compute_hvg(
    h5ad_path: Path,
    matrix_source: str,
    cell_indices: np.ndarray,
    batch_key: str,
    matrix_value_type: str = "counts",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
) -> tuple[list[str], np.ndarray]:
    """Compute HVGs from specified cells. Returns (hvg_gene_names, hvg_boolean_mask).
    Uses scanpy flavor='seurat' with batch correction."""
```

Key change from source: accepts `cell_indices` directly instead of a split DataFrame. The caller decides which cells to use — the function doesn't know about train/val/test.

### 3. `sjanpy.ml.h5ad_io` — Low-Level h5ad I/O via h5py

**Source:** `build_standardized_h5ad.py` lines 90–298, plus consolidation with `build_dataset.py` lines 73–212

**Problem solved:** Both `build_standardized_h5ad.py` and `sjanpy.ml.build_dataset` have their own h5py DataFrame readers (`_read_dataframe_group` vs `read_obs_h5py`/`read_var_h5py`). They handle the same edge cases (categorical encoding, bytes decoding, legacy formats) with slightly different implementations.

**Public API:**
```python
# DataFrame readers (consolidate existing duplicates)
def read_obs(h5ad_path: Path) -> pd.DataFrame:
    """Read obs DataFrame via h5py. Handles categorical, legacy, scalar categories."""

def read_var(h5ad_path: Path, group: str = "var") -> pd.DataFrame:
    """Read var DataFrame via h5py. Supports raw/var resolution."""

# Matrix access
def locate_matrix(f: h5py.File, source: str) -> tuple[object, h5py.Group, str]:
    """Resolve 'raw.X', 'X', or 'layers/<name>' to (matrix_obj, var_group, label)."""

def get_matrix_shape(matrix_obj: object) -> tuple[int, int]:
    """Get (n_obs, n_vars) from sparse group or dense dataset."""

def read_matrix_rows(matrix_obj: object, row_indices: np.ndarray) -> csr_matrix:
    """Read specified rows as CSR. Works for both sparse groups and dense datasets."""

def read_sparse_chunk(matrix_obj, start: int, end: int, n_vars: int) -> csr_matrix:
    """Read contiguous row range from sparse CSR group. Used by streaming writers."""

# Validation
def validate_matrix_values(
    matrix_obj: object, expected_type: str,
    sample_n: int = 200_000, strict: bool = False,
) -> None:
    """Sample matrix values and check consistency with declared type (counts vs normalized)."""
```

**Migration plan for `build_dataset.py`:** After `h5ad_io` is created, refactor `build_dataset.py` to import from `h5ad_io` instead of its internal `read_obs_h5py`/`read_var_h5py`. Keep the old names as re-exports in `sjanpy.ml.__init__` for backward compatibility.

### 4. `sjanpy.ml.standardize` — Standardized h5ad Builder

**Source:** `build_standardized_h5ad.py` lines 628–1100

**Public API:**
```python
def build_standardized_h5ads(
    h5ad_path: Path,
    output_dir: Path,
    split_col: np.ndarray,          # array of "train"/"val"/"test" per cell
    hvg_mask: np.ndarray,           # boolean mask over all genes
    all_var: pd.DataFrame,          # full var DataFrame
    obs: pd.DataFrame,              # full obs DataFrame
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    matrix_source: str = "raw.X",
    chunk_size: int = 50_000,
    target_sum: float = 1e4,
    extra_obs_columns: dict[str, str] | None = None,
    streaming: bool = False,
) -> dict:
    """Build train/val/test h5ad files from a source h5ad.

    Each output h5ad contains:
      .X = raw counts (CSR sparse, float32)
      .layers['normalized'] = log1p(normalize_total(X, target_sum))
      .obs = standardized columns (cell_type, batch, tissue, dataset, library_size) + extras
      .var = all genes with highly_variable annotation
      .obsm = carried over from source

    Returns per-split stats dict.
    """
```

Internally dispatches to either accumulate-then-write (for moderate datasets) or streaming-write (for very large datasets like brain 2.5M cells), based on the `streaming` flag.

**Helper exported for reuse:**
```python
def build_standardized_obs(
    obs: pd.DataFrame,
    cell_indices: np.ndarray,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    library_size: np.ndarray,
    extra_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build standardized obs with consistent column names and types."""
```

## What Stays in scgraph

- `EXTRA_OBS_COLUMNS` dictionary (per-dataset metadata mapping)
- Shell scripts (`0.download_data.sh`, `1.*.sh`, `2.*.sh`)
- Per-dataset configuration (batch_key, cell_type_col, matrix_source choices)
- `annloader_utils.py`, `classifiers.py`, `run_classification.py`, `plot_overview.py`
- UMAP split visualization (uses `sjanpy.pl` if needed)

## Dependency Changes

`sjanpy` already depends on `scanpy`, `anndata`, `h5py`, `scipy`, `numpy`, `pandas`, `scikit-learn`. No new dependencies needed.

## Backward Compatibility

- `sjanpy.ml.read_obs_h5py` and `sjanpy.ml.read_var_h5py` remain importable (re-exported from `h5ad_io`)
- `sjanpy.ml.build_dataset` continues to work unchanged; internal imports updated to use `h5ad_io`

## File Summary

| New file | LOC (est.) | From |
|---|---|---|
| `sjanpy/pp/split.py` | ~100 | `stratified_split.py` |
| `sjanpy/pp/hvg.py` | ~180 | `build_standardized_h5ad.py` |
| `sjanpy/ml/h5ad_io.py` | ~250 | `build_standardized_h5ad.py` + `build_dataset.py` |
| `sjanpy/ml/standardize.py` | ~450 | `build_standardized_h5ad.py` |

Total: ~980 LOC of new library code. scgraph's `build_standardized_h5ad.py` shrinks from ~1285 to ~200 lines.
