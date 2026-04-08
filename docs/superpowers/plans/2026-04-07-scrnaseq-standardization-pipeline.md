# scRNA-seq Standardization Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract reusable scRNA-seq preprocessing code from `genofoundation/data/scgraph/` into `sjanpy` as four new modules.

**Architecture:** Bottom-up build order — `h5ad_io` first (foundation), then `split` and `hvg` (independent of each other, `hvg` uses `h5ad_io`), then `standardize` (uses all three). Finally, refactor `build_dataset.py` to use `h5ad_io`, update `__init__.py` exports, and rewrite scgraph's script to use `sjanpy`.

**Tech Stack:** Python 3.10+, h5py, scipy.sparse, scanpy, anndata, numpy, pandas, scikit-learn

**Source files (read-only references):**
- `scgraph/stratified_split.py` → `/scratch/users/chensj16/projects/genofoundation/data/scgraph/stratified_split.py`
- `scgraph/build_standardized_h5ad.py` → `/scratch/users/chensj16/projects/genofoundation/data/scgraph/build_standardized_h5ad.py`
- `sjanpy/ml/build_dataset.py` → `/home/users/chensj16/s/projects/sjanpy/sjanpy/ml/build_dataset.py`

**Target package root:** `/home/users/chensj16/s/projects/sjanpy/`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `sjanpy/ml/h5ad_io.py` | Low-level h5ad reading via h5py: obs/var DataFrames, matrix access, validation |
| Create | `sjanpy/pp/split.py` | Stratified train/val/test splitting |
| Create | `sjanpy/pp/hvg.py` | HVG computation from specified cells (train-only) |
| Create | `sjanpy/ml/standardize.py` | Build standardized h5ad files (accumulate + streaming modes) |
| Create | `tests/conftest.py` | Shared pytest fixtures (synthetic h5ad files) |
| Create | `tests/test_h5ad_io.py` | Tests for h5ad_io |
| Create | `tests/test_split.py` | Tests for pp.split |
| Create | `tests/test_hvg.py` | Tests for pp.hvg |
| Create | `tests/test_standardize.py` | Tests for ml.standardize |
| Modify | `sjanpy/pp/__init__.py` | Add split, hvg imports |
| Modify | `sjanpy/ml/__init__.py` | Add h5ad_io, standardize imports; keep backward-compat re-exports |
| Modify | `sjanpy/ml/build_dataset.py` | Replace internal h5py readers with `h5ad_io` imports |

---

### Task 1: Test Fixtures — Synthetic h5ad Files

**Files:**
- Create: `tests/conftest.py`

These fixtures create minimal h5ad files in a temp directory that all test modules share. Two variants: one with `raw.X` (CSR sparse counts) and one with `X` only (dense).

- [ ] **Step 1: Create tests directory and conftest.py**

```python
"""Shared pytest fixtures for sjanpy tests."""

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse


@pytest.fixture(scope="session")
def tmp_h5ad_dir(tmp_path_factory):
    """Session-scoped temp directory with synthetic h5ad files."""
    d = tmp_path_factory.mktemp("h5ad_fixtures")

    # --- Fixture 1: sparse counts in raw.X, 200 cells, 50 genes, 3 batches, 4 cell types ---
    rng = np.random.default_rng(42)
    n_cells, n_genes = 200, 50
    X_counts = sparse.random(n_cells, n_genes, density=0.3, format="csr",
                              random_state=42, dtype=np.float32)
    X_counts.data = np.round(X_counts.data * 100).astype(np.float32)  # integer-like counts

    cell_types = np.array(["TypeA"] * 80 + ["TypeB"] * 60 + ["TypeC"] * 40 + ["TypeD"] * 20)
    batches = np.array(["B1"] * 70 + ["B2"] * 65 + ["B3"] * 65)
    tissues = np.array(["lung"] * 100 + ["heart"] * 100)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(cell_types),
        "batch": pd.Categorical(batches),
        "tissue": pd.Categorical(tissues),
        "extra_col": pd.Categorical(rng.choice(["X", "Y", "Z"], n_cells)),
    }, index=[f"cell_{i}" for i in range(n_cells)])

    gene_names = [f"Gene{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_names)

    # Create with raw layer
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.zeros((n_cells, n_genes), dtype=np.float32)),
        obs=obs, var=var,
    )
    adata.raw = ad.AnnData(X=X_counts, var=var)
    adata.obsm["X_umap"] = rng.standard_normal((n_cells, 2)).astype(np.float32)
    adata.write_h5ad(d / "sparse_rawX.h5ad")

    # --- Fixture 2: counts directly in X (dense), 100 cells, 30 genes ---
    n2, g2 = 100, 30
    X_dense = rng.poisson(5, size=(n2, g2)).astype(np.float32)
    obs2 = pd.DataFrame({
        "cell_type": pd.Categorical(["A"] * 50 + ["B"] * 30 + ["C"] * 20),
        "batch": pd.Categorical(["D1"] * 50 + ["D2"] * 50),
    }, index=[f"c_{i}" for i in range(n2)])
    var2 = pd.DataFrame(index=[f"G{i}" for i in range(g2)])
    adata2 = ad.AnnData(X=X_dense, obs=obs2, var=var2)
    adata2.write_h5ad(d / "dense_X.h5ad")

    # --- Fixture 3: tiny dataset for edge cases (10 cells, 5 genes, 1 rare type) ---
    n3, g3 = 10, 5
    X3 = sparse.random(n3, g3, density=0.5, format="csr", random_state=7, dtype=np.float32)
    X3.data = np.round(X3.data * 50).astype(np.float32)
    obs3 = pd.DataFrame({
        "cell_type": pd.Categorical(["Common"] * 9 + ["Rare"]),
        "batch": pd.Categorical(["B1"] * 10),
    }, index=[f"t_{i}" for i in range(n3)])
    var3 = pd.DataFrame(index=[f"g{i}" for i in range(g3)])
    adata3 = ad.AnnData(X=X3, obs=obs3, var=var3)
    adata3.raw = ad.AnnData(X=X3.copy(), var=var3)
    adata3.write_h5ad(d / "tiny.h5ad")

    return d
```

- [ ] **Step 2: Verify fixtures can be created**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -c "import tests.conftest; print('OK')" 2>&1 || python -c "import conftest; print('OK')"`

This will fail (no `__init__.py` in tests) — that's fine, we just need the file to exist for pytest discovery.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic h5ad fixtures for new modules"
```

---

### Task 2: `sjanpy/ml/h5ad_io.py` — Low-Level h5ad I/O

**Files:**
- Create: `sjanpy/ml/h5ad_io.py`
- Create: `tests/test_h5ad_io.py`

This is the foundation module. It consolidates h5py readers from both `build_dataset.py` and `build_standardized_h5ad.py`.

- [ ] **Step 1: Write tests for h5ad_io**

```python
"""Tests for sjanpy.ml.h5ad_io."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


class TestReadObs:
    def test_reads_all_columns(self, tmp_h5ad_dir):
        from sjanpy.ml.h5ad_io import read_obs
        obs = read_obs(tmp_h5ad_dir / "sparse_rawX.h5ad")
        assert isinstance(obs, pd.DataFrame)
        assert len(obs) == 200
        assert "cell_type" in obs.columns
        assert "batch" in obs.columns
        assert obs.index[0] == "cell_0"

    def test_reads_dense_file(self, tmp_h5ad_dir):
        from sjanpy.ml.h5ad_io import read_obs
        obs = read_obs(tmp_h5ad_dir / "dense_X.h5ad")
        assert len(obs) == 100
        assert set(obs["cell_type"].unique()) == {"A", "B", "C"}


class TestReadVar:
    def test_reads_gene_names(self, tmp_h5ad_dir):
        from sjanpy.ml.h5ad_io import read_var
        var = read_var(tmp_h5ad_dir / "sparse_rawX.h5ad")
        assert len(var) == 50
        assert var.index[0] == "Gene0"

    def test_reads_raw_var(self, tmp_h5ad_dir):
        from sjanpy.ml.h5ad_io import read_var
        var = read_var(tmp_h5ad_dir / "sparse_rawX.h5ad", group="raw/var")
        assert len(var) == 50


class TestLocateMatrix:
    def test_locate_raw_X(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, get_matrix_shape
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, var_grp, label = locate_matrix(f, "raw.X")
            assert label == "raw.X"
            shape = get_matrix_shape(mat)
            assert shape == (200, 50)

    def test_locate_X(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, get_matrix_shape
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            mat, var_grp, label = locate_matrix(f, "X")
            assert label == "X"
            shape = get_matrix_shape(mat)
            assert shape == (100, 30)

    def test_invalid_source_raises(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            with pytest.raises(ValueError, match="raw.X"):
                locate_matrix(f, "raw.X")


class TestReadMatrixRows:
    def test_read_sparse_rows(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, read_matrix_rows
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            rows = read_matrix_rows(mat, np.array([0, 5, 10]))
            assert isinstance(rows, sparse.csr_matrix)
            assert rows.shape == (3, 50)

    def test_read_dense_rows(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, read_matrix_rows
        with h5py.File(tmp_h5ad_dir / "dense_X.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "X")
            rows = read_matrix_rows(mat, np.array([0, 1, 2]))
            assert isinstance(rows, sparse.csr_matrix)
            assert rows.shape == (3, 30)


class TestReadSparseChunk:
    def test_contiguous_chunk(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, read_sparse_chunk
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            chunk = read_sparse_chunk(mat, start=0, end=50, n_vars=50)
            assert chunk.shape == (50, 50)
            assert isinstance(chunk, sparse.csr_matrix)


class TestValidateMatrixValues:
    def test_counts_pass(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, validate_matrix_values
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            # Should not raise
            validate_matrix_values(mat, "counts", strict=True)

    def test_wrong_type_strict_raises(self, tmp_h5ad_dir):
        import h5py
        from sjanpy.ml.h5ad_io import locate_matrix, validate_matrix_values
        with h5py.File(tmp_h5ad_dir / "sparse_rawX.h5ad", "r") as f:
            mat, _, _ = locate_matrix(f, "raw.X")
            with pytest.raises(ValueError):
                validate_matrix_values(mat, "normalized", strict=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_h5ad_io.py -v 2>&1 | head -30`

Expected: ImportError — `sjanpy.ml.h5ad_io` does not exist yet.

- [ ] **Step 3: Implement h5ad_io.py**

Create `sjanpy/ml/h5ad_io.py`. The implementation merges the best of both existing readers:
- From `build_dataset.py`: `_decode_stringlike`, `read_obs_h5py` (handles scalar categoricals, legacy `__categories` format, `encoding-type` attrs)
- From `build_standardized_h5ad.py`: `locate_matrix_source`, `get_matrix_shape`, `read_sparse_rows_from_group`, `read_matrix_rows`, `validate_matrix_values`, `_read_source_chunk`

```python
"""Low-level h5ad I/O via h5py.

Provides direct h5py-based readers for obs/var DataFrames and expression
matrices without loading the full AnnData object. Supports both modern
and legacy h5ad formats, sparse (CSR) and dense matrices.

This module consolidates h5py reading logic previously duplicated across
sjanpy.ml.build_dataset and genofoundation's build_standardized_h5ad.py.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# String decoding
# ---------------------------------------------------------------------------

def _decode_stringlike(values):
    """Decode h5py string/object arrays to Python str arrays."""
    arr = np.asarray(values)
    if not hasattr(arr, "dtype") or arr.dtype.kind not in ("S", "O", "U"):
        return values

    def _one(v):
        return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)

    if arr.ndim == 0:
        return _one(arr.item())
    return np.array([_one(v) for v in arr.tolist()], dtype=object).reshape(arr.shape)


# ---------------------------------------------------------------------------
# DataFrame readers
# ---------------------------------------------------------------------------

def _read_h5_group_to_dataframe(grp: h5py.Group) -> pd.DataFrame:
    """Read an h5py group (obs or var) into a pandas DataFrame.

    Handles modern h5ad format (encoding-type attrs), legacy format
    (__categories group), scalar categoricals, and byte-string decoding.
    """
    data_dict = {}
    index_key = grp.attrs.get("_index", "_index")

    # Legacy h5ad: __categories group holds category arrays
    if "__categories" in grp:
        index_key = grp.attrs.get("_index", "index")
        obs_index = grp[index_key][:]
        if obs_index.dtype.kind in ("S", "O"):
            obs_index = _decode_stringlike(obs_index)
        for key in grp.keys():
            if key in ("__categories", index_key):
                continue
            col_data = grp[key][:]
            if col_data.dtype.kind in ("S", "O"):
                col_data = _decode_stringlike(col_data)
            data_dict[key] = col_data
        df = pd.DataFrame(data_dict)
        df.index = obs_index
        return df

    # Modern h5ad format
    obs_index = None
    if index_key in grp:
        idx_ds = grp[index_key]
        obs_index = idx_ds[:]
        if hasattr(obs_index, "dtype") and obs_index.dtype.kind in ("S", "O"):
            obs_index = _decode_stringlike(obs_index)

    for key in grp.keys():
        if key == index_key:
            continue
        node = grp[key]
        enc_type = ""

        if isinstance(node, h5py.Group):
            enc_type = node.attrs.get("encoding-type", "")
            if enc_type == "categorical" or ("categories" in node and "codes" in node):
                codes = node["codes"][:]
                categories = node["categories"][:]
                if categories.dtype.kind in ("S", "O"):
                    categories = _decode_stringlike(categories)
                if categories.ndim == 0:
                    categories = np.array([str(categories)])
                data_dict[key] = pd.Categorical.from_codes(codes, categories=categories)
                continue
            # Skip unsupported groups
            continue

        # h5py.Dataset
        enc_type = node.attrs.get("encoding-type", "")
        if enc_type == "categorical":
            codes = node["codes"][:]
            categories = node["categories"][:]
            if categories.dtype.kind in ("S", "O"):
                categories = _decode_stringlike(categories)
            if categories.ndim == 0:
                categories = np.array([str(categories)])
            data_dict[key] = pd.Categorical.from_codes(codes, categories=categories)
        else:
            col_data = node[:]
            if col_data.dtype.kind in ("S", "O"):
                col_data = _decode_stringlike(col_data)
            data_dict[key] = col_data

    df = pd.DataFrame(data_dict)
    if obs_index is not None:
        df.index = obs_index
    return df


def read_obs(h5ad_path: str | Path) -> pd.DataFrame:
    """Read the obs DataFrame from an h5ad file via h5py.

    Handles categorical encoding, legacy formats, scalar categories,
    and byte-string decoding without loading the full AnnData object.
    """
    with h5py.File(h5ad_path, "r") as f:
        if "obs" not in f:
            raise ValueError(f"h5ad file missing 'obs' group: {h5ad_path}")
        return _read_h5_group_to_dataframe(f["obs"])


def read_var(h5ad_path: str | Path, group: str = "var") -> pd.DataFrame:
    """Read a var DataFrame from an h5ad file via h5py.

    Args:
        h5ad_path: Path to the h5ad file.
        group: HDF5 group path (e.g. 'var', 'raw/var').
    """
    with h5py.File(h5ad_path, "r") as f:
        parts = group.split("/")
        grp = f
        for p in parts:
            if p not in grp:
                raise ValueError(f"Group '{group}' not found in {h5ad_path}")
            grp = grp[p]
        return _read_h5_group_to_dataframe(grp)


# ---------------------------------------------------------------------------
# Matrix location and shape
# ---------------------------------------------------------------------------

def locate_matrix(
    f: h5py.File, source: str
) -> tuple[object, h5py.Group, str]:
    """Resolve a matrix source string to the h5py object, var group, and label.

    Args:
        f: Open h5py.File handle.
        source: One of 'raw.X', 'X', or 'layers/<name>'.

    Returns:
        (matrix_obj, var_group, resolved_label)
    """
    if source == "raw.X":
        if "raw" not in f or "X" not in f["raw"]:
            raise ValueError("matrix source 'raw.X' requested but raw/X not found in file")
        var_grp = f["raw"]["var"] if ("raw" in f and "var" in f["raw"]) else f["var"]
        return f["raw"]["X"], var_grp, "raw.X"

    if source == "X":
        if "X" not in f:
            raise ValueError("matrix source 'X' requested but X not found in file")
        return f["X"], f["var"], "X"

    if source.startswith("layers/"):
        layer_name = source.split("/", 1)[1]
        if "layers" not in f or layer_name not in f["layers"]:
            raise ValueError(f"matrix source '{source}' requested but layer not found")
        return f["layers"][layer_name], f["var"], source

    raise ValueError(
        f"Unsupported matrix source '{source}'. Use: raw.X, X, or layers/<name>."
    )


def get_matrix_shape(matrix_obj: object) -> tuple[int, int]:
    """Get (n_obs, n_vars) from a sparse group or dense dataset."""
    if isinstance(matrix_obj, h5py.Group) and "shape" in matrix_obj.attrs:
        shp = matrix_obj.attrs["shape"]
        return int(shp[0]), int(shp[1])
    if isinstance(matrix_obj, h5py.Dataset):
        if matrix_obj.ndim != 2:
            raise ValueError(f"Dense matrix has unexpected ndim={matrix_obj.ndim}")
        return int(matrix_obj.shape[0]), int(matrix_obj.shape[1])
    raise ValueError("Cannot determine matrix shape")


# ---------------------------------------------------------------------------
# Matrix reading
# ---------------------------------------------------------------------------

def read_matrix_rows(
    matrix_obj: object, row_indices: np.ndarray
) -> sparse.csr_matrix:
    """Read specified rows from a matrix as a CSR sparse matrix.

    Works for both sparse (CSR h5py group) and dense (h5py dataset) matrices.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)

    if isinstance(matrix_obj, h5py.Group) and "data" in matrix_obj:
        return _read_sparse_rows(matrix_obj, row_indices)

    if isinstance(matrix_obj, h5py.Dataset):
        block = matrix_obj[row_indices, :]
        return sparse.csr_matrix(block)

    raise TypeError(f"Unsupported matrix object type: {type(matrix_obj)}")


def _read_sparse_rows(
    grp: h5py.Group, row_indices: np.ndarray
) -> sparse.csr_matrix:
    """Read arbitrary rows from a CSR sparse h5py group."""
    if row_indices.size == 0:
        shape = grp.attrs.get("shape", (0, 0))
        return sparse.csr_matrix((0, int(shape[1])))

    data_arr = grp["data"]
    indices_arr = grp["indices"]
    indptr_arr = grp["indptr"][:]
    n_vars = (
        int(grp.attrs["shape"][1])
        if "shape" in grp.attrs
        else int(indices_arr[:].max()) + 1
    )

    starts = indptr_arr[row_indices]
    ends = indptr_arr[row_indices + 1]
    lengths = (ends - starts).astype(np.int64)
    total_nnz = int(lengths.sum())

    out_data = np.empty(total_nnz, dtype=data_arr.dtype)
    out_indices = np.empty(total_nnz, dtype=indices_arr.dtype)
    out_indptr = np.zeros(row_indices.size + 1, dtype=np.int64)

    cursor = 0
    for i, (s, e, ln) in enumerate(zip(starts, ends, lengths)):
        s, e, ln = int(s), int(e), int(ln)
        if ln > 0:
            out_data[cursor:cursor + ln] = data_arr[s:e]
            out_indices[cursor:cursor + ln] = indices_arr[s:e]
        cursor += ln
        out_indptr[i + 1] = cursor

    return sparse.csr_matrix(
        (out_data, out_indices, out_indptr), shape=(row_indices.size, n_vars)
    )


def read_sparse_chunk(
    matrix_obj: object, start: int, end: int, n_vars: int
) -> sparse.csr_matrix:
    """Read a contiguous row range from a matrix as CSR.

    Optimized for sequential chunk reading (uses indptr slicing for sparse).
    Falls back to dense→CSR conversion for dense datasets.
    """
    if isinstance(matrix_obj, h5py.Group) and "data" in matrix_obj:
        indptr = matrix_obj["indptr"][start:end + 1]
        d_start = int(indptr[0])
        d_end = int(indptr[-1])
        return sparse.csr_matrix(
            (matrix_obj["data"][d_start:d_end],
             matrix_obj["indices"][d_start:d_end],
             indptr - d_start),
            shape=(end - start, n_vars),
        )

    if isinstance(matrix_obj, h5py.Dataset):
        return sparse.csr_matrix(matrix_obj[start:end, :])

    raise TypeError(f"Unsupported matrix object type: {type(matrix_obj)}")


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------

def _sample_matrix_values(matrix_obj: object, sample_n: int = 200_000) -> np.ndarray:
    """Sample non-structural values from a matrix for type checking."""
    if isinstance(matrix_obj, h5py.Group) and "data" in matrix_obj:
        data = matrix_obj["data"]
        n = min(sample_n, int(data.shape[0]))
        return np.asarray(data[:n], dtype=np.float64)

    if isinstance(matrix_obj, h5py.Dataset):
        if matrix_obj.ndim == 2:
            r = min(512, int(matrix_obj.shape[0]))
            c = min(512, int(matrix_obj.shape[1]))
            return np.asarray(matrix_obj[:r, :c], dtype=np.float64).reshape(-1)
        if matrix_obj.ndim == 1:
            n = min(sample_n, int(matrix_obj.shape[0]))
            return np.asarray(matrix_obj[:n], dtype=np.float64)

    return np.array([], dtype=np.float64)


def validate_matrix_values(
    matrix_obj: object,
    expected_type: str,
    sample_n: int = 200_000,
    strict: bool = False,
) -> None:
    """Check sampled matrix values are consistent with declared type.

    Args:
        matrix_obj: h5py matrix object (Group for sparse, Dataset for dense).
        expected_type: 'counts' or 'normalized'.
        sample_n: Number of values to sample.
        strict: If True, raise ValueError on mismatch; else print warning.
    """
    vals = _sample_matrix_values(matrix_obj, sample_n=sample_n)
    if vals.size == 0:
        msg = "Unable to sample matrix values for validation."
        if strict:
            raise ValueError(msg)
        return

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        msg = "Sampled matrix values are all non-finite."
        if strict:
            raise ValueError(msg)
        return

    vmin, vmax = float(vals.min()), float(vals.max())
    int_ratio = float(np.mean(np.isclose(vals, np.round(vals), atol=1e-6)))

    if expected_type == "counts":
        ok = (vmin >= -1e-8) and (int_ratio >= 0.999)
        msg = (
            "Matrix declared as counts but values are not mostly integer-like "
            f"or contain negatives (min={vmin:.4g}, int_ratio={int_ratio:.4f})."
        )
    elif expected_type == "normalized":
        ok = (vmin >= -1e-8) and (vmax <= 20.0) and (int_ratio <= 0.2)
        msg = (
            "Matrix declared as normalized but values look suspicious "
            f"(min={vmin:.4g}, max={vmax:.4g}, int_ratio={int_ratio:.4f})."
        )
    else:
        raise ValueError("expected_type must be 'counts' or 'normalized'")

    if not ok:
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_h5ad_io.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sjanpy/ml/h5ad_io.py tests/conftest.py tests/test_h5ad_io.py
git commit -m "feat(ml): add h5ad_io module — consolidated h5py readers for obs, var, and matrix access"
```

---

### Task 3: `sjanpy/pp/split.py` — Stratified Splitting

**Files:**
- Create: `sjanpy/pp/split.py`
- Create: `tests/test_split.py`

- [ ] **Step 1: Write tests for split**

```python
"""Tests for sjanpy.pp.split."""

import numpy as np
import pandas as pd
import pytest


def _make_obs(n=200, n_types=4):
    """Helper: create a simple obs DataFrame."""
    types = []
    per = n // n_types
    for i in range(n_types):
        count = per if i < n_types - 1 else n - len(types)
        types.extend([f"Type{i}"] * count)
    return pd.DataFrame({"cell_type": types}, index=[f"c{i}" for i in range(n)])


class TestStratifiedSplit:
    def test_returns_correct_columns(self):
        from sjanpy.pp.split import stratified_split
        obs = _make_obs(200, 4)
        result = stratified_split(obs, "cell_type", val_ratio=0.1, test_ratio=0.1)
        assert "cell_index" in result.columns
        assert "split" in result.columns
        assert len(result) == 200

    def test_split_ratios_approximate(self):
        from sjanpy.pp.split import stratified_split
        obs = _make_obs(1000, 5)
        result = stratified_split(obs, "cell_type", val_ratio=0.05, test_ratio=0.05)
        counts = result["split"].value_counts()
        assert counts["train"] == pytest.approx(900, abs=20)
        assert counts["val"] == pytest.approx(50, abs=15)
        assert counts["test"] == pytest.approx(50, abs=15)

    def test_all_indices_covered(self):
        from sjanpy.pp.split import stratified_split
        obs = _make_obs(100, 3)
        result = stratified_split(obs, "cell_type", val_ratio=0.1, test_ratio=0.1)
        assert set(result["cell_index"]) == set(range(100))

    def test_rare_types_go_to_train(self):
        from sjanpy.pp.split import stratified_split
        # 1 cell of RareType — must end up in train
        obs = pd.DataFrame({"ct": ["A"] * 50 + ["B"] * 49 + ["Rare"]},
                           index=[f"c{i}" for i in range(100)])
        result = stratified_split(obs, "ct", val_ratio=0.1, test_ratio=0.1)
        rare_idx = 99
        rare_split = result.loc[result["cell_index"] == rare_idx, "split"].values[0]
        assert rare_split == "train"

    def test_reproducible_with_seed(self):
        from sjanpy.pp.split import stratified_split
        obs = _make_obs(200, 4)
        r1 = stratified_split(obs, "cell_type", seed=42)
        r2 = stratified_split(obs, "cell_type", seed=42)
        assert (r1["split"].values == r2["split"].values).all()

    def test_all_rare_types(self):
        from sjanpy.pp.split import stratified_split
        # Every type has only 1 cell — all go to train
        obs = pd.DataFrame({"ct": [f"T{i}" for i in range(10)]},
                           index=[f"c{i}" for i in range(10)])
        result = stratified_split(obs, "ct")
        assert (result["split"] == "train").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_split.py -v 2>&1 | head -20`

Expected: ImportError — `sjanpy.pp.split` does not exist.

- [ ] **Step 3: Implement split.py**

```python
"""Stratified train/val/test splitting for single-cell datasets.

Performs stratified sampling by a categorical column (typically cell type),
producing a DataFrame that records which cells belong to train, val, or test.
Rare categories with fewer than 2 cells are always placed in the train set.
"""

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
    """Stratified train/val/test split.

    Two-stage stratified split:
      1. Split into train vs held-out, stratified by stratify_col
      2. Split held-out into val vs test, stratified by stratify_col

    Cells whose category has fewer than 2 samples are always placed
    in train (cannot be stratified). In the held-out split, categories
    with count < 2 go to val.

    Args:
        obs: DataFrame with at least ``stratify_col``.
        stratify_col: Column name for stratification (e.g. cell type).
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``cell_index`` (int) and ``split``
        (one of 'train', 'val', 'test').
    """
    labels = obs[stratify_col].astype(str)
    indices = np.arange(len(obs))

    # Rare categories (<2 cells) cannot be stratified
    counts = labels.value_counts()
    rare_mask = labels.isin(counts[counts < 2].index).values

    rare_indices = indices[rare_mask]
    split_indices = indices[~rare_mask]
    split_labels = labels.iloc[split_indices]

    held_out_ratio = val_ratio + test_ratio

    if len(split_indices) == 0:
        # All cells are rare — everything goes to train
        split_col = np.full(len(obs), "train", dtype=object)
        return pd.DataFrame({"cell_index": indices, "split": split_col})

    # Stage 1: train vs held-out
    train_idx, held_out_idx = train_test_split(
        split_indices,
        test_size=held_out_ratio,
        random_state=seed,
        stratify=split_labels,
    )
    train_idx = np.concatenate([train_idx, rare_indices])

    # Stage 2: val vs test within held-out
    held_out_labels = labels.iloc[held_out_idx]
    held_out_counts = held_out_labels.value_counts()
    held_out_rare_mask = held_out_labels.isin(
        held_out_counts[held_out_counts < 2].index
    ).values

    held_out_rare = held_out_idx[held_out_rare_mask]
    held_out_splittable = held_out_idx[~held_out_rare_mask]
    held_out_splittable_labels = labels.iloc[held_out_splittable]

    test_fraction = test_ratio / held_out_ratio

    if len(held_out_splittable) == 0:
        val_idx = held_out_idx
        test_idx = np.array([], dtype=int)
    else:
        val_idx_split, test_idx = train_test_split(
            held_out_splittable,
            test_size=test_fraction,
            random_state=seed,
            stratify=held_out_splittable_labels,
        )
        val_idx = np.concatenate([val_idx_split, held_out_rare])

    split_col = np.full(len(obs), "train", dtype=object)
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    return pd.DataFrame({"cell_index": indices, "split": split_col})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_split.py -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sjanpy/pp/split.py tests/test_split.py
git commit -m "feat(pp): add stratified_split for train/val/test splitting"
```

---

### Task 4: `sjanpy/pp/hvg.py` — Train-Only HVG Computation

**Files:**
- Create: `sjanpy/pp/hvg.py`
- Create: `tests/test_hvg.py`

- [ ] **Step 1: Write tests for hvg**

```python
"""Tests for sjanpy.pp.hvg."""

import math
import numpy as np
import pandas as pd
import pytest


class TestPrepareHvgSample:
    def test_returns_none_when_small(self):
        from sjanpy.pp.hvg import prepare_hvg_sample
        obs = pd.DataFrame({"ct": ["A"] * 50 + ["B"] * 50})
        train_indices = np.arange(100)
        result = prepare_hvg_sample(obs, train_indices, "ct", target_size=200)
        assert result is None  # 100 < 200, no sampling needed

    def test_returns_subset_when_large(self):
        from sjanpy.pp.hvg import prepare_hvg_sample
        obs = pd.DataFrame({"ct": ["A"] * 500 + ["B"] * 500})
        train_indices = np.arange(1000)
        result = prepare_hvg_sample(obs, train_indices, "ct", target_size=200)
        assert result is not None
        assert len(result) == 200
        assert all(r in train_indices for r in result)

    def test_preserves_small_types(self):
        from sjanpy.pp.hvg import prepare_hvg_sample
        # 5 cells of type Rare (below min_cells=10) + 500 of type Big
        obs = pd.DataFrame({"ct": ["Rare"] * 5 + ["Big"] * 500})
        train_indices = np.arange(505)
        result = prepare_hvg_sample(obs, train_indices, "ct",
                                     target_size=100, min_cells=10)
        assert result is not None
        # All 5 rare cells must be included
        rare_in_result = set(range(5)) & set(result)
        assert len(rare_in_result) == 5

    def test_reproducible(self):
        from sjanpy.pp.hvg import prepare_hvg_sample
        obs = pd.DataFrame({"ct": ["A"] * 300 + ["B"] * 300})
        idx = np.arange(600)
        r1 = prepare_hvg_sample(obs, idx, "ct", target_size=100, seed=42)
        r2 = prepare_hvg_sample(obs, idx, "ct", target_size=100, seed=42)
        np.testing.assert_array_equal(r1, r2)


class TestComputeHvg:
    def test_returns_gene_list_and_mask(self, tmp_h5ad_dir):
        from sjanpy.pp.hvg import compute_hvg
        path = tmp_h5ad_dir / "tiny.h5ad"
        # Use all 10 cells, single batch — HVG from raw.X
        genes, mask = compute_hvg(
            h5ad_path=path,
            matrix_source="raw.X",
            cell_indices=np.arange(10),
            batch_key="batch",
        )
        assert isinstance(genes, list)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == 5  # 5 genes in fixture
        assert len(genes) == mask.sum()

    def test_from_X_source(self, tmp_h5ad_dir):
        from sjanpy.pp.hvg import compute_hvg
        path = tmp_h5ad_dir / "dense_X.h5ad"
        genes, mask = compute_hvg(
            h5ad_path=path,
            matrix_source="X",
            cell_indices=np.arange(100),
            batch_key="batch",
        )
        assert len(mask) == 30
        assert all(isinstance(g, str) for g in genes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_hvg.py -v 2>&1 | head -20`

Expected: ImportError.

- [ ] **Step 3: Implement hvg.py**

```python
"""Train-only highly variable gene (HVG) computation.

Provides functions to subsample training cells (preserving rare types)
and compute HVGs using scanpy's seurat method with batch correction.
The caller decides which cells to use — these functions have no knowledge
of train/val/test splits.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from ..ml.h5ad_io import (
    locate_matrix,
    get_matrix_shape,
    read_matrix_rows,
    read_var,
    validate_matrix_values,
)


def prepare_hvg_sample(
    obs: pd.DataFrame,
    train_indices: np.ndarray,
    stratify_col: str,
    target_size: int = 300_000,
    min_cells: int = 100,
    seed: int = 42,
) -> np.ndarray | None:
    """Stratified subsample of cells for HVG computation.

    If ``len(train_indices) <= target_size``, returns None (use all cells).
    Otherwise, returns a sorted array of global cell indices, preserving
    all cells from small categories (< min_cells) and proportionally
    sampling from larger categories.

    Args:
        obs: Full obs DataFrame (indexed by global cell position).
        train_indices: Global indices of training cells.
        stratify_col: Column in obs to stratify by.
        target_size: Target number of cells to sample.
        min_cells: Categories with this many cells or fewer are kept entirely.
        seed: Random seed.

    Returns:
        Sorted array of global indices, or None if no sampling needed.
    """
    n_train = len(train_indices)
    if n_train <= target_size:
        return None

    rng = np.random.default_rng(seed)
    obs_train = obs.iloc[train_indices]
    cell_types = obs_train[stratify_col].astype(str).fillna("NA")

    by_type: dict[str, list[int]] = {}
    for local_idx, ct in enumerate(cell_types.values):
        by_type.setdefault(ct, []).append(local_idx)

    small_kept = []
    large_groups: dict[str, np.ndarray] = {}
    for ct, local_indices in by_type.items():
        if len(local_indices) <= min_cells:
            small_kept.extend(local_indices)
        else:
            large_groups[ct] = np.asarray(local_indices, dtype=np.int64)

    fixed_n = len(small_kept)
    if fixed_n >= target_size:
        sampled_local = np.asarray(small_kept, dtype=np.int64)
        return np.sort(train_indices[sampled_local])

    remain_budget = target_size - fixed_n
    large_total = sum(len(v) for v in large_groups.values())
    alloc: dict[str, int] = {}

    if large_total > 0 and remain_budget > 0:
        frac_parts = []
        total_assigned = 0
        for ct, arr in large_groups.items():
            ideal = remain_budget * (len(arr) / large_total)
            take = min(int(math.floor(ideal)), len(arr))
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
        if n_take > 0:
            chosen = rng.choice(arr, size=n_take, replace=False)
            sampled_local.extend(chosen.tolist())

    sampled_local = np.asarray(sampled_local, dtype=np.int64)
    return np.sort(train_indices[sampled_local])


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
    """Compute highly variable genes from specified cells.

    Reads the matrix rows for ``cell_indices``, optionally normalizes
    (if counts), then runs scanpy's HVG detection with batch correction.

    Args:
        h5ad_path: Path to source h5ad file.
        matrix_source: Matrix location ('raw.X', 'X', or 'layers/<name>').
        cell_indices: Global row indices of cells to use.
        batch_key: obs column for batch correction in HVG computation.
        matrix_value_type: 'counts' or 'normalized'.
        min_mean, max_mean, min_disp: Scanpy HVG parameters.

    Returns:
        (hvg_gene_names, hvg_boolean_mask) where mask has length n_total_genes.
    """
    import h5py
    from ..ml.h5ad_io import _read_h5_group_to_dataframe

    h5ad_path = Path(h5ad_path)

    with h5py.File(h5ad_path, "r") as f:
        matrix_obj, var_grp, _ = locate_matrix(f, matrix_source)
        X = read_matrix_rows(matrix_obj, np.asarray(cell_indices, dtype=np.int64))
        var = _read_h5_group_to_dataframe(var_grp)

    # Read obs for batch_key
    from ..ml.h5ad_io import read_obs
    obs = read_obs(h5ad_path)
    obs_subset = obs.iloc[cell_indices].copy()

    # Ensure gene names are strings and unique
    gene_names = [str(x) for x in var.index]
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for n in gene_names:
        if n not in seen:
            seen[n] = 0
            unique_names.append(n)
        else:
            seen[n] += 1
            unique_names.append(f"{n}-{seen[n]}")
    var.index = unique_names

    adata = sc.AnnData(X=X, obs=obs_subset, var=var)

    if matrix_value_type == "counts":
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

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

    del adata, X
    gc.collect()

    return hvg_genes, hvg_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_hvg.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sjanpy/pp/hvg.py tests/test_hvg.py
git commit -m "feat(pp): add HVG computation with stratified sampling (train-only, no leakage)"
```

---

### Task 5: `sjanpy/ml/standardize.py` — Standardized h5ad Builder

**Files:**
- Create: `sjanpy/ml/standardize.py`
- Create: `tests/test_standardize.py`

This is the largest module. It contains the single-pass accumulate writer and the streaming writer.

- [ ] **Step 1: Write tests for standardize**

```python
"""Tests for sjanpy.ml.standardize."""

import json
import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse


class TestBuildStandardizedObs:
    def test_standard_columns(self):
        from sjanpy.ml.standardize import build_standardized_obs
        obs = pd.DataFrame({
            "my_ct": ["A", "B", "A"],
            "my_batch": ["X", "X", "Y"],
            "tissue": ["lung", "lung", "heart"],
        }, index=["c0", "c1", "c2"])
        result = build_standardized_obs(
            obs, np.array([0, 2]), "my_ct", "my_batch", "test_ds",
            np.array([1000.0, 2000.0]),
        )
        assert list(result.columns[:5]) == ["cell_type", "batch", "tissue", "dataset", "library_size"]
        assert result["cell_type"].tolist() == ["A", "A"]
        assert result["batch"].tolist() == ["X", "Y"]
        assert result["dataset"].iloc[0] == "test_ds"
        assert result["library_size"].tolist() == [1000.0, 2000.0]

    def test_extra_columns(self):
        from sjanpy.ml.standardize import build_standardized_obs
        obs = pd.DataFrame({
            "ct": ["A", "B"],
            "bt": ["X", "Y"],
            "fine_ct": ["A1", "B2"],
        }, index=["c0", "c1"])
        result = build_standardized_obs(
            obs, np.array([0, 1]), "ct", "bt", "ds",
            np.array([100.0, 200.0]),
            extra_columns={"fine_ct": "cell_type_fine"},
        )
        assert "cell_type_fine" in result.columns
        assert result["cell_type_fine"].tolist() == ["A1", "B2"]

    def test_missing_tissue_uses_dataset_name(self):
        from sjanpy.ml.standardize import build_standardized_obs
        obs = pd.DataFrame({"ct": ["A"], "bt": ["X"]}, index=["c0"])
        result = build_standardized_obs(
            obs, np.array([0]), "ct", "bt", "my_ds",
            np.array([500.0]),
        )
        assert result["tissue"].iloc[0] == "my_ds"


class TestBuildStandardizedH5ads:
    def test_creates_split_files(self, tmp_h5ad_dir, tmp_path):
        from sjanpy.ml.standardize import build_standardized_h5ads
        from sjanpy.ml.h5ad_io import read_obs, read_var

        h5ad_path = tmp_h5ad_dir / "sparse_rawX.h5ad"
        obs = read_obs(h5ad_path)
        var = read_var(h5ad_path, group="raw/var")

        # Create a split: 160 train, 20 val, 20 test
        split_col = np.array(["train"] * 160 + ["val"] * 20 + ["test"] * 20)
        hvg_mask = np.ones(len(var), dtype=bool)  # all genes are HVG

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        stats = build_standardized_h5ads(
            h5ad_path=h5ad_path,
            output_dir=output_dir,
            split_col=split_col,
            hvg_mask=hvg_mask,
            all_var=var,
            obs=obs,
            cell_type_col="cell_type",
            batch_key="batch",
            dataset_name="test",
            matrix_source="raw.X",
            chunk_size=64,
        )

        # Check files exist
        assert (output_dir / "train.h5ad").exists()
        assert (output_dir / "val.h5ad").exists()
        assert (output_dir / "test.h5ad").exists()

        # Check stats
        assert stats["train"]["n_cells"] == 160
        assert stats["val"]["n_cells"] == 20
        assert stats["test"]["n_cells"] == 20

        # Verify train h5ad contents
        adata = ad.read_h5ad(output_dir / "train.h5ad")
        assert adata.shape == (160, 50)
        assert "normalized" in adata.layers
        assert "cell_type" in adata.obs.columns
        assert "batch" in adata.obs.columns
        assert "library_size" in adata.obs.columns
        assert adata.X.dtype == np.float32

    def test_streaming_mode(self, tmp_h5ad_dir, tmp_path):
        from sjanpy.ml.standardize import build_standardized_h5ads
        from sjanpy.ml.h5ad_io import read_obs, read_var

        h5ad_path = tmp_h5ad_dir / "sparse_rawX.h5ad"
        obs = read_obs(h5ad_path)
        var = read_var(h5ad_path, group="raw/var")

        split_col = np.array(["train"] * 160 + ["val"] * 20 + ["test"] * 20)
        hvg_mask = np.ones(len(var), dtype=bool)

        output_dir = tmp_path / "stream_out"
        output_dir.mkdir()

        stats = build_standardized_h5ads(
            h5ad_path=h5ad_path,
            output_dir=output_dir,
            split_col=split_col,
            hvg_mask=hvg_mask,
            all_var=var,
            obs=obs,
            cell_type_col="cell_type",
            batch_key="batch",
            dataset_name="test",
            matrix_source="raw.X",
            chunk_size=50,
            streaming=True,
        )

        adata = ad.read_h5ad(output_dir / "train.h5ad")
        assert adata.shape[0] == 160
        assert "normalized" in adata.layers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_standardize.py -v 2>&1 | head -20`

Expected: ImportError.

- [ ] **Step 3: Implement standardize.py**

```python
"""Standardized h5ad dataset builder.

Reads a source h5ad file, splits cells by a pre-computed split assignment,
applies normalization, and writes per-split h5ad files with a consistent
schema (raw counts in .X, log-normalized in .layers['normalized'],
standardized obs columns).

Supports two write strategies:
- Accumulate mode (default): single pass, accumulates per-split chunks in
  memory, then writes. Good for datasets up to ~1M cells.
- Streaming mode: one pass per split, writes CSR components directly to h5py
  without holding the full matrix. For very large datasets (>1M cells).
"""

from __future__ import annotations

import gc
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .h5ad_io import (
    locate_matrix,
    get_matrix_shape,
    read_sparse_chunk,
)


# ---------------------------------------------------------------------------
# Obs builder
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
    """Build a standardized obs DataFrame for one split.

    Always produces columns: cell_type, batch, tissue, dataset, library_size.
    Additional columns can be carried over via extra_columns mapping.

    Args:
        obs: Full source obs DataFrame.
        cell_indices: Row indices into obs for this split.
        cell_type_col: Source column for cell type.
        batch_key: Source column for batch.
        dataset_name: Name to fill in the 'dataset' column.
        library_size: Per-cell total UMI counts (pre-HVG).
        extra_columns: Mapping of source_col -> dest_col for additional columns.
    """
    obs_split = obs.iloc[cell_indices]
    std = pd.DataFrame(index=obs_split.index)
    std["cell_type"] = obs_split[cell_type_col].astype(str).values
    std["batch"] = obs_split[batch_key].astype(str).values
    if "tissue" in obs.columns:
        std["tissue"] = obs_split["tissue"].astype(str).values
    else:
        std["tissue"] = dataset_name
    std["dataset"] = dataset_name
    std["library_size"] = library_size

    if extra_columns:
        for src_col, dst_col in extra_columns.items():
            if src_col in obs_split.columns:
                std[dst_col] = obs_split[src_col].astype(str).values
                std[dst_col] = std[dst_col].astype("category")

    for col in ["cell_type", "batch", "tissue", "dataset"]:
        std[col] = std[col].astype("category")
    return std


# ---------------------------------------------------------------------------
# Var builder
# ---------------------------------------------------------------------------

def _build_var(all_var: pd.DataFrame, hvg_mask: np.ndarray) -> pd.DataFrame:
    """Build var DataFrame with HVG annotation."""
    var = all_var.copy()
    var["highly_variable"] = hvg_mask
    if "feature_name" in var.columns:
        symbols = var["feature_name"].values
        valid = [s for s in symbols if s and str(s) not in ("nan", "")]
        if len(valid) > len(symbols) * 0.5:
            var.index = pd.Index(symbols)
            var.index.name = None
    return var


# ---------------------------------------------------------------------------
# Accumulate-mode writer
# ---------------------------------------------------------------------------

def _write_accumulate(
    h5ad_path: Path,
    matrix_source: str,
    hvg_mask: np.ndarray,
    all_var: pd.DataFrame,
    obs: pd.DataFrame,
    split_col: np.ndarray,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    output_dir: Path,
    chunk_size: int,
    target_sum: float,
    extra_columns: dict[str, str] | None,
) -> dict:
    """Single-pass accumulate writer. Reads all chunks, splits into
    per-split accumulators, then writes h5ad files."""

    accum = {
        name: {"X_chunks": [], "lib_sizes": [], "obs_indices": []}
        for name in ["train", "val", "test"]
    }

    # Read obsm upfront
    obsm_arrays = {}
    with h5py.File(h5ad_path, "r") as f:
        if "obsm" in f:
            for key in f["obsm"]:
                obsm_arrays[key] = f["obsm"][key][:]

    with h5py.File(h5ad_path, "r") as f:
        matrix_obj, _, _ = locate_matrix(f, matrix_source)
        n_obs, n_vars = get_matrix_shape(matrix_obj)
        n_chunks = (n_obs + chunk_size - 1) // chunk_size

        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n_obs)

            chunk = read_sparse_chunk(matrix_obj, start, end, n_vars)
            lib_size = np.asarray(chunk.sum(axis=1)).flatten().astype(np.float32)

            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            chunk_splits = split_col[start:end]
            for name in ["train", "val", "test"]:
                mask = chunk_splits == name
                if mask.sum() == 0:
                    continue
                accum[name]["X_chunks"].append(chunk[mask])
                accum[name]["lib_sizes"].append(lib_size[mask])
                accum[name]["obs_indices"].append(np.arange(start, end)[mask])

            del chunk, lib_size
            gc.collect()

    var = _build_var(all_var, hvg_mask)
    all_stats = {}

    for name in ["train", "val", "test"]:
        a = accum[name]
        if not a["X_chunks"]:
            continue

        X = sparse.vstack(a["X_chunks"], format="csr")
        library_size = np.concatenate(a["lib_sizes"])
        obs_indices = np.concatenate(a["obs_indices"])
        del a["X_chunks"], a["lib_sizes"], a["obs_indices"]
        gc.collect()

        # Normalize
        row_sums = np.asarray(X.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        scale = sparse.diags((target_sum / row_sums).astype(np.float32))
        X_norm = (scale @ X).tocsr()
        X_norm.data = np.log1p(X_norm.data).astype(np.float32)

        std_obs = build_standardized_obs(
            obs, obs_indices, cell_type_col, batch_key,
            dataset_name, library_size, extra_columns,
        )
        split_obsm = {k: v[obs_indices] for k, v in obsm_arrays.items()}

        adata = ad.AnnData(
            X=X, obs=std_obs, var=var.copy(),
            layers={"normalized": X_norm}, obsm=split_obsm,
        )
        output_path = output_dir / f"{name}.h5ad"
        adata.write_h5ad(output_path, compression=None)

        file_mb = output_path.stat().st_size / 1e6
        all_stats[name] = {
            "n_cells": int(X.shape[0]),
            "nnz_counts": int(X.nnz),
            "nnz_normalized": int(X_norm.nnz),
            "library_size_mean": float(library_size.mean()),
            "library_size_median": float(np.median(library_size)),
            "file_size_mb": round(file_mb, 1),
        }
        del adata, X, X_norm, library_size, obs_indices
        gc.collect()

    return all_stats


# ---------------------------------------------------------------------------
# Streaming-mode writer (for very large datasets)
# ---------------------------------------------------------------------------

def _write_csr_to_h5(grp, csr_mat):
    """Write a CSR sparse matrix into an h5py group in anndata format."""
    grp.create_dataset("data", data=csr_mat.data)
    grp.create_dataset("indices", data=csr_mat.indices)
    grp.create_dataset("indptr", data=csr_mat.indptr)
    grp.attrs["encoding-type"] = "csr_matrix"
    grp.attrs["encoding-version"] = "0.1.0"
    grp.attrs["shape"] = list(csr_mat.shape)


def _write_obs_to_h5(grp, std_obs):
    """Write obs DataFrame into an h5py group in anndata format."""
    grp.attrs["encoding-type"] = "dataframe"
    grp.attrs["encoding-version"] = "0.2.0"
    grp.attrs["_index"] = "index"
    grp.attrs["column-order"] = list(std_obs.columns)

    idx = std_obs.index.astype(str).values
    grp.create_dataset("index", data=idx.astype("S"))

    for col in std_obs.columns:
        vals = std_obs[col]
        if hasattr(vals, "cat"):
            cat_grp = grp.create_group(col)
            cats = vals.cat.categories.values.astype(str)
            codes = vals.cat.codes.values.astype(np.int8)
            cat_grp.create_dataset("categories", data=cats.astype("S"))
            cat_grp.create_dataset("codes", data=codes)
            cat_grp.attrs["encoding-type"] = "categorical"
            cat_grp.attrs["encoding-version"] = "0.2.0"
            cat_grp.attrs["ordered"] = False
        else:
            grp.create_dataset(col, data=vals.values)


def _write_var_to_h5(grp, var):
    """Write var DataFrame into an h5py group in anndata format."""
    grp.attrs["encoding-type"] = "dataframe"
    grp.attrs["encoding-version"] = "0.2.0"
    grp.attrs["_index"] = "_index"
    grp.attrs["column-order"] = list(var.columns)

    idx = var.index.astype(str).values
    grp.create_dataset("_index", data=idx.astype("S"))

    for col in var.columns:
        vals = var[col].values
        if vals.dtype == bool:
            grp.create_dataset(col, data=vals)
        elif vals.dtype.kind in ("U", "O"):
            grp.create_dataset(col, data=np.array(vals, dtype="S"))
        else:
            grp.create_dataset(col, data=vals)


def _write_streaming(
    h5ad_path: Path,
    matrix_source: str,
    hvg_mask: np.ndarray,
    all_var: pd.DataFrame,
    obs: pd.DataFrame,
    split_col: np.ndarray,
    cell_type_col: str,
    batch_key: str,
    dataset_name: str,
    output_dir: Path,
    chunk_size: int,
    target_sum: float,
    extra_columns: dict[str, str] | None,
) -> dict:
    """Streaming writer for very large datasets. One pass per split,
    writes CSR components directly to h5py chunk by chunk."""

    var = _build_var(all_var, hvg_mask)

    obsm_arrays = {}
    with h5py.File(h5ad_path, "r") as f:
        if "obsm" in f:
            for key in f["obsm"]:
                obsm_arrays[key] = f["obsm"][key][:]

    all_stats = {}

    for name in ["train", "val", "test"]:
        split_mask_full = split_col == name
        n_cells = int(split_mask_full.sum())
        if n_cells == 0:
            continue

        output_path = output_dir / f"{name}.h5ad"
        obs_indices = np.where(split_mask_full)[0]

        with h5py.File(h5ad_path, "r") as src_f:
            matrix_obj, _, _ = locate_matrix(src_f, matrix_source)
            n_obs, n_vars = get_matrix_shape(matrix_obj)
            n_chunks = (n_obs + chunk_size - 1) // chunk_size

            with h5py.File(output_path, "w") as out_f:
                x_grp = out_f.create_group("X")
                x_data = x_grp.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32)
                x_indices = x_grp.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32)
                x_indptr_list = [np.int64(0)]

                norm_grp = out_f.create_group("layers").create_group("normalized")
                n_data = norm_grp.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32)
                n_indices = norm_grp.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32)
                n_indptr_list = [np.int64(0)]

                lib_sizes = []
                total_nnz_x = 0
                total_nnz_n = 0

                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, n_obs)

                    chunk_split_mask = split_mask_full[start:end]
                    n_split = chunk_split_mask.sum()
                    if n_split == 0:
                        continue

                    full_chunk = read_sparse_chunk(matrix_obj, start, end, n_vars)
                    lib_size = np.asarray(full_chunk.sum(axis=1)).flatten().astype(np.float32)

                    if full_chunk.dtype != np.float32:
                        full_chunk = full_chunk.astype(np.float32)

                    split_chunk = full_chunk[chunk_split_mask]
                    split_lib = lib_size[chunk_split_mask]
                    del full_chunk, lib_size

                    if not isinstance(split_chunk, sparse.csr_matrix):
                        split_chunk = split_chunk.tocsr()

                    # Normalize
                    row_sums = np.asarray(split_chunk.sum(axis=1)).flatten()
                    row_sums[row_sums == 0] = 1.0
                    scale = sparse.diags((target_sum / row_sums).astype(np.float32))
                    norm_chunk = (scale @ split_chunk).tocsr()
                    norm_chunk.data = np.log1p(norm_chunk.data).astype(np.float32)

                    # Append counts CSR
                    nnz_x = split_chunk.nnz
                    if nnz_x > 0:
                        x_data.resize(total_nnz_x + nnz_x, axis=0)
                        x_data[total_nnz_x:] = split_chunk.data
                        x_indices.resize(total_nnz_x + nnz_x, axis=0)
                        x_indices[total_nnz_x:] = split_chunk.indices.astype(np.int32)
                    for row_nnz in np.diff(split_chunk.indptr):
                        x_indptr_list.append(x_indptr_list[-1] + row_nnz)
                    total_nnz_x += nnz_x

                    # Append normalized CSR
                    nnz_n = norm_chunk.nnz
                    if nnz_n > 0:
                        n_data.resize(total_nnz_n + nnz_n, axis=0)
                        n_data[total_nnz_n:] = norm_chunk.data
                        n_indices.resize(total_nnz_n + nnz_n, axis=0)
                        n_indices[total_nnz_n:] = norm_chunk.indices.astype(np.int32)
                    for row_nnz in np.diff(norm_chunk.indptr):
                        n_indptr_list.append(n_indptr_list[-1] + row_nnz)
                    total_nnz_n += nnz_n

                    lib_sizes.append(split_lib)
                    del split_chunk, norm_chunk, scale
                    gc.collect()

                # Finalize h5py file
                x_grp.create_dataset("indptr", data=np.array(x_indptr_list, dtype=np.int64))
                x_grp.attrs["encoding-type"] = "csr_matrix"
                x_grp.attrs["encoding-version"] = "0.1.0"
                x_grp.attrs["shape"] = [n_cells, n_vars]

                norm_grp.create_dataset("indptr", data=np.array(n_indptr_list, dtype=np.int64))
                norm_grp.attrs["encoding-type"] = "csr_matrix"
                norm_grp.attrs["encoding-version"] = "0.1.0"
                norm_grp.attrs["shape"] = [n_cells, n_vars]

                library_size = np.concatenate(lib_sizes)
                std_obs = build_standardized_obs(
                    obs, obs_indices, cell_type_col, batch_key,
                    dataset_name, library_size, extra_columns,
                )
                _write_obs_to_h5(out_f.create_group("obs"), std_obs)
                _write_var_to_h5(out_f.create_group("var"), var)

                if obsm_arrays:
                    obsm_grp = out_f.create_group("obsm")
                    for key, arr in obsm_arrays.items():
                        obsm_grp.create_dataset(key, data=arr[obs_indices])

                out_f["layers"].attrs["encoding-type"] = "dict"
                out_f["layers"].attrs["encoding-version"] = "0.1.0"
                out_f.attrs["encoding-type"] = "anndata"
                out_f.attrs["encoding-version"] = "0.1.0"

        file_mb = output_path.stat().st_size / 1e6
        all_stats[name] = {
            "n_cells": n_cells,
            "nnz_counts": int(total_nnz_x),
            "nnz_normalized": int(total_nnz_n),
            "library_size_mean": float(library_size.mean()),
            "library_size_median": float(np.median(library_size)),
            "file_size_mb": round(file_mb, 1),
        }
        del obs_indices, library_size, std_obs
        gc.collect()

    return all_stats


# ---------------------------------------------------------------------------
# Public entry point
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
    """Build standardized train/val/test h5ad files from a source h5ad.

    Each output h5ad contains:
      .X = raw counts (float32, CSR sparse, no compression)
      .layers['normalized'] = log1p(normalize_total(X, target_sum))
      .obs = standardized columns + optional extras
      .var = all genes with highly_variable boolean annotation
      .obsm = carried over from source (X_umap, X_pca, etc.)

    Args:
        h5ad_path: Path to source h5ad file.
        output_dir: Directory for output files (train.h5ad, val.h5ad, test.h5ad).
        split_col: Array of 'train'/'val'/'test' per cell (length = n_obs).
        hvg_mask: Boolean mask over all genes.
        all_var: Full var DataFrame from source.
        obs: Full obs DataFrame from source.
        cell_type_col: obs column for cell type.
        batch_key: obs column for batch.
        dataset_name: Name for the 'dataset' obs column.
        matrix_source: Where to read expression ('raw.X', 'X', 'layers/<name>').
        chunk_size: Cells per read chunk.
        target_sum: Target sum for normalize_total.
        extra_obs_columns: Source→dest column mapping for additional obs columns.
        streaming: Use streaming mode for very large datasets.

    Returns:
        Dict of per-split stats: n_cells, nnz_counts, file_size_mb, etc.
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = _write_streaming if streaming else _write_accumulate
    return writer(
        h5ad_path=h5ad_path,
        matrix_source=matrix_source,
        hvg_mask=hvg_mask,
        all_var=all_var,
        obs=obs,
        split_col=split_col,
        cell_type_col=cell_type_col,
        batch_key=batch_key,
        dataset_name=dataset_name,
        output_dir=output_dir,
        chunk_size=chunk_size,
        target_sum=target_sum,
        extra_columns=extra_obs_columns,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/test_standardize.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sjanpy/ml/standardize.py tests/test_standardize.py
git commit -m "feat(ml): add standardize module — build standardized h5ad with accumulate and streaming modes"
```

---

### Task 6: Update `__init__.py` Exports and Refactor `build_dataset.py`

**Files:**
- Modify: `sjanpy/pp/__init__.py`
- Modify: `sjanpy/ml/__init__.py`
- Modify: `sjanpy/ml/build_dataset.py` (lines 73–212: remove internal h5py readers, import from h5ad_io)

- [ ] **Step 1: Update `sjanpy/pp/__init__.py`**

Add imports for new modules:

```python
from .genecraft import (
    filter_human_sc_genes,
    filter_mouse_sc_genes,
    filter_rat_sc_genes,
    get_background_gene_dict,
)
from .split import stratified_split
from .hvg import prepare_hvg_sample, compute_hvg
```

- [ ] **Step 2: Update `sjanpy/ml/__init__.py`**

Replace the current content with:

```python
from .build_dataset import (
    # Gene filtering
    load_gene_list,
    resolve_gene_indices,
    # Condition DSL
    parse_numerical_spec,
    parse_cat_spec,
    apply_transforms,
    build_condition_schema,
    build_condition_tensor,
    # Condition schema I/O
    save_condition_schema,
    load_condition_schema,
    # Core processing
    process_file,
    build_dataset,
)

from .h5ad_io import (
    read_obs,
    read_var,
    locate_matrix,
    get_matrix_shape,
    read_matrix_rows,
    read_sparse_chunk,
    validate_matrix_values,
)

from .standardize import (
    build_standardized_h5ads,
    build_standardized_obs,
)

# Backward compatibility: old names from build_dataset
from .h5ad_io import read_obs as read_obs_h5py
from .h5ad_io import read_var as read_var_h5py
```

- [ ] **Step 3: Refactor `build_dataset.py` to use `h5ad_io`**

In `sjanpy/ml/build_dataset.py`, replace the internal h5py readers (lines 69–212) with imports from `h5ad_io`:

Remove the following functions from `build_dataset.py`:
- `_decode_stringlike` (lines 73–87)
- `read_obs_h5py` (lines 89–162)
- `read_var_h5py` (lines 165–212)

Replace with imports at the top of the file:

```python
from .h5ad_io import read_obs as read_obs_h5py, read_var as read_var_h5py
```

Keep the rest of `build_dataset.py` unchanged.

- [ ] **Step 4: Run all tests to verify nothing broke**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/ -v`

Expected: All tests PASS.

Also verify imports work:

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -c "from sjanpy.pp import stratified_split, prepare_hvg_sample, compute_hvg; from sjanpy.ml import read_obs, read_var, build_standardized_h5ads, read_obs_h5py; print('All imports OK')"`

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add sjanpy/pp/__init__.py sjanpy/ml/__init__.py sjanpy/ml/build_dataset.py
git commit -m "refactor(ml): consolidate h5py readers into h5ad_io, update exports for new modules"
```

---

### Task 7: Run Full Test Suite and Fix Issues

**Files:** All test files

- [ ] **Step 1: Run full test suite**

Run: `cd /home/users/chensj16/s/projects/sjanpy && python -m pytest tests/ -v --tb=short`

Expected: All tests PASS.

- [ ] **Step 2: Run import smoke test**

Run:
```bash
cd /home/users/chensj16/s/projects/sjanpy && python -c "
import sjanpy
from sjanpy.pp import stratified_split, prepare_hvg_sample, compute_hvg
from sjanpy.ml import read_obs, read_var, locate_matrix, get_matrix_shape
from sjanpy.ml import read_matrix_rows, read_sparse_chunk, validate_matrix_values
from sjanpy.ml import build_standardized_h5ads, build_standardized_obs
from sjanpy.ml import read_obs_h5py, read_var_h5py  # backward compat
from sjanpy.ml import build_dataset  # existing module still works
print('All imports OK')
print(f'sjanpy version: {sjanpy.__version__}')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Fix any issues found, commit**

If all passes:
```bash
git add -A
git commit -m "test: verify full test suite and import compatibility"
```

---
