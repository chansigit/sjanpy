# sjanpy

[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Subjacent Analysis Toolkits for Single-Cell Omics in Python**

sjanpy extends the [Scanpy](https://scanpy.readthedocs.io/) / [AnnData](https://anndata.readthedocs.io/) ecosystem with publication-quality visualizations, fast differential expression analysis, and preprocessing utilities for single-cell RNA-seq.

## Package Structure

sjanpy follows the Scanpy subpackage convention:

| Subpackage | Purpose | Key Functions |
|---|---|---|
| `sjanpy.pl` | **Plotting** | Embedding, dot plot, bar plot, volcano plot, Nebulosa density |
| `sjanpy.tl` | **Tools** | Differential expression, Pearson residuals normalization |
| `sjanpy.pp` | **Preprocessing** | Gene filtering, stratified splitting, HVG selection |
| `sjanpy.ml` | **Machine Learning** | h5ad I/O, standardization pipeline, chunked `.pt` dataset builder |

## Installation

```bash
git clone https://github.com/chansigit/sjanpy.git
cd sjanpy
pip install .
```

## Quick Start

### Embedding visualization

```python
import scanpy as sc
from sjanpy.pl import fancy_embedding_pro

adata = sc.datasets.pbmc3k_processed()
fancy_embedding_pro(adata, basis='umap', color='louvain')
```

### Differential expression

```python
from sjanpy.tl import fast_two_group_deg
from sjanpy.pl import plot_volcano

deg = fast_two_group_deg(adata, label_col='louvain', lst1=['B cells'], lst2=['CD4 T cells'])
plot_volcano(deg, logfc_col='log2FC', padj_col='padj')
```

### Nebulosa density

Traditional scatter plots obscure gene expression patterns due to point overlap. Nebulosa uses weighted kernel density estimation to reveal true expression distributions:

```python
from sjanpy.pl import nebulosa_density

nebulosa_density(adata, coord_key='X_umap', gene='CD3D', show=True)
```

| Standard scatter | Nebulosa density |
|---|---|
| <img width="328" alt="before" src="https://github.com/user-attachments/assets/4c481b00-583b-4e7e-b064-95db59160024" /> | <img width="328" alt="after" src="https://github.com/user-attachments/assets/d4e2cc47-7d73-40d1-9b81-8360083780d1" /> |

### Gene filtering

```python
from sjanpy.pp import filter_human_sc_genes

# Mask artifact genes from HVG selection (predicted, non-coding, IG variable, etc.)
adata = filter_human_sc_genes(adata, mask_hvg_only=True)
```

### Complex dot plot

```python
from sjanpy.pl import complex_dotplot

complex_dotplot(
    adata,
    genes=marker_genes,
    groupby='cell_type',
    z_score=True,
    cluster_rows=True,
    cmap='RdBu_r',
)
```

## Module Reference

### `sjanpy.pl` — Plotting

| Function | Description |
|---|---|
| `fancy_embedding_pro` | UMAP/t-SNE with density overlays, auto-labels, equal-aspect axes |
| `complex_dotplot` | Dot plot with hierarchical clustering and dendrograms |
| `fan_dotplot` | Polar/radial dot plot layout |
| `plot_stacked_bar_repel` | Stacked bar plot with smart label placement |
| `plot_volcano` | Volcano plot for DEG visualization |
| `plot_cluster_deg_jitter_highlight` | Per-cluster jitter plot with gene annotations |
| `nebulosa_density` | Weighted KDE density on embeddings |
| `wkde2d` / `wkde3d` | Low-level 2D/3D weighted kernel density estimation |

### `sjanpy.tl` — Tools

| Function / Class | Description |
|---|---|
| `fast_two_group_deg` | Vectorized Welch's t-test DEG between two groups |
| `compute_nested_deg_df` | Within-cluster DEG between two conditions |
| `clip_logfc_in_nested_deg_df` | Per-cluster quantile clipping of logFC |
| `generate_highlight_dict` | Select genes to label (top-N, k-times, manual) |
| `PearsonResidualsScaler` | NB-based Pearson residuals normalization |

### `sjanpy.pp` — Preprocessing

| Function | Description |
|---|---|
| `filter_human_sc_genes` | Remove/mask artifact genes (human) |
| `filter_mouse_sc_genes` | Remove/mask artifact genes (mouse) |
| `filter_rat_sc_genes` | Remove/mask artifact genes (rat) |
| `get_background_gene_dict` | Catalog artifact gene categories in a dataset |
| `stratified_split` | Two-stage stratified train/val/test splitting |
| `prepare_hvg_sample` | Stratified subsample of training cells for HVG computation |
| `compute_hvg` | Highly-variable gene selection with stratified sampling |

### `sjanpy.ml` — Machine Learning

#### h5ad I/O (`sjanpy.ml.h5ad_io`)

| Function | Description |
|---|---|
| `read_obs` / `read_var` | Read obs/var DataFrames from h5ad via h5py |
| `locate_matrix` | Locate expression matrix path in h5ad |
| `get_matrix_shape` | Get matrix dimensions without loading data |
| `read_matrix_rows` | Read specific rows from dense/sparse matrices |
| `read_sparse_chunk` | Read a chunk of a sparse matrix as CSR |
| `validate_matrix_values` | Validate matrix values (NaN, Inf checks) |

#### Standardization (`sjanpy.ml.standardize`)

| Function | Description |
|---|---|
| `build_standardized_h5ads` | Build per-split standardized h5ad files (accumulate or streaming) |
| `build_standardized_obs` | Build standardized obs with split assignments |

#### Dataset Building (`sjanpy.ml.build_dataset`)

| Function | Description |
|---|---|
| `build_dataset` | Stream h5ad → chunked `.pt` files with condition vectors |
| `build_condition_schema` | Build encoding schema from condition DSL specs |
| `process_file` | Process a single h5ad file into chunks |
| `load_gene_list` / `resolve_gene_indices` | Gene list loading and index resolution |
| `save_condition_schema` / `load_condition_schema` | Condition schema persistence |

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scanpy`, `anndata`, `adjustText`, `statsmodels`, `scikit-learn`

Optional: `plotly` (3D visualization), `torch` / `h5py` (ML dataset building)

## License

MIT
