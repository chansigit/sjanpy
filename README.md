# sjanpy

**Subjacent Analysis Toolkits for Single-Cell Omics in Python**

A collection of visualization and analysis utilities designed to enhance single-cell RNA-seq workflows. Built on top of [Scanpy](https://scanpy.readthedocs.io/) and [AnnData](https://anndata.readthedocs.io/), sjanpy provides publication-ready plotting functions and efficient analysis tools.

## Features

- **Nebulosa Density Plots** - Weighted kernel density estimation to address the overplotting problem in single-cell visualizations
- **Differential Expression Analysis** - Fast vectorized DEG computation with volcano plots and cluster-level comparisons
- **Advanced Dot Plots** - Hierarchical clustering, K-means grouping, dendrograms, and fan-shaped polar layouts
- **Enhanced Embeddings** - High-quality UMAP/t-SNE visualizations with density overlays and smart labeling
- **Stacked Bar Plots** - Cell composition analysis with intelligent label placement
- **Gene Filtering** - Utilities to remove uninformative genes (predicted, non-coding, artifacts) from analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sjanpy.git
cd sjanpy

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn scanpy anndata adjustText statsmodels scikit-learn
```

## Quick Start

```python
import scanpy as sc
from sjanpy import nebulosa, dotplot, deg, embedding

# Load your AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# Nebulosa density plot
nebulosa.nebulosa_density(adata, coord_key="X_umap", gene="CD3D", show=True)

# Complex dot plot with clustering
dotplot.complex_dotplot(adata, genes=marker_genes, groupby="cell_type")

# Differential expression analysis
results = deg.fast_two_group_deg(adata, label_col="condition", lst1=["Disease"], lst2=["Control"])

# High-quality embedding
embedding.fancy_embedding_pro(adata, basis="umap", color="cell_type")
```

## Modules

| Module | Description |
|--------|-------------|
| `nebulosa` | Weighted 2D KDE for gene expression visualization |
| `deg` | Differential expression analysis and volcano plots |
| `dotplot` | Complex dot plots with hierarchical clustering and fan layouts |
| `embedding` | Publication-ready UMAP/t-SNE visualizations |
| `barplot` | Stacked bar plots for cell composition |
| `genecraft` | Gene filtering utilities for scRNA-seq |

## Nebulosa: Solving the Overplotting Problem

Traditional scatter plots can obscure gene expression patterns due to point overlap. Nebulosa uses weighted kernel density estimation to reveal true expression distributions.

| Before | After |
|--------|-------|
| <img width="328" alt="before" src="https://github.com/user-attachments/assets/4c481b00-583b-4e7e-b064-95db59160024" /> | <img width="328" alt="after" src="https://github.com/user-attachments/assets/d4e2cc47-7d73-40d1-9b81-8360083780d1" /> |

## Dependencies

- Python >= 3.8
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scanpy
- anndata
- adjustText
- statsmodels
- scikit-learn

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
