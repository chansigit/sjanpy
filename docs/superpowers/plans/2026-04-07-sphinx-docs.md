# Sphinx Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ReadTheDocs-compatible Sphinx documentation with API reference and one tutorial per module.

**Architecture:** Sphinx RST docs with `sphinx-rtd-theme`, `autodoc`+`napoleon` for API extraction, static code-block tutorials. No notebook execution. `.readthedocs.yaml` for RTD integration.

**Tech Stack:** Sphinx >= 7.0, sphinx-rtd-theme >= 2.0, RST markup

---

## File Structure

```
.readthedocs.yaml                      # RTD build config (project root)
docs/
├── conf.py                            # Sphinx configuration
├── Makefile                           # Build commands
├── requirements.txt                   # Doc build dependencies
├── index.rst                          # Landing page
├── installation.rst                   # Install guide
├── _static/                           # Empty dir for custom assets
├── api/
│   ├── index.rst                      # API overview + toctree
│   ├── pl.rst                         # pl subpackage API
│   ├── tl.rst                         # tl subpackage API
│   └── pp.rst                         # pp subpackage API
└── tutorials/
    ├── index.rst                      # Tutorial overview + toctree
    ├── embedding.rst                  # fancy_embedding_pro tutorial
    ├── dotplot.rst                    # complex_dotplot + fan_dotplot
    ├── barplot.rst                    # plot_stacked_bar_repel
    ├── volcano.rst                    # plot_volcano + jitter highlight
    ├── nebulosa.rst                   # nebulosa_density + wkde3d
    ├── deg.rst                        # DEG computation functions
    ├── pres.rst                       # PearsonResidualsScaler
    └── genecraft.rst                  # Gene filtering
```

---

### Task 1: Sphinx scaffolding and build infrastructure

**Files:**
- Create: `.readthedocs.yaml`
- Create: `docs/conf.py`
- Create: `docs/Makefile`
- Create: `docs/requirements.txt`
- Create: `docs/_static/.gitkeep`

- [ ] **Step 1: Create `docs/requirements.txt`**

```
sphinx>=7.0
sphinx-rtd-theme>=2.0
```

- [ ] **Step 2: Create `docs/conf.py`**

```python
project = "sjanpy"
copyright = "2026, sjanpy contributors"
author = "sjanpy contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "superpowers"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_undoc_members = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
```

- [ ] **Step 3: Create `docs/Makefile`**

```makefile
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
```

- [ ] **Step 4: Create `.readthedocs.yaml` at project root**

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

- [ ] **Step 5: Create `docs/_static/.gitkeep`**

```bash
mkdir -p docs/_static
touch docs/_static/.gitkeep
```

- [ ] **Step 6: Verify Sphinx builds with no content yet**

Create a minimal `docs/index.rst` to test the build:

```rst
sjanpy
======

Placeholder.
```

Run:

```bash
cd /scratch/users/chensj16/projects/sjanpy
pip install sphinx sphinx-rtd-theme -q
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

Expected: build succeeds with warnings about missing toctree (that's fine).

- [ ] **Step 7: Commit**

```bash
git add .readthedocs.yaml docs/conf.py docs/Makefile docs/requirements.txt docs/_static/.gitkeep docs/index.rst
git commit -m "docs: add Sphinx scaffolding and RTD config"
```

---

### Task 2: Landing page, installation, and top-level toctree

**Files:**
- Create: `docs/index.rst`
- Create: `docs/installation.rst`

- [ ] **Step 1: Write `docs/index.rst`**

```rst
sjanpy
======

**Subjacent Analysis Toolkits for Single-Cell Omics in Python**

A collection of visualization and analysis utilities for single-cell RNA-seq
workflows. Built on top of `Scanpy <https://scanpy.readthedocs.io/>`_ and
`AnnData <https://anndata.readthedocs.io/>`_.

sjanpy follows the Scanpy subpackage convention:

- **sjanpy.pl** -- Plotting: embeddings, dotplots, bar plots, volcano plots, Nebulosa density
- **sjanpy.tl** -- Tools: differential expression analysis, Pearson residuals normalization
- **sjanpy.pp** -- Preprocessing: organism-specific gene filtering

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
```

- [ ] **Step 2: Write `docs/installation.rst`**

```rst
Installation
============

From source
-----------

.. code-block:: bash

   git clone https://github.com/yourusername/sjanpy.git
   cd sjanpy
   pip install .

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Dependencies
------------

sjanpy requires Python >= 3.8 and the following packages:

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
- plotly
```

- [ ] **Step 3: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

Expected: build succeeds. Warnings about missing `tutorials/index` and `api/index` are expected (created in later tasks).

- [ ] **Step 4: Commit**

```bash
git add docs/index.rst docs/installation.rst
git commit -m "docs: add landing page and installation guide"
```

---

### Task 3: API reference pages

**Files:**
- Create: `docs/api/index.rst`
- Create: `docs/api/pl.rst`
- Create: `docs/api/tl.rst`
- Create: `docs/api/pp.rst`

- [ ] **Step 1: Write `docs/api/index.rst`**

```rst
API Reference
=============

.. toctree::
   :maxdepth: 2

   pl
   tl
   pp
```

- [ ] **Step 2: Write `docs/api/pl.rst`**

```rst
Plotting (``sjanpy.pl``)
=========================

Embedding
---------

.. automodule:: sjanpy.pl.embedding
   :members:
   :undoc-members:

Dot Plot
--------

.. automodule:: sjanpy.pl.dotplot
   :members:
   :undoc-members:

Bar Plot
--------

.. automodule:: sjanpy.pl.barplot
   :members:
   :undoc-members:

Volcano Plot
------------

.. automodule:: sjanpy.pl.volcano
   :members:
   :undoc-members:

Nebulosa Density
----------------

.. automodule:: sjanpy.pl.nebulosa
   :members:
   :undoc-members:
```

- [ ] **Step 3: Write `docs/api/tl.rst`**

```rst
Tools (``sjanpy.tl``)
======================

Differential Expression
-----------------------

.. automodule:: sjanpy.tl.deg
   :members:
   :undoc-members:

Pearson Residuals
-----------------

.. automodule:: sjanpy.tl.pres
   :members:
   :undoc-members:
```

- [ ] **Step 4: Write `docs/api/pp.rst`**

```rst
Preprocessing (``sjanpy.pp``)
==============================

Gene Filtering
--------------

.. automodule:: sjanpy.pp.genecraft
   :members:
   :undoc-members:
```

- [ ] **Step 5: Verify autodoc works**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -10
```

Expected: build succeeds. Check that API pages are generated with function signatures. Warnings about missing cross-references to external types (AnnData, DataFrame) are acceptable.

- [ ] **Step 6: Commit**

```bash
git add docs/api/
git commit -m "docs: add API reference pages with autodoc"
```

---

### Task 4: Tutorial — embedding

**Files:**
- Create: `docs/tutorials/index.rst`
- Create: `docs/tutorials/embedding.rst`

- [ ] **Step 1: Write `docs/tutorials/index.rst`**

```rst
Tutorials
=========

Step-by-step guides for each sjanpy module.

Plotting
--------

.. toctree::
   :maxdepth: 1

   embedding
   dotplot
   barplot
   volcano
   nebulosa

Analysis
--------

.. toctree::
   :maxdepth: 1

   deg
   pres

Preprocessing
-------------

.. toctree::
   :maxdepth: 1

   genecraft
```

- [ ] **Step 2: Write `docs/tutorials/embedding.rst`**

```rst
Embedding Visualization
=======================

The :func:`~sjanpy.pl.embedding.fancy_embedding_pro` function creates
publication-quality UMAP/t-SNE scatter plots with automatic label placement,
density overlays, and equal-aspect axes.

Loading data
------------

This tutorial uses the PBMC 3k dataset from Scanpy:

.. code-block:: python

   import scanpy as sc
   adata = sc.datasets.pbmc3k_processed()

Basic categorical plot
----------------------

Color cells by cluster identity with density contours:

.. code-block:: python

   from sjanpy.pl import fancy_embedding_pro

   fancy_embedding_pro(adata, basis='umap', color='louvain')

This produces a scatter plot with:

- Each cluster colored by the ``tab20`` palette
- KDE density contours in the background
- Bold centroid labels with automatic repelling to avoid overlap
- A legend on the right side

Customizing the plot
--------------------

Change the legend title, hide density, and adjust dot size:

.. code-block:: python

   fancy_embedding_pro(
       adata,
       basis='umap',
       color='louvain',
       legend_title='Cell Type',
       show_density=False,
       dot_size=8,
       alpha=0.6,
       figsize=(12, 10),
   )

Continuous variable (gene expression)
--------------------------------------

Pass a gene name to ``color`` to visualize expression:

.. code-block:: python

   fancy_embedding_pro(
       adata,
       basis='umap',
       color='CST3',
       legend_title='CST3 Expression',
       show_density=False,
   )

When ``color`` is a gene name, the function automatically switches to a
continuous colormap (``viridis``) and replaces the legend with a colorbar.

Saving the figure
-----------------

.. code-block:: python

   fancy_embedding_pro(
       adata,
       basis='umap',
       color='louvain',
       save_path='embedding.pdf',
       dpi=300,
   )
```

- [ ] **Step 3: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/index.rst docs/tutorials/embedding.rst
git commit -m "docs: add tutorials index and embedding tutorial"
```

---

### Task 5: Tutorial — dotplot

**Files:**
- Create: `docs/tutorials/dotplot.rst`

- [ ] **Step 1: Write `docs/tutorials/dotplot.rst`**

```rst
Dot Plot
========

sjanpy provides two dot plot styles: a rectangular dot plot with hierarchical
clustering and dendrograms, and a fan-shaped polar dot plot.

Preparing data
--------------

Create a synthetic AnnData with marker genes across cell types:

.. code-block:: python

   import numpy as np
   import anndata as ad
   import pandas as pd

   np.random.seed(42)
   n_cells, n_genes = 300, 20
   X = np.random.rand(n_cells, n_genes)
   gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
   cell_types = np.repeat(["B cell", "T cell", "Monocyte", "NK", "DC"], 60)

   # Boost marker genes per type
   X[:60, :4] += 3      # B cell markers
   X[60:120, 4:8] += 3  # T cell markers
   X[120:180, 8:12] += 3
   X[180:240, 12:16] += 3
   X[240:, 16:] += 3

   adata = ad.AnnData(
       X=X,
       var=pd.DataFrame(index=gene_names),
       obs=pd.DataFrame({"cell_type": pd.Categorical(cell_types)}),
   )

Complex dot plot with clustering
---------------------------------

.. code-block:: python

   from sjanpy.pl import complex_dotplot

   complex_dotplot(
       adata,
       genes=gene_names,
       groupby='cell_type',
       z_score=True,
       cluster_rows=True,
       cluster_cols=True,
       use_olo=True,
       cmap='RdBu_r',
   )

Key parameters:

- ``z_score=True``: normalize expression across groups for better contrast
- ``cluster_rows`` / ``cluster_cols``: enable hierarchical clustering with
  dendrograms
- ``use_olo``: optimal leaf ordering for cleaner dendrograms
- ``row_km`` / ``col_km``: use K-means clustering instead (pass an integer)

Manual gene and group ordering
-------------------------------

Override clustering with explicit order:

.. code-block:: python

   complex_dotplot(
       adata,
       genes=gene_names,
       groupby='cell_type',
       manual_gene_order=["Gene_1", "Gene_5", "Gene_10", "Gene_15", "Gene_20"],
       manual_group_order=["T cell", "B cell", "NK", "Monocyte", "DC"],
       show_dendrogram_x=False,
       show_dendrogram_y=False,
   )

Preparing data with filtering
------------------------------

Use :func:`~sjanpy.pl.dotplot.get_dotplot_df` to filter low-expression genes:

.. code-block:: python

   from sjanpy.pl import get_dotplot_df

   df = get_dotplot_df(
       adata,
       genes=gene_names,
       groupby='cell_type',
       expr_threshold=0.5,
       min_pct=10,
       keep_genes=["Gene_1", "Gene_2"],  # always include these
   )
   print(df.head())

Fan-shaped dot plot
-------------------

Use :func:`~sjanpy.pl.dotplot.fan_dotplot` for a polar layout:

.. code-block:: python

   from sjanpy.pl import fan_dotplot

   fan_dotplot(
       df,
       start_deg=-60,
       end_deg=60,
       cmap='RdYlBu_r',
       title='Marker Gene Expression',
   )
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/dotplot.rst
git commit -m "docs: add dotplot tutorial"
```

---

### Task 6: Tutorial — barplot

**Files:**
- Create: `docs/tutorials/barplot.rst`

- [ ] **Step 1: Write `docs/tutorials/barplot.rst`**

```rst
Stacked Bar Plot
================

:func:`~sjanpy.pl.barplot.plot_stacked_bar_repel` creates stacked bar plots
with intelligent label placement for visualizing cell type composition.

Preparing data
--------------

Create a synthetic observation DataFrame:

.. code-block:: python

   import pandas as pd
   import numpy as np

   np.random.seed(42)
   n = 1000
   obs_df = pd.DataFrame({
       'cell_type': np.random.choice(
           ['B cell', 'T cell', 'Monocyte', 'NK', 'DC'],
           size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1]
       ),
       'sample': np.random.choice(
           ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4'],
           size=n
       ),
   })

Relative composition
--------------------

Show proportions per cell type:

.. code-block:: python

   from sjanpy.pl import plot_stacked_bar_repel

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       mode='relative',
       label_content='percentage',
   )

Small slices are automatically labeled with leader lines using ``adjustText``
to avoid overlap.

Absolute counts
---------------

Switch to raw counts with optional log scale:

.. code-block:: python

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       mode='absolute',
       log_scale=True,
       label_content='count',
   )

Label options
-------------

The ``label_content`` parameter controls what appears on each bar segment:

- ``'percentage'``: show percentage (default)
- ``'count'``: show raw count
- ``'both'``: show count and percentage

Adjust ``min_label_threshold`` to control when labels switch from inline to
leader-line style (default: 0.03 = 3%).

Saving
------

.. code-block:: python

   plot_stacked_bar_repel(
       obs_df,
       group_col='sample',
       type_col='cell_type',
       save_path='barplot.pdf',
   )
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/barplot.rst
git commit -m "docs: add barplot tutorial"
```

---

### Task 7: Tutorial — volcano

**Files:**
- Create: `docs/tutorials/volcano.rst`

- [ ] **Step 1: Write `docs/tutorials/volcano.rst`**

```rst
Volcano and Jitter Plots
=========================

sjanpy provides two plotting functions for visualizing differential expression
results: a volcano plot and a per-cluster jitter plot with gene highlights.

Preparing DEG results
---------------------

These plots consume DataFrames produced by the DEG computation functions in
``sjanpy.tl``. Here we use PBMC 3k as an example:

.. code-block:: python

   import scanpy as sc
   from sjanpy.tl import fast_two_group_deg

   adata = sc.datasets.pbmc3k_processed()

   # Compare B cells vs T cells
   deg_results = fast_two_group_deg(
       adata,
       label_col='louvain',
       lst1=['B cells'],
       lst2=['CD4 T cells'],
   )
   print(deg_results.head())

Volcano plot
------------

.. code-block:: python

   from sjanpy.pl import plot_volcano

   plot_volcano(
       deg_results,
       logfc_col='log2FC',
       padj_col='padj',
       lfc_thr=1.0,
       adj_p_thr=0.05,
       title='B cells vs CD4 T cells',
   )

The plot marks genes as Up (teal), Down (salmon), or NS (grey) based on the
log fold-change and adjusted p-value thresholds. Dashed lines show the cutoffs.

Cluster-level jitter plot
--------------------------

When using :func:`~sjanpy.tl.deg.compute_nested_deg_df` to compute DEGs within
each cluster, visualize results with a jitter plot:

.. code-block:: python

   from sjanpy.tl import compute_nested_deg_df, generate_highlight_dict
   from sjanpy.pl import plot_cluster_deg_jitter_highlight

   # Compute within-cluster DEGs (requires a condition column)
   # For demonstration, assume adata.obs has a 'condition' column
   nested_deg = compute_nested_deg_df(
       adata,
       cluster_key='louvain',
       condition_key='condition',
       target_condition='Disease',
       reference_condition='Control',
   )

   # Select genes to highlight
   highlights = generate_highlight_dict(
       nested_deg,
       strategies=['topn'],
       cluster_key='cluster',
       top_n=3,
       exclude_regex=[r'^MT-', r'^RP[SL]'],
   )

   # Plot
   plot_cluster_deg_jitter_highlight(
       nested_deg,
       cluster_key='cluster',
       target_name='Disease',
       reference_name='Control',
       highlight_dict=highlights,
       vrange=(-5, 5),
   )

Customizing highlight strategies
---------------------------------

``generate_highlight_dict`` supports three strategies that can be combined:

- ``'topn'``: top N genes by absolute logFC per cluster
- ``'ktimes'``: genes that are significant in at least k clusters
- ``'manual'``: explicitly specified gene list

.. code-block:: python

   highlights = generate_highlight_dict(
       nested_deg,
       strategies=['topn', 'ktimes', 'manual'],
       top_n=5,
       k=3,
       ktimes_poscut=1.0,
       ktimes_negcut=-1.0,
       manual_genes=['CD3D', 'MS4A1', 'LYZ'],
       exclude_regex=[r'^MT-', r'^RP[SL]', r'^AC\d+'],
   )
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/volcano.rst
git commit -m "docs: add volcano and jitter plot tutorial"
```

---

### Task 8: Tutorial — nebulosa

**Files:**
- Create: `docs/tutorials/nebulosa.rst`

- [ ] **Step 1: Write `docs/tutorials/nebulosa.rst`**

```rst
Nebulosa Density Plots
======================

Nebulosa uses weighted kernel density estimation to address the overplotting
problem in single-cell embeddings. Instead of coloring each cell by raw
expression, it computes a smoothed density surface weighted by gene expression.

The core idea: cells with high expression in dense regions produce bright
density peaks, while isolated expressing cells produce dimmer signals.

Preparing data
--------------

Create a synthetic AnnData with two clusters and a cluster-specific gene:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import anndata as ad

   np.random.seed(42)
   n = 500

   # Two clusters in UMAP space
   coords = np.vstack([
       np.random.normal(loc=[-2, -2], scale=0.8, size=(n // 2, 2)),
       np.random.normal(loc=[2, 2], scale=0.8, size=(n // 2, 2)),
   ])

   # Gene expressed only in cluster 1
   expr = np.zeros((n, 1))
   expr[:n // 2, 0] = np.random.exponential(2, size=n // 2)

   adata = ad.AnnData(
       X=expr,
       var=pd.DataFrame(index=['MarkerGene']),
       obsm={'X_umap': coords},
   )

Computing density values
------------------------

Use ``show=False`` to get per-cell density values (e.g. for downstream use):

.. code-block:: python

   from sjanpy.pl import nebulosa_density

   densities = nebulosa_density(
       adata,
       coord_key='X_umap',
       gene='MarkerGene',
       show=False,
   )
   print(densities.shape)  # (500,)

   # Store in obs for other plotting tools
   adata.obs['marker_density'] = densities

Plotting
--------

Use ``show=True`` to produce a scatter plot colored by density:

.. code-block:: python

   nebulosa_density(
       adata,
       coord_key='X_umap',
       gene='MarkerGene',
       show=True,
       cmap='magma',
   )

Adjusting bandwidth
-------------------

The ``adjust`` parameter scales the KDE bandwidth. Smaller values give sharper
peaks; larger values produce smoother density fields:

.. code-block:: python

   # Sharper
   nebulosa_density(adata, 'X_umap', 'MarkerGene', adjust=0.5, show=True)

   # Smoother
   nebulosa_density(adata, 'X_umap', 'MarkerGene', adjust=2.0, show=True)

Low-level 2D KDE
-----------------

Use :func:`~sjanpy.pl.nebulosa.wkde2d` directly for custom workflows:

.. code-block:: python

   from sjanpy.pl.nebulosa import wkde2d

   x = coords[:, 0]
   y = coords[:, 1]
   w = expr[:, 0]

   gx, gy, z = wkde2d(x, y, w, adjust=1.0, n=100)
   print(z.shape)  # (100, 100)

3D Weighted KDE
---------------

For 3D embeddings (e.g. from UMAP with ``n_components=3``), use
:func:`~sjanpy.pl.nebulosa.wkde3d`:

.. code-block:: python

   from sjanpy.pl.nebulosa import wkde3d

   # Synthetic 3D coordinates
   coords_3d = np.random.randn(200, 3)
   weights = np.random.exponential(1, size=200)

   gx, gy, gz, Z = wkde3d(
       coords_3d[:, 0],
       coords_3d[:, 1],
       coords_3d[:, 2],
       weights,
       adjust=1.0,
       n=30,
   )
   print(Z.shape)  # (30, 30, 30)

The returned grid and density array can be visualized with Plotly or
matplotlib's 3D scatter.
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/nebulosa.rst
git commit -m "docs: add nebulosa tutorial"
```

---

### Task 9: Tutorial — deg

**Files:**
- Create: `docs/tutorials/deg.rst`

- [ ] **Step 1: Write `docs/tutorials/deg.rst`**

```rst
Differential Expression Analysis
=================================

``sjanpy.tl`` provides fast vectorized differential expression computation
and helper functions for multi-cluster analyses.

Fast two-group DEG
------------------

:func:`~sjanpy.tl.deg.fast_two_group_deg` uses Welch's t-test on the
expression matrix directly, much faster than Scanpy's rank_genes_groups for
simple two-group comparisons:

.. code-block:: python

   import scanpy as sc
   from sjanpy.tl import fast_two_group_deg

   adata = sc.datasets.pbmc3k_processed()

   results = fast_two_group_deg(
       adata,
       label_col='louvain',
       lst1=['B cells'],
       lst2=['CD4 T cells'],
   )
   print(results.head(10))

The result DataFrame contains:

- ``gene``: gene name
- ``log2FC``: log2 fold change (group1 vs group2)
- ``pct.1``, ``pct.2``: detection rates in each group
- ``pval``, ``padj``: raw and FDR-adjusted p-values

Within-cluster DEG
------------------

:func:`~sjanpy.tl.deg.compute_nested_deg_df` computes DEGs between two
conditions within each cluster, using Scanpy's rank_genes_groups:

.. code-block:: python

   from sjanpy.tl import compute_nested_deg_df

   # Requires a condition column in adata.obs
   nested_deg = compute_nested_deg_df(
       adata,
       cluster_key='louvain',
       condition_key='condition',
       target_condition='Disease',
       reference_condition='Control',
       method='wilcoxon',
       min_cells=10,
       compute_pct=True,
   )

Key parameters:

- ``min_cells``: skip clusters with fewer cells in either condition
- ``compute_pct``: add detection rate columns (``pct_target``, ``pct_reference``)
- ``expr_layer``: use a specific layer for detection rate (e.g. ``'counts'``)

Clipping extreme logFC
-----------------------

:func:`~sjanpy.tl.deg.clip_logfc_in_nested_deg_df` clips outlier logFC values
per cluster to prevent extreme values from dominating visualizations:

.. code-block:: python

   from sjanpy.tl import clip_logfc_in_nested_deg_df

   clipped = clip_logfc_in_nested_deg_df(
       nested_deg,
       logfc_col='logfc',
       cluster_col='cluster',
       quantile=0.95,
   )

Selecting genes to highlight
-----------------------------

:func:`~sjanpy.tl.deg.generate_highlight_dict` selects important genes per
cluster for labeling in plots:

.. code-block:: python

   from sjanpy.tl import generate_highlight_dict

   highlights = generate_highlight_dict(
       nested_deg,
       strategies=['topn', 'ktimes'],
       cluster_key='cluster',
       top_n=5,
       k=3,
       exclude_regex=[r'^MT-', r'^RP[SL]'],
   )

   # Returns: {'Cluster_0': ['GENE1', ...], 'Cluster_1': [...], ...}
   for cluster, genes in highlights.items():
       print(f"{cluster}: {genes}")

Three strategies can be combined:

- ``'topn'``: select top N genes by absolute logFC per cluster
- ``'ktimes'``: genes that exceed logFC cutoffs in at least k clusters
- ``'manual'``: user-specified gene list (filtered to those present in the data)

``exclude_regex`` removes unwanted genes (mitochondrial, ribosomal, etc.)
after selection.
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/deg.rst
git commit -m "docs: add DEG tutorial"
```

---

### Task 10: Tutorial — pres

**Files:**
- Create: `docs/tutorials/pres.rst`

- [ ] **Step 1: Write `docs/tutorials/pres.rst`**

```rst
Pearson Residuals Normalization
================================

:class:`~sjanpy.tl.pres.PearsonResidualsScaler` provides analytic Pearson
residuals normalization — a variance-stabilizing alternative to the standard
``log1p`` workflow. It models gene expression as a Negative Binomial
distribution and computes clipped residuals.

Creating synthetic count data
-----------------------------

.. code-block:: python

   import numpy as np
   import scipy.sparse as sp

   np.random.seed(42)
   n_cells, n_genes = 500, 200

   # Simulate sparse count matrix
   X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
   X = sp.csr_matrix(X)
   gene_names = [f"Gene_{i}" for i in range(n_genes)]

Diagnosing the input
---------------------

Before fitting, run diagnostics to check for data issues:

.. code-block:: python

   from sjanpy.tl import PearsonResidualsScaler

   scaler = PearsonResidualsScaler(theta=100, feature_names=gene_names)
   report = scaler.diagnose(X)
   print(report)

This reports NaN/Inf values, negative values, and zero-count genes.

Fitting and transforming
------------------------

.. code-block:: python

   residuals = scaler.fit_transform(X)
   print(residuals.shape)  # (500, 200)
   print(residuals.min(), residuals.max())

The ``theta`` parameter controls overdispersion (higher = closer to Poisson).
Residuals are clipped to ``[-sqrt(n_cells), sqrt(n_cells)]`` by default.

Custom clipping:

.. code-block:: python

   scaler = PearsonResidualsScaler(theta=100, clip=30, feature_names=gene_names)
   residuals = scaler.fit_transform(X)

Inspecting gene statistics
---------------------------

After fitting, retrieve per-gene statistics:

.. code-block:: python

   stats = scaler.get_statistics()
   print(stats.head())

The DataFrame includes:

- ``mean_counts``: observed mean expression
- ``residual_variance``: variance of the Pearson residuals (higher = more variable)
- ``gene_probability``: estimated relative abundance
- ``is_zero_count``: flag for unexpressed genes

You can use ``residual_variance`` to select highly variable genes:

.. code-block:: python

   hvg = stats.nlargest(50, 'residual_variance')
   print(hvg.index.tolist())

Using with AnnData
-------------------

.. code-block:: python

   import anndata as ad

   adata = ad.AnnData(X=X)
   adata.var_names = gene_names

   scaler = PearsonResidualsScaler(
       theta=100,
       feature_names=adata.var_names,
   )
   adata.layers['pearson_residuals'] = scaler.fit_transform(adata.X)
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/pres.rst
git commit -m "docs: add Pearson residuals tutorial"
```

---

### Task 11: Tutorial — genecraft

**Files:**
- Create: `docs/tutorials/genecraft.rst`

- [ ] **Step 1: Write `docs/tutorials/genecraft.rst`**

```rst
Gene Filtering
==============

``sjanpy.pp.genecraft`` provides organism-specific functions to remove or mask
uninformative genes from scRNA-seq data — predicted genes, non-coding RNAs,
hemoglobin, metallothioneins, and more.

Listing background genes
-------------------------

Use :func:`~sjanpy.pp.genecraft.get_background_gene_dict` to see which genes
in your dataset fall into artifact categories:

.. code-block:: python

   import scanpy as sc
   from sjanpy.pp import get_background_gene_dict

   adata = sc.datasets.pbmc3k_processed()

   bg = get_background_gene_dict(adata)
   for category, genes in bg.items():
       print(f"{category}: {len(genes)} genes — {genes[:5]}")

Categories include ``Mito_Encoded``, ``Ribosomal``, ``Hemoglobin``, ``HSP``,
``IEG``, ``Cell_Cycle``, ``Histone``, ``Genomic_Clone``, ``Predicted_LOC``,
and more.

Masking genes from HVG selection
---------------------------------

The recommended approach: keep the genes in the matrix but prevent them from
driving PCA/clustering by setting ``highly_variable = False``:

.. code-block:: python

   from sjanpy.pp import filter_human_sc_genes

   # Requires sc.pp.highly_variable_genes to have been run
   sc.pp.highly_variable_genes(adata)
   print(f"HVGs before: {adata.var['highly_variable'].sum()}")

   adata = filter_human_sc_genes(
       adata,
       mask_hvg_only=True,        # default: mask, don't remove
       remove_predicted=True,
       remove_non_coding=True,
       remove_antisense=True,
       remove_ig_var=True,
       remove_hb=True,
       remove_metallothionein=True,
       remove_mt_encoded=False,    # keep MT- for QC
       remove_ribo=False,          # keep ribosomal for QC
   )
   print(f"HVGs after: {adata.var['highly_variable'].sum()}")

Physically removing genes
--------------------------

Set ``mask_hvg_only=False`` to remove genes from the AnnData entirely:

.. code-block:: python

   n_before = adata.n_vars
   adata = filter_human_sc_genes(adata, mask_hvg_only=False)
   print(f"Genes: {n_before} -> {adata.n_vars}")

Mouse and rat data
-------------------

Separate functions handle organism-specific naming conventions:

.. code-block:: python

   from sjanpy.pp import filter_mouse_sc_genes, filter_rat_sc_genes

   # Mouse: Gm... predicted genes, mt- mito, Rp[sl] ribosomal
   adata_mouse = filter_mouse_sc_genes(adata_mouse, mask_hvg_only=True)

   # Rat: LOC/RGD predicted genes, Mt- mito
   adata_rat = filter_rat_sc_genes(adata_rat, mask_hvg_only=True)

Choosing what to remove
------------------------

Each gene category can be toggled independently. Typical choices:

+------------------------+-------------------+-------------------------------------------+
| Parameter              | Default           | Rationale                                 |
+========================+===================+===========================================+
| remove_predicted       | True              | LOC/AC/AL clones add noise                |
+------------------------+-------------------+-------------------------------------------+
| remove_non_coding      | True              | LINC/MIR/SNOR rarely informative          |
+------------------------+-------------------+-------------------------------------------+
| remove_antisense       | True              | -AS transcripts confound analyses         |
+------------------------+-------------------+-------------------------------------------+
| remove_ig_var          | True              | IG variable regions dominate B cell PCA   |
+------------------------+-------------------+-------------------------------------------+
| remove_hb              | True              | Hemoglobin contamination                  |
+------------------------+-------------------+-------------------------------------------+
| remove_metallothionein | True              | Stress response artifact                  |
+------------------------+-------------------+-------------------------------------------+
| remove_mt_encoded      | **False**         | Keep for QC (% mitochondrial)             |
+------------------------+-------------------+-------------------------------------------+
| remove_ribo            | **False**         | Keep for QC (% ribosomal)                 |
+------------------------+-------------------+-------------------------------------------+
| remove_histone         | False             | Usually fine unless studying cell cycle    |
+------------------------+-------------------+-------------------------------------------+
```

- [ ] **Step 2: Verify build**

```bash
sphinx-build -b html docs docs/_build/html 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/genecraft.rst
git commit -m "docs: add genecraft tutorial"
```

---

### Self-review

**Spec coverage check:**
- Sphinx scaffolding: Task 1
- Landing page + installation: Task 2
- API reference (pl, tl, pp): Task 3
- Tutorials (one per module): Tasks 4-11 (embedding, dotplot, barplot, volcano, nebulosa, deg, pres, genecraft)
- .readthedocs.yaml: Task 1
- Dataset strategy (pbmc3k vs synthetic): applied per tutorial as specified

All spec requirements covered. No placeholders found. Function names match the actual codebase.
