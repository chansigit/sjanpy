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
