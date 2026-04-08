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
