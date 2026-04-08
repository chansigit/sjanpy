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
