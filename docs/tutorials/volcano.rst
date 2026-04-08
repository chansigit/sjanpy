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
