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
