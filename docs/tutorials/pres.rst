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
