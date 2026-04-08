sjanpy
======

**Subjacent Analysis Toolkits for Single-Cell Omics in Python**

A collection of visualization and analysis utilities for single-cell RNA-seq
workflows. Built on top of `Scanpy <https://scanpy.readthedocs.io/>`_ and
`AnnData <https://anndata.readthedocs.io/>`_.

sjanpy follows the Scanpy subpackage convention:

- **sjanpy.pl** -- Plotting: embeddings, dotplots, bar plots, volcano plots, Nebulosa density
- **sjanpy.tl** -- Tools: differential expression analysis, Pearson residuals normalization
- **sjanpy.pp** -- Preprocessing: gene filtering, stratified splitting, HVG selection
- **sjanpy.ml** -- Machine Learning: h5ad I/O, standardization, dataset building (safetensors/pt), GPU/streaming datasets

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
