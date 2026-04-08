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

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Rationale
   * - remove_predicted
     - True
     - LOC/AC/AL clones add noise
   * - remove_non_coding
     - True
     - LINC/MIR/SNOR rarely informative
   * - remove_antisense
     - True
     - \-AS transcripts confound analyses
   * - remove_ig_var
     - True
     - IG variable regions dominate B cell PCA
   * - remove_hb
     - True
     - Hemoglobin contamination
   * - remove_metallothionein
     - True
     - Stress response artifact
   * - remove_mt_encoded
     - **False**
     - Keep for QC (% mitochondrial)
   * - remove_ribo
     - **False**
     - Keep for QC (% ribosomal)
   * - remove_histone
     - False
     - Usually fine unless studying cell cycle
