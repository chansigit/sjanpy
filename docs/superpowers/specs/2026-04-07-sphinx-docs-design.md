# Sphinx Documentation Design

## Goal

Add ReadTheDocs-compatible Sphinx documentation to sjanpy with API reference pages and one tutorial per module.

## Architecture

Sphinx with RST files, `sphinx-rtd-theme`, autodoc + napoleon for API extraction from docstrings, static code-block tutorials (no notebook execution).

## Structure

```
docs/
├── conf.py
├── Makefile
├── requirements.txt           # sphinx, sphinx-rtd-theme
├── index.rst                  # Landing: overview, toctree to install/tutorials/api
├── installation.rst
├── api/
│   ├── index.rst              # API overview, toctree to pl/tl/pp
│   ├── pl.rst                 # automodule for each pl module
│   ├── tl.rst                 # automodule for each tl module
│   └── pp.rst                 # automodule for each pp module
├── tutorials/
│   ├── index.rst              # Tutorial overview, toctree
│   ├── embedding.rst          # pbmc3k
│   ├── dotplot.rst            # synthetic
│   ├── barplot.rst            # synthetic
│   ├── volcano.rst            # pbmc3k
│   ├── nebulosa.rst           # synthetic
│   ├── deg.rst                # pbmc3k
│   ├── pres.rst               # synthetic
│   └── genecraft.rst          # pbmc3k
└── _static/
```

Plus `.readthedocs.yaml` at project root.

## Sphinx Configuration

- Theme: `sphinx-rtd-theme`
- Extensions: `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `sphinx.ext.autosummary`, `sphinx.ext.viewcode`
- Napoleon settings: numpy style enabled, google style disabled
- autodoc: member ordering by source, undoc-members shown

## API Reference Pages

One RST per subpackage. Each uses `.. automodule::` for the subpackage modules:

- **pl.rst**: documents `sjanpy.pl.embedding`, `sjanpy.pl.dotplot`, `sjanpy.pl.barplot`, `sjanpy.pl.volcano`, `sjanpy.pl.nebulosa`
- **tl.rst**: documents `sjanpy.tl.deg`, `sjanpy.tl.pres`
- **pp.rst**: documents `sjanpy.pp.genecraft`

## Tutorials

Each tutorial RST has: intro paragraph, code blocks showing usage, explanation of parameters, expected output description.

Dataset strategy:
- **pbmc3k_processed()**: embedding, volcano, deg, genecraft (need real biology)
- **Synthetic AnnData**: nebulosa, dotplot, barplot, pres (math/viz demos)

Tutorial scope per module:
- **embedding.rst**: load pbmc3k, call `fancy_embedding_pro` with categorical and continuous color, show density overlay
- **dotplot.rst**: create synthetic adata with marker genes, call `complex_dotplot` with dendrograms, `fan_dotplot`
- **barplot.rst**: create synthetic obs DataFrame, call `plot_stacked_bar_repel` in relative and absolute modes
- **volcano.rst**: assume DEG results from `fast_two_group_deg`, call `plot_volcano`, call `plot_cluster_deg_jitter_highlight`
- **nebulosa.rst**: create synthetic adata with known expression pattern, call `nebulosa_density` with show=True and show=False, show `wkde3d` standalone usage
- **deg.rst**: load pbmc3k, run `fast_two_group_deg` between two clusters, run `compute_nested_deg_df`, demonstrate `generate_highlight_dict`
- **pres.rst**: create synthetic count matrix, instantiate `PearsonResidualsScaler`, run `diagnose`, `fit_transform`, `get_statistics`
- **genecraft.rst**: load pbmc3k, show `get_background_gene_dict`, demonstrate `filter_human_sc_genes` with mask_hvg_only=True and False

## ReadTheDocs Config

`.readthedocs.yaml` at project root:
- Build with Python 3.11
- Install project with `pip install .` (needed for autodoc)
- Install doc dependencies from `docs/requirements.txt`

## Doc requirements.txt

```
sphinx>=7.0
sphinx-rtd-theme>=2.0
```
