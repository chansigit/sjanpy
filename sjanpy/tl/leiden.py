"""GPU-accelerated Leiden community detection, scanpy-compatible.

This module provides :func:`sjanpy.tl.leiden`, a drop-in replacement for
:func:`scanpy.tl.leiden` that uses the `gpu_leiden` CUDA backend instead of
leidenalg/igraph. The API follows scanpy's as closely as possible so
existing pipelines can switch by replacing ``sc.tl.leiden`` with
``sjanpy.tl.leiden``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted

if TYPE_CHECKING:
    from anndata import AnnData


def _resolve_adjacency(
    adata: "AnnData",
    adjacency: Any,
    neighbors_key: str | None,
    obsp: str | None,
) -> Any:
    """Resolve which adjacency matrix to use (scanpy priority order).

    Returns an arbitrary array-like (typically a scipy sparse matrix). The
    caller normalizes it to CSR via :func:`scipy.sparse.csr_matrix`.
    """
    if adjacency is not None and obsp is not None:
        raise ValueError("Cannot specify both `adjacency` and `obsp`.")
    if adjacency is not None and neighbors_key is not None:
        raise ValueError("Cannot specify both `adjacency` and `neighbors_key`.")
    if obsp is not None and neighbors_key is not None:
        raise ValueError("Cannot specify both `obsp` and `neighbors_key`.")

    if adjacency is not None:
        return adjacency
    if obsp is not None:
        return adata.obsp[obsp]
    if neighbors_key is not None:
        if neighbors_key not in adata.uns:
            raise KeyError(
                f"neighbors_key '{neighbors_key}' not found in adata.uns. "
                "Did you run sc.pp.neighbors?"
            )
        ckey = adata.uns[neighbors_key].get("connectivities_key", "connectivities")
        return adata.obsp[ckey]
    # Default
    if "connectivities" not in adata.obsp:
        raise KeyError(
            "adata.obsp['connectivities'] not found. "
            "Run sc.pp.neighbors first, or pass `adjacency`/`obsp`/`neighbors_key`."
        )
    return adata.obsp["connectivities"]


def leiden(
    adata: "AnnData",
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    key_added: str = "leiden",
    adjacency: Any = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int = -1,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    flavor: str = "gpu",
    **clustering_args: Any,
) -> "AnnData | None":
    """Cluster cells into subgroups using the GPU Leiden algorithm.

    A drop-in replacement for :func:`scanpy.tl.leiden` that uses the
    ``gpu_leiden`` CUDA backend. The adjacency matrix is passed as-is to the
    GPU kernel (treated as symmetric/undirected, matching scanpy's default
    connectivities graph).

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        Resolution (gamma) parameter. Higher values -> more, smaller communities.
    random_state
        Seed for reproducibility. Currently accepted but not yet used by the
        GPU backend (the algorithm is deterministic modulo atomic ordering).
    key_added
        ``adata.obs`` key under which to store the cluster labels.
    adjacency
        Sparse adjacency matrix to cluster on. Defaults to
        ``adata.obsp['connectivities']`` (see ``neighbors_key`` / ``obsp``).
    directed
        Accepted for API compatibility. The GPU backend always treats the
        graph as symmetric; pass a symmetric matrix (scanpy's connectivities
        are symmetric).
    use_weights
        If ``False``, replace the edge weights with ones before running the
        algorithm. Default ``True``.
    n_iterations
        Maximum number of Leiden iterations. ``-1`` for unlimited. Currently
        the GPU backend always runs to convergence; this is passed through
        but not yet enforced.
    neighbors_key
        Look up ``adata.obsp[adata.uns[neighbors_key]['connectivities_key']]``
        if specified, instead of the default ``adata.obsp['connectivities']``.
    obsp
        Use ``adata.obsp[obsp]`` directly. Mutually exclusive with
        ``adjacency`` and ``neighbors_key``.
    copy
        If ``True``, return a copy of ``adata`` with the labels added,
        instead of modifying ``adata`` in place.
    flavor
        Kept for scanpy API parity. Currently only ``"gpu"`` is supported by
        this function.
    **clustering_args
        Ignored for API parity with ``scanpy.tl.leiden``.

    Returns
    -------
    ``None`` if ``copy=False``, otherwise an ``AnnData`` copy with
    ``obs[key_added]`` set.
    """
    import gpu_leiden  # type: ignore[import-not-found]

    # `directed` and `**clustering_args` are accepted for scanpy API parity
    # but not used by the GPU backend.
    del directed, clustering_args

    if flavor != "gpu":
        raise ValueError(
            f"sjanpy.tl.leiden only supports flavor='gpu'; got {flavor!r}. "
            "Use scanpy.tl.leiden for 'leidenalg' or 'igraph' backends."
        )

    if copy:
        adata = adata.copy()

    adj_raw = _resolve_adjacency(adata, adjacency, neighbors_key, obsp)

    # Normalize to CSR so `.data` / `.copy()` are always available downstream.
    adj: sp.csr_matrix = sp.csr_matrix(adj_raw)

    if not use_weights:
        # Replace all edge weights with 1.0 (matches scanpy behavior)
        adj = adj.copy()
        adj.data = np.ones_like(adj.data, dtype=np.float64)

    # Run the GPU backend. Under the hood it takes a CSR matrix and returns
    # int32 labels per node.
    labels = gpu_leiden.leiden_from_csr(
        adj,
        resolution=float(resolution),
        max_iterations=int(n_iterations),
        random_seed=int(random_state),
    )

    # Store labels as a pandas Categorical (matching scanpy exactly)
    adata.obs[key_added] = pd.Categorical(
        values=labels.astype("U"),
        categories=natsorted(map(str, np.unique(labels))),
    )

    # Record parameters in adata.uns (matching scanpy)
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
    )

    return adata if copy else None
