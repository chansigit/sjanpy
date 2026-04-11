"""GPU-accelerated Leiden community detection, scanpy-compatible.

This module provides :func:`sjanpy.tl.gpuleiden`, a drop-in replacement for
:func:`scanpy.tl.leiden` that uses the ``gpu_leiden`` CUDA backend instead of
leidenalg/igraph. The API follows scanpy's as closely as possible so
existing pipelines can switch by replacing ``sc.tl.leiden`` with
``sjanpy.tl.gpuleiden``.

If ``gpu_leiden`` is not installed (e.g. on a CPU-only machine), importing
this module succeeds but calling :func:`gpuleiden` raises :exc:`ImportError`
with a clear install message. You can gate your code on
``sjanpy.tl.GPU_LEIDEN_AVAILABLE`` to check at import time.
"""
from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted

if TYPE_CHECKING:
    from anndata import AnnData

# Public flag: True only when gpu_leiden is importable.
GPU_LEIDEN_AVAILABLE: bool = importlib.util.find_spec("gpu_leiden") is not None


def _resolve_adjacency(
    adata: "AnnData",
    adjacency: Any,
    neighbors_key: str | None,
    obsp: str | None,
) -> Any:
    """Resolve which adjacency matrix to use (scanpy priority order)."""
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


def gpuleiden(
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
    gpu_flavor: str = "deterministic",
    n_restarts: int = 4,
    temperature: float = 0.5,
    verbose: bool = False,
    **clustering_args: Any,
) -> "AnnData | None":
    """Cluster cells using the GPU Leiden algorithm (drop-in for sc.tl.leiden).

    Requires the ``gpu_leiden`` package (CUDA backend). If it is not installed
    the function raises :exc:`ImportError` immediately — check
    ``sjanpy.tl.GPU_LEIDEN_AVAILABLE`` before calling if you want to branch.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        Resolution (gamma) parameter. Higher values → more, smaller communities.
    random_state
        Random seed. For ``gpu_flavor="deterministic"`` the output is
        bit-reproducible regardless of seed. For ``gpu_flavor="quality"`` the
        seed drives the ILS restart diversity.
    key_added
        ``adata.obs`` column name for the cluster labels.
    adjacency
        Sparse adjacency matrix to cluster on. Defaults to
        ``adata.obsp['connectivities']``.
    directed
        Accepted for API compatibility; the GPU backend always treats the
        graph as undirected/symmetric.
    use_weights
        If ``False``, replace edge weights with 1.0 before running.
    n_iterations
        Number of full Leiden passes (local-moving + refinement +
        aggregation hierarchy). ``-1`` uses the default (2), matching
        leidenalg's ``n_iterations=2``.
    neighbors_key
        Look up the connectivities via
        ``adata.uns[neighbors_key]['connectivities_key']``.
    obsp
        Use ``adata.obsp[obsp]`` directly.
    copy
        Return a modified copy instead of editing in place.
    gpu_flavor
        Algorithm flavor:

        * ``"deterministic"`` (default): bit-reproducible, 3x–12x faster
          than leidenalg, ~95–99% of leidenalg modularity.
        * ``"quality"``: shake-kick ILS over ``n_restarts`` restarts.
          Closes most of the gap to leidenalg at ~3–5x the deterministic
          runtime. Never worse than deterministic.
    n_restarts
        Number of ILS restarts for ``gpu_flavor="quality"``.
    temperature
        Gumbel noise scale for ``gpu_flavor="quality"``. Lower = greedier.
    verbose
        If ``True``, print per-level Leiden progress and kernel timings.
    **clustering_args
        Silently ignored for drop-in compatibility with ``sc.tl.leiden``.

    Returns
    -------
    ``None`` if ``copy=False``, otherwise an :class:`~anndata.AnnData` copy
    with ``obs[key_added]`` set.

    Raises
    ------
    ImportError
        If ``gpu_leiden`` is not installed.
    """
    if not GPU_LEIDEN_AVAILABLE:
        raise ImportError(
            "gpu_leiden is not installed. "
            "Install it with: pip install -e <path-to-gpu-leiden>/python\n"
            "See https://github.com/chansigit/gpu-leiden for build instructions."
        )
    import gpu_leiden  # type: ignore[import-not-found]

    # `directed` and extra kwargs accepted for scanpy API parity but unused.
    del directed, clustering_args

    if copy:
        adata = adata.copy()

    adj_raw = _resolve_adjacency(adata, adjacency, neighbors_key, obsp)
    adj: sp.csr_matrix = sp.csr_matrix(adj_raw)

    if not use_weights:
        adj = adj.copy()
        adj.data = np.ones_like(adj.data, dtype=np.float64)

    labels = gpu_leiden.leiden_from_csr(
        adj,
        resolution=float(resolution),
        max_iterations=int(n_iterations),
        random_seed=int(random_state),
        flavor=gpu_flavor,
        n_restarts=int(n_restarts),
        temperature=float(temperature),
        verbose=verbose,
    )

    adata.obs[key_added] = pd.Categorical(
        values=labels.astype("U"),
        categories=natsorted(map(str, np.unique(labels))),
    )
    adata.uns[key_added] = {
        "params": dict(
            resolution=resolution,
            random_state=random_state,
            n_iterations=n_iterations,
            gpu_flavor=gpu_flavor,
        )
    }

    return adata if copy else None
