"""Evaluation toolkit for single-cell latent embeddings.

Provides GPU-accelerated, self-contained metrics for benchmarking batch
integration and biological conservation in latent spaces — compatible with
the scIB framework (Luecken et al., *Nature Methods* 2022) and the scGraph
benchmark (Wang et al., *Nature Biotechnology* 2025).

Key design choices:

* **No C extensions** — pure Python/NumPy/PyTorch; works on old glibc.
* **GPU-first** — pairwise distances via PyTorch ``matmul``, LISI binary
  search parallelized across all cells on GPU (6x faster than CPU).
* **kNN sharing** — build once with :func:`build_knn_graph`, reuse for
  UMAP, LISI, kBET, graph connectivity, and Leiden clustering.
* **O(n²) isolation** — silhouette scores use a separate subsample
  (default 50k) so kNN-based metrics can run on the full dataset.

Functions
---------
Data loading
    :func:`load_latent`, :func:`load_split_obs`
Subsampling
    :func:`subsample_indices`
kNN graph
    :func:`build_knn_graph` (GPU matmul or pynndescent, auto-selected),
    :func:`knn_to_sparse`
UMAP
    :func:`fit_umap` (precomputed kNN support)
Individual metrics
    :func:`batch_asw`, :func:`celltype_asw`, :func:`graph_connectivity`,
    :func:`lisi`, :func:`ilisi`, :func:`clisi`, :func:`kbet`,
    :func:`leiden_nmi_ari`
Aggregate benchmarks
    :func:`batch_integration_report` — quick ASW + graph + Leiden report
    :func:`scib_metrics` — 9 scIB metrics + overall score (0.4 batch + 0.6 bio)
    :func:`scgraph_score` — Corr-Weighted / Corr-PCA / Rank-PCA
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .h5ad_io import read_obs

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_latent(
    run_dir: str | Path,
    eval_subdir: str = "eval",
) -> np.ndarray:
    """Load latent embeddings from a run directory.

    Checks (in order):
    1. ``<run_dir>/<eval_subdir>/benchmark_qz_mean.npy`` — test-split benchmark latent
    2. ``<run_dir>/<eval_subdir>/analysis_qz_mean.npy`` — test-split analysis latent
    3. ``<run_dir>/train_qz_mean.npy`` + ``<run_dir>/val_qz_mean.npy`` — training-time latent

    Legacy names (``qz_mean.npy``, ``train_latent.npy``, ``val_latent.npy``)
    are also checked for backward compatibility.
    """
    run_dir = Path(run_dir)
    eval_dir = run_dir / eval_subdir

    # Test-split latent (preferred)
    for name in ("benchmark_qz_mean.npy", "analysis_qz_mean.npy", "qz_mean.npy"):
        p = eval_dir / name
        if p.exists():
            return np.load(p)

    # Training-time latent (fallback)
    for train_name, val_name in [
        ("train_qz_mean.npy", "val_qz_mean.npy"),
        ("train_latent.npy", "val_latent.npy"),
    ]:
        train_path = run_dir / train_name
        val_path = run_dir / val_name
        if train_path.exists() and val_path.exists():
            return np.concatenate([np.load(train_path), np.load(val_path)], axis=0)

    raise FileNotFoundError(
        f"No latent embeddings found in {run_dir}. "
        "Expected benchmark_qz_mean.npy, analysis_qz_mean.npy, "
        "or train_qz_mean.npy + val_qz_mean.npy"
    )


def load_split_obs(
    dataset_dir: str | Path,
    splits: tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    """Load and concatenate obs metadata from split h5ad files.

    Args:
        dataset_dir: Directory containing ``{split}.h5ad`` files.
        splits: Which splits to load and concatenate.

    Returns:
        Concatenated obs DataFrame with reset index.
    """
    dataset_dir = Path(dataset_dir)
    frames = []
    for split in splits:
        h5ad_path = dataset_dir / f"{split}.h5ad"
        if h5ad_path.exists():
            frames.append(read_obs(str(h5ad_path)))
    if not frames:
        raise FileNotFoundError(f"No h5ad files found in {dataset_dir}")
    return pd.concat(frames, axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def subsample_indices(
    n: int,
    max_cells: int = 50_000,
    seed: int = 42,
) -> np.ndarray | None:
    """Return random subsample indices, or ``None`` if *n* <= *max_cells*.

    Used to cap O(n^2) operations (silhouette, pairwise distances) while
    letting O(n*k) metrics (LISI, kBET, Leiden) run on the full dataset.
    """
    if n <= max_cells:
        return None
    return np.random.RandomState(seed).choice(n, max_cells, replace=False)


# ---------------------------------------------------------------------------
# kNN graph
# ---------------------------------------------------------------------------


def _gpu_knn_cosine(
    X: np.ndarray,
    k: int,
    chunk_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN via PyTorch matmul on GPU. Handles large datasets via chunking.

    Cosine distance is computed as 1 - (normalized_X @ normalized_X.T).
    Memory: O(chunk_size * n) per chunk, so fits in GPU even for large n.
    """
    import torch

    device = torch.device("cuda")
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    X_t = X_t / X_t.norm(dim=1, keepdim=True)
    n = X_t.shape[0]

    all_indices = torch.empty(n, k, dtype=torch.long, device=device)
    all_values = torch.empty(n, k, dtype=torch.float32, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = X_t[start:end] @ X_t.T  # (chunk, n)
        # Exclude self from neighbors
        sim[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -float("inf")
        values, indices = sim.topk(k, dim=1)
        all_values[start:end] = values
        all_indices[start:end] = indices

    distances = (1 - all_values).cpu().numpy()
    indices = all_indices.cpu().numpy()
    return indices, distances


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "cosine",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Build kNN graph with automatic backend selection.

    Backend priority (when ``backend="auto"``):
      1. **PyTorch GPU** — exact cosine via matmul+topk. Fast, no JIT overhead.
         Used when CUDA is available and data fits in GPU memory.
      2. **pynndescent** — approximate kNN on CPU. Auto-tuned tree count.
         Used as fallback or when metric != "cosine".

    Returns:
        (indices, distances) — both shape ``(n_samples, n_neighbors)``.
    """
    n = X.shape[0]

    # GPU path: PyTorch matmul (exact, cosine only)
    if metric == "cosine" and backend in ("auto", "gpu"):
        try:
            import torch

            if torch.cuda.is_available():
                # Memory estimate: (chunk_size * n * 4 bytes) for similarity matrix
                # With chunking, always fits; but skip GPU for trivially small data
                return _gpu_knn_cosine(X, n_neighbors)
        except ImportError:
            pass
        if backend == "gpu":
            raise RuntimeError("GPU backend requested but torch+CUDA not available")

    # CPU path: pynndescent (approximate, supports many metrics)
    from pynndescent import NNDescent

    if n < 20_000:
        n_trees = 8
    elif n < 100_000:
        n_trees = 16
    else:
        n_trees = None  # pynndescent default

    index = NNDescent(
        X, metric=metric, n_neighbors=n_neighbors,
        n_trees=n_trees, diversify_prob=0.5, low_memory=(n > 100_000),
    )
    indices, distances = index.neighbor_graph
    return indices, distances


def knn_to_sparse(
    indices: np.ndarray,
    distances: np.ndarray,
) -> sparse.csr_matrix:
    """Convert kNN (indices, distances) arrays to a sparse connectivity matrix.

    Returns:
        Sparse CSR matrix of shape ``(n, n)`` with binary connectivity
        (1 where i→j is a neighbor, 0 otherwise). Self-loops are excluded.
    """
    n = indices.shape[0]
    k = indices.shape[1]

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()

    # Filter out self-loops and invalid (-1) entries
    valid = (cols >= 0) & (cols != rows)
    rows = rows[valid]
    cols = cols[valid]

    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def fit_umap(
    X: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "cosine",
    precomputed_knn: tuple[np.ndarray, np.ndarray] | None = None,
    n_components: int = 2,
    min_dist: float = 0.3,
    n_jobs: int = -1,
) -> np.ndarray:
    """Fit UMAP, optionally reusing a precomputed kNN graph.

    Args:
        X: Input data, shape ``(n_samples, n_features)``.
        precomputed_knn: Tuple of ``(indices, distances)`` from
            :func:`build_knn_graph`. Skips internal kNN computation.

    Returns:
        UMAP coordinates, shape ``(n_samples, n_components)``.
    """
    import umap as _umap

    kwargs = dict(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        n_jobs=n_jobs,
    )
    if precomputed_knn is not None:
        kwargs["precomputed_knn"] = precomputed_knn
    reducer = _umap.UMAP(**kwargs)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Batch integration metrics
# ---------------------------------------------------------------------------


def _pairwise_distances(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Precompute pairwise distance matrix. Uses GPU (PyTorch) when available."""
    if metric == "cosine":
        try:
            import torch

            if torch.cuda.is_available():
                X_t = torch.from_numpy(np.ascontiguousarray(X)).float().cuda()
                X_t = X_t / X_t.norm(dim=1, keepdim=True)
                dist = (1 - X_t @ X_t.T).clamp(min=0).cpu().numpy()
                return dist
        except ImportError:
            pass
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(X, metric=metric)


def batch_asw(
    X: np.ndarray, labels, metric: str = "cosine",
    precomputed_dist: np.ndarray | None = None,
) -> float | None:
    """Average silhouette width by batch label (lower is better mixing).

    Accepts a precomputed distance matrix to avoid redundant O(n^2) work
    when computing both batch and cell-type ASW.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    if precomputed_dist is not None:
        return float(silhouette_score(precomputed_dist, labels, metric="precomputed"))
    return float(silhouette_score(X, labels, metric=metric))


def celltype_asw(
    X: np.ndarray, labels, metric: str = "cosine",
    precomputed_dist: np.ndarray | None = None,
) -> float | None:
    """Average silhouette width by cell type (higher is better conservation).

    Same interface as :func:`batch_asw`; see that function for details.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    if precomputed_dist is not None:
        return float(silhouette_score(precomputed_dist, labels, metric="precomputed"))
    return float(silhouette_score(X, labels, metric=metric))


def graph_connectivity(
    knn_sparse: sparse.csr_matrix,
    labels,
) -> float | None:
    """Graph connectivity per label (scIB-compatible).

    For each label, computes the fraction of cells in the largest connected
    component of the kNN subgraph restricted to that label. Returns the
    weighted average across labels. Higher → better (all cells of each type
    are reachable via same-type neighbors).
    """
    from scipy.sparse.csgraph import connected_components

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    total_score = 0.0
    total_cells = 0
    # Symmetrize for undirected connectivity
    graph = knn_sparse + knn_sparse.T

    for lab in unique_labels:
        mask = labels == lab
        n_lab = int(mask.sum())
        idx = np.where(mask)[0]
        # Subgraph for this label
        subgraph = graph[np.ix_(idx, idx)]
        n_components, comp_labels = connected_components(subgraph, directed=False)
        if n_components == 0:
            continue
        largest_cc = np.bincount(comp_labels).max()
        total_score += largest_cc
        total_cells += n_lab

    return float(total_score / total_cells) if total_cells > 0 else 0.0


def lisi(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    labels,
    perplexity: float | None = None,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute Local Inverse Simpson's Index (LISI) for each cell.

    Reimplements the algorithm from Korsunsky et al. 2019 (Harmony) and
    the scIB benchmark. Uses perplexity-based Gaussian kernel weighting
    (same as t-SNE) to convert kNN distances to probabilities, then
    computes the inverse Simpson's index on the weighted label frequencies.

    Args:
        knn_indices: (n, k) array of neighbor indices.
        knn_distances: (n, k) array of neighbor distances.
        labels: (n,) array of categorical labels.
        perplexity: Effective neighborhood size. Default: k // 3.
        tol: Tolerance for binary search convergence.

    Returns:
        (n,) array of per-cell LISI scores.
    """
    labels = np.asarray(labels)
    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    n_categories = len(unique_labels)
    n, k = knn_indices.shape

    if perplexity is None:
        perplexity = k // 3

    nb_labels = label_codes[knn_indices]  # (n, k)

    # Self-mask: exclude self from neighbors
    self_mask = knn_indices != np.arange(n)[:, None]  # (n, k)

    simpson = _compute_simpson_batch(
        knn_distances, nb_labels, self_mask, n_categories, float(perplexity), tol,
    )
    # LISI = 1 / simpson; handle edge cases
    simpson = np.clip(simpson, 1e-12, None)
    return 1.0 / simpson


def _compute_simpson_batch(
    knn_dists: np.ndarray,
    knn_labels: np.ndarray,
    self_mask: np.ndarray,
    n_categories: int,
    perplexity: float,
    tol: float,
    max_iter: int = 50,
) -> np.ndarray:
    """Simpson index with perplexity-based Gaussian kernel.

    Uses GPU (PyTorch) when available for batch-parallel binary search.
    Falls back to CPU loop otherwise.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return _compute_simpson_gpu(
                knn_dists, knn_labels, self_mask, n_categories, perplexity, tol, max_iter,
            )
    except ImportError:
        pass

    return _compute_simpson_cpu(
        knn_dists, knn_labels, self_mask, n_categories, perplexity, tol, max_iter,
    )


def _compute_simpson_cpu(
    knn_dists: np.ndarray,
    knn_labels: np.ndarray,
    self_mask: np.ndarray,
    n_categories: int,
    perplexity: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """CPU fallback: per-cell loop."""
    n, k = knn_dists.shape
    logU = np.log(perplexity)
    simpson = np.ones(n, dtype=np.float64)

    for i in range(n):
        D = knn_dists[i].astype(np.float64)
        mask = self_mask[i]
        beta, betamin, betamax = 1.0, -np.inf, np.inf
        sumP = 0.0

        for _ in range(max_iter):
            P = np.exp(-D * beta)
            P[~mask] = 0
            sumP = P.sum()
            if sumP == 0:
                break
            H = np.log(sumP) + beta * np.dot(D, P) / sumP
            P /= sumP
            Hdiff = H - logU
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2

        if sumP == 0:
            simpson[i] = 1.0
            continue
        batch_freq = np.bincount(knn_labels[i], weights=P, minlength=n_categories)
        simpson[i] = np.dot(batch_freq, batch_freq)

    return simpson


def _compute_simpson_gpu(
    knn_dists: np.ndarray,
    knn_labels: np.ndarray,
    self_mask: np.ndarray,
    n_categories: int,
    perplexity: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """GPU batch-parallel binary search for perplexity-based Simpson index."""
    import torch

    device = torch.device("cuda")
    D = torch.from_numpy(knn_dists.astype(np.float64)).to(device)
    labels = torch.from_numpy(knn_labels.astype(np.int64)).to(device)
    mask = torch.from_numpy(self_mask).to(device)
    n, k = D.shape
    logU = np.log(perplexity)

    beta = torch.ones(n, device=device, dtype=torch.float64)
    betamin = torch.full((n,), -float("inf"), device=device, dtype=torch.float64)
    betamax = torch.full((n,), float("inf"), device=device, dtype=torch.float64)

    for _ in range(max_iter):
        P = torch.exp(-D * beta[:, None])
        P[~mask] = 0
        sumP = P.sum(dim=1)
        safe = sumP.clamp(min=1e-300)
        H = torch.log(safe) + beta * (D * P).sum(dim=1) / safe
        H[sumP == 0] = 0
        Hdiff = H - logU

        go_up = Hdiff > 0
        go_down = (Hdiff < -tol) & ~go_up
        betamin = torch.where(go_up, beta, betamin)
        betamax = torch.where(go_down, beta, betamax)
        beta = torch.where(
            go_up,
            torch.where(betamax == float("inf"), beta * 2, (beta + betamax) / 2),
            torch.where(
                go_down,
                torch.where(betamin == -float("inf"), beta / 2, (beta + betamin) / 2),
                beta,
            ),
        )

    # Final probabilities
    P = torch.exp(-D * beta[:, None])
    P[~mask] = 0
    P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-300)

    # Simpson index via one-hot weighted sum
    one_hot = torch.zeros(n, k, n_categories, device=device, dtype=torch.float64)
    one_hot.scatter_(2, labels.unsqueeze(2), 1.0)
    freq = (P.unsqueeze(2) * one_hot).sum(dim=1)  # (n, n_cat)
    simpson = (freq**2).sum(dim=1)

    return simpson.cpu().numpy()


def ilisi(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    batch_labels,
    perplexity: float | None = None,
    scale: bool = True,
) -> float | None:
    """Integration LISI (batch mixing). Higher → better batch integration.

    Returns median iLISI, optionally scaled to [0, 1] where
    1 = perfect mixing (LISI = n_batches).
    """
    batch_labels = np.asarray(batch_labels)
    n_batches = len(np.unique(batch_labels))
    if n_batches < 2:
        return None
    scores = lisi(knn_indices, knn_distances, batch_labels, perplexity)
    median = float(np.nanmedian(scores))
    if scale:
        return float(np.clip((median - 1) / (n_batches - 1), 0, 1))
    return median


def clisi(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    celltype_labels,
    perplexity: float | None = None,
    scale: bool = True,
) -> float | None:
    """Cell type LISI (bio conservation). Higher → better cell type purity.

    Returns median cLISI, optionally scaled to [0, 1] where
    1 = perfect purity (LISI = 1, only one cell type per neighborhood).
    """
    celltype_labels = np.asarray(celltype_labels)
    n_types = len(np.unique(celltype_labels))
    if n_types < 2:
        return None
    scores = lisi(knn_indices, knn_distances, celltype_labels, perplexity)
    median = float(np.nanmedian(scores))
    if scale:
        return float(np.clip((n_types - median) / (n_types - 1), 0, 1))
    return median


def kbet(
    knn_indices: np.ndarray,
    batch_labels,
    alpha: float = 0.05,
) -> float | None:
    """k-nearest-neighbor Batch Effect Test (kBET).

    For each cell, tests whether the batch composition in its k-neighborhood
    matches the global batch frequencies using a chi-squared test.
    Returns the acceptance rate (fraction of cells that pass the test).

    Higher → better batch mixing (more cells have expected batch composition).

    Args:
        knn_indices: (n, k) array of neighbor indices.
        batch_labels: (n,) array of batch labels.
        alpha: Significance level for the chi-squared test.

    Returns:
        Acceptance rate in [0, 1]. Higher is better.
    """
    from scipy.stats import chi2

    batch_labels = np.asarray(batch_labels)
    unique_batches, batch_codes = np.unique(batch_labels, return_inverse=True)
    n_batches = len(unique_batches)
    if n_batches < 2:
        return None

    n, k = knn_indices.shape

    # Global batch frequencies
    global_freq = np.bincount(batch_codes, minlength=n_batches) / n

    # Degrees of freedom for chi-squared test
    df = n_batches - 1
    chi2_threshold = chi2.ppf(1 - alpha, df)

    # Vectorized: one-hot encode, compute observed counts, chi-squared test
    nb_batches = batch_codes[knn_indices]  # (n, k)
    one_hot = np.eye(n_batches, dtype=np.float32)[nb_batches]  # (n, k, n_batches)
    observed = one_hot.sum(axis=1)  # (n, n_batches)
    expected = global_freq * k  # (n_batches,)

    # Chi-squared: sum((O - E)^2 / E) for each cell, only where E > 0
    mask = expected > 0
    chi2_stats = ((observed[:, mask] - expected[mask]) ** 2 / expected[mask]).sum(axis=1)
    accepted = (chi2_stats <= chi2_threshold).sum()

    return float(accepted / n)


def leiden_nmi_ari(
    knn_sparse: sparse.csr_matrix,
    true_labels,
    resolution: float = 1.0,
) -> dict:
    """Run Leiden clustering on kNN graph and compute NMI/ARI vs true labels.

    Returns:
        Dict with keys: nmi, ari, n_leiden_clusters, n_true_labels, pred_labels.
    """
    import anndata as ad
    import scanpy as sc

    true_labels = np.asarray(true_labels)
    n = knn_sparse.shape[0]

    # Build minimal AnnData with precomputed kNN
    adata = ad.AnnData(X=sparse.csr_matrix((n, 1)))
    adata.obsp["connectivities"] = knn_sparse
    # Leiden also needs distances — use 1-connectivity as proxy
    adata.obsp["distances"] = knn_sparse.copy()
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"method": "pynndescent"},
    }

    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    pred = adata.obs["leiden"].values

    return {
        "nmi": float(normalized_mutual_info_score(true_labels, pred)),
        "ari": float(adjusted_rand_score(true_labels, pred)),
        "n_leiden_clusters": int(len(np.unique(pred))),
        "n_true_labels": int(len(np.unique(true_labels))),
        "pred_labels": pred,
    }


def batch_integration_report(
    X: np.ndarray,
    obs: pd.DataFrame,
    knn_indices: np.ndarray | None = None,
    knn_distances: np.ndarray | None = None,
    n_neighbors: int = 30,
    metric: str = "cosine",
    resolution: float = 1.0,
    batch_key: str = "batch",
    celltype_key: str = "cell_type",
    max_cells_silhouette: int = 50_000,
) -> dict:
    """Compute all batch integration metrics in one call.

    kNN-based metrics (graph_connectivity, Leiden NMI/ARI) run on the full
    input. Silhouette scores use a separate subsample capped at
    ``max_cells_silhouette`` to avoid O(n^2) memory blowup.

    If knn_indices/knn_distances are not provided, builds kNN internally.

    Returns:
        JSON-serializable dict with all metrics.
    """
    n = len(X)
    if knn_indices is None or knn_distances is None:
        knn_indices, knn_distances = build_knn_graph(X, n_neighbors, metric)

    knn_sparse = knn_to_sparse(knn_indices, knn_distances)

    metrics: dict = {}

    # --- Silhouette (O(n²) pairwise) on separate subsample ---
    need_batch_asw = batch_key in obs.columns and obs[batch_key].nunique() > 1
    need_ct_asw = celltype_key in obs.columns and obs[celltype_key].nunique() > 1

    if need_batch_asw or need_ct_asw:
        sil_idx = subsample_indices(n, max_cells=max_cells_silhouette)
        if sil_idx is not None:
            sil_X = X[sil_idx]
            sil_obs = obs.iloc[sil_idx]
        else:
            sil_X = X
            sil_obs = obs
        dist_matrix = _pairwise_distances(sil_X, metric)
        metrics["n_cells_silhouette"] = int(len(sil_X))
    else:
        sil_obs = obs
        dist_matrix = None
        metrics["n_cells_silhouette"] = 0

    if need_batch_asw:
        metrics["batch_asw"] = batch_asw(
            None, sil_obs[batch_key].values, metric, precomputed_dist=dist_matrix
        )
    else:
        metrics["batch_asw"] = None

    if need_ct_asw:
        metrics["celltype_asw"] = celltype_asw(
            None, sil_obs[celltype_key].values, metric, precomputed_dist=dist_matrix
        )
    else:
        metrics["celltype_asw"] = None

    del dist_matrix  # free O(n²) memory immediately

    # --- kNN-based metrics (O(n·k), scale to millions) ---
    if batch_key in obs.columns:
        metrics["graph_connectivity"] = graph_connectivity(
            knn_sparse, obs[batch_key].values
        )
    else:
        metrics["graph_connectivity"] = None

    if celltype_key in obs.columns:
        leiden_result = leiden_nmi_ari(knn_sparse, obs[celltype_key].values, resolution)
        metrics["nmi"] = leiden_result["nmi"]
        metrics["ari"] = leiden_result["ari"]
        metrics["n_leiden_clusters"] = leiden_result["n_leiden_clusters"]
        metrics["n_cell_types"] = leiden_result["n_true_labels"]
    else:
        metrics["nmi"] = None
        metrics["ari"] = None

    # Counts
    metrics["n_cells_eval"] = int(n)
    if batch_key in obs.columns:
        metrics["n_batches"] = int(obs[batch_key].nunique())

    return metrics


# ---------------------------------------------------------------------------
# scIB-compatible full benchmark
# ---------------------------------------------------------------------------


def scib_metrics(
    X: np.ndarray,
    obs: pd.DataFrame,
    knn_indices: np.ndarray | None = None,
    knn_distances: np.ndarray | None = None,
    n_neighbors: int = 30,
    metric: str = "cosine",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    max_cells_silhouette: int = 50_000,
    resolution: float = 1.0,
) -> dict:
    """Compute scIB-compatible metrics from latent embeddings + obs.

    Self-contained reimplementation of the scIB benchmark (Luecken et al.,
    *Nature Methods* 2022). No C extensions or JAX required — works on any
    platform with NumPy and (optionally) PyTorch+CUDA.

    Computes 9 of the 14 scIB metrics — the ones derivable from a latent
    embedding and cell metadata alone:

    **Batch correction** (4 of 5; PCR skipped — needs unintegrated adata):
        ``batch_asw``, ``graph_connectivity``, ``kBET``, ``iLISI``

    **Bio conservation** (5 of 9; cell cycle, HVG overlap, isolated F1,
    trajectory skipped — need extra annotations):
        ``celltype_asw``, ``NMI``, ``ARI``, ``cLISI``, ``isolated_label_asw``

    **Overall score** = 0.4 * mean(batch metrics) + 0.6 * mean(bio metrics)

    Performance (14.7k cells, H100):
        ~5s total. LISI binary search runs on GPU; pairwise distances for
        silhouette computed via GPU matmul; kBET fully vectorized.

    Example::

        from sjanpy.ml.eval import load_latent, load_split_obs, scib_metrics

        latent = load_latent("outputs/my_run")
        obs = load_split_obs("data/ds_skin")
        results = scib_metrics(latent, obs)
        print(results["scib_overall"])  # 0.4 * batch + 0.6 * bio

    Args:
        X: Latent embedding, shape ``(n_cells, n_dims)``.
        obs: Cell metadata with ``batch_key`` and ``label_key`` columns.
        knn_indices: Precomputed kNN indices from :func:`build_knn_graph`.
        knn_distances: Precomputed kNN distances.
        n_neighbors: Number of neighbors (used only if kNN not provided).
        metric: Distance metric for kNN and silhouette.
        batch_key: Column name for batch labels in *obs*.
        label_key: Column name for cell-type labels in *obs*.
        max_cells_silhouette: Cap for silhouette subsample (O(n^2) guard).
        resolution: Leiden clustering resolution.

    Returns:
        JSON-serializable dict with all metrics, sub-scores, and metadata.
    """
    n = len(X)

    # Build kNN if not provided
    if knn_indices is None or knn_distances is None:
        knn_indices, knn_distances = build_knn_graph(X, n_neighbors, metric)
    knn_sparse = knn_to_sparse(knn_indices, knn_distances)

    results: dict = {}
    has_batch = batch_key in obs.columns and obs[batch_key].nunique() > 1
    has_label = label_key in obs.columns and obs[label_key].nunique() > 1

    # === Silhouette (O(m²) on subsample, GPU pairwise) ===
    sil_idx = subsample_indices(n, max_cells=max_cells_silhouette)
    sil_X = X[sil_idx] if sil_idx is not None else X
    sil_obs = obs.iloc[sil_idx] if sil_idx is not None else obs
    dist_matrix = _pairwise_distances(sil_X, metric) if (has_batch or has_label) else None

    results["batch_asw"] = batch_asw(
        None, sil_obs[batch_key].values, metric, precomputed_dist=dist_matrix
    ) if has_batch else None

    results["celltype_asw"] = celltype_asw(
        None, sil_obs[label_key].values, metric, precomputed_dist=dist_matrix
    ) if has_label else None

    # Isolated label ASW: silhouette for cell types present in only few batches
    if has_label and has_batch and dist_matrix is not None:
        batch_per_type = obs.groupby(label_key)[batch_key].nunique()
        n_batches_total = obs[batch_key].nunique()
        isolated_types = batch_per_type[batch_per_type < n_batches_total].index.tolist()
        if isolated_types:
            iso_mask = sil_obs[label_key].isin(isolated_types).values
            if iso_mask.sum() > 10:
                iso_dist = dist_matrix[np.ix_(iso_mask, iso_mask)]
                iso_labels = sil_obs.loc[iso_mask, label_key].values
                n_unique = len(np.unique(iso_labels))
                if n_unique >= 2:
                    results["isolated_label_asw"] = float(
                        silhouette_score(iso_dist, iso_labels, metric="precomputed")
                    )
                else:
                    results["isolated_label_asw"] = np.nan
            else:
                results["isolated_label_asw"] = np.nan
        else:
            results["isolated_label_asw"] = np.nan
    else:
        results["isolated_label_asw"] = np.nan

    del dist_matrix  # free O(m²) memory

    # === kNN-based metrics (O(n·k), vectorized) ===

    results["graph_connectivity"] = graph_connectivity(
        knn_sparse, obs[batch_key].values
    ) if has_batch else None

    results["kBET"] = kbet(knn_indices, obs[batch_key].values) if has_batch else None
    results["iLISI"] = ilisi(knn_indices, knn_distances, obs[batch_key].values) if has_batch else None
    results["cLISI"] = clisi(knn_indices, knn_distances, obs[label_key].values) if has_label else None

    # Leiden → NMI, ARI
    if has_label:
        leiden_result = leiden_nmi_ari(knn_sparse, obs[label_key].values, resolution)
        results["nmi"] = leiden_result["nmi"]
        results["ari"] = leiden_result["ari"]
        results["n_leiden_clusters"] = leiden_result["n_leiden_clusters"]
    else:
        results["nmi"] = None
        results["ari"] = None

    # === Overall score (Luecken 2022) ===
    # Overall = 0.4 * mean(batch metrics) + 0.6 * mean(bio metrics)
    batch_keys = ["batch_asw", "graph_connectivity", "kBET", "iLISI"]
    bio_keys = ["celltype_asw", "nmi", "ari", "cLISI", "isolated_label_asw"]

    def _valid(keys):
        return [results[k] for k in keys
                if results.get(k) is not None and not np.isnan(results.get(k, np.nan))]

    batch_vals = _valid(batch_keys)
    bio_vals = _valid(bio_keys)

    batch_score = float(np.mean(batch_vals)) if batch_vals else np.nan
    bio_score = float(np.mean(bio_vals)) if bio_vals else np.nan
    results["batch_score"] = batch_score
    results["bio_score"] = bio_score
    results["scib_overall"] = float(0.4 * batch_score + 0.6 * bio_score) if (
        not np.isnan(batch_score) and not np.isnan(bio_score)
    ) else np.nan

    # Metadata
    results["n_cells"] = n
    results["n_cells_silhouette"] = int(len(sil_X))
    results["n_batches"] = int(obs[batch_key].nunique()) if has_batch else 0
    results["n_cell_types"] = int(obs[label_key].nunique()) if has_label else 0
    results["n_metrics_batch"] = len(batch_vals)
    results["n_metrics_bio"] = len(bio_vals)

    return results


# ---------------------------------------------------------------------------
# scGraph (Islander) metrics
# ---------------------------------------------------------------------------


def scgraph_score(
    latent: np.ndarray,
    adata_path: str | Path | None = None,
    adata=None,
    batch_key: str = "batch",
    label_key: str = "cell_type",
    trim_rate: float = 0.05,
    thres_batch: int = 100,
    thres_celltype: int = 10,
) -> dict:
    """Compute scGraph embedding quality metrics (Wang et al., Nature Biotech 2025).

    Self-contained reimplementation (no scgraph_bench dependency). Evaluates
    how well an embedding preserves cell-type relationships by comparing
    pairwise centroid distances against a PCA-based consensus built
    independently per batch.

    Requires the original gene expression matrix to build the PCA consensus.
    Provide either ``adata`` or ``adata_path``.

    Args:
        latent: Latent embedding, shape ``(n_cells, n_dims)``.
        adata_path: Path to h5ad file(s). Can also be a directory containing
            ``train.h5ad`` + ``val.h5ad``.
        adata: AnnData with expression in ``.X`` and obs metadata.
        batch_key: Column name for batch information.
        label_key: Column name for cell type labels.
        trim_rate: Trim proportion for robust centroid calculation.
        thres_batch: Minimum cells per batch.
        thres_celltype: Minimum cells per cell type.

    Returns:
        Dict with ``corr_weighted`` (main metric), ``corr_pca``,
        ``rank_pca``.
    """
    import scanpy as sc
    from scipy.spatial.distance import cdist
    from scipy.stats import trim_mean

    # --- Load data ---
    if adata is None:
        if adata_path is None:
            raise ValueError("Provide either adata or adata_path")
        p = Path(adata_path)
        if p.is_dir():
            import anndata as ad
            parts = [sc.read_h5ad(str(p / f"{s}.h5ad")) for s in ("train", "val") if (p / f"{s}.h5ad").exists()]
            adata = ad.concat(parts)
        else:
            adata = sc.read_h5ad(str(p))
    else:
        adata = adata.copy()

    # Normalize if raw counts
    x = adata.X
    xmax = x.data.max() if sparse.issparse(x) and x.nnz > 0 else (x.max() if not sparse.issparse(x) else 0)
    if xmax > 20:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    labels = adata.obs[label_key]
    batches = adata.obs[batch_key]

    # Filter rare cell types
    ct_counts = labels.value_counts()
    ignore_ct = set(ct_counts[ct_counts < thres_celltype].index)

    # --- Helpers ---
    def _trimmed_centroids(X, labs):
        if sparse.issparse(X):
            X = X.toarray()
        centroids = {}
        for lab in sorted(labs.unique()):
            if lab in ignore_ct:
                continue
            centroids[lab] = trim_mean(X[labs == lab], proportiontocut=trim_rate, axis=0)
        return centroids

    def _pairwise_dist_df(centroids):
        keys = sorted(centroids.keys())
        vecs = np.array([centroids[k] for k in keys])
        dist = cdist(vecs, vecs, "euclidean")
        df = pd.DataFrame(dist, index=keys, columns=keys)
        return df.div(df.max(axis=0), axis=1)

    # --- Build PCA consensus (per-batch HVG → PCA → centroids → distances) ---
    pca_dists = {}
    for batch in batches.unique():
        adata_b = adata[batches == batch].copy()
        if len(adata_b) < thres_batch:
            continue
        sc.pp.highly_variable_genes(adata_b, n_top_genes=min(1000, adata_b.n_vars))
        sc.pp.pca(adata_b, n_comps=10, use_highly_variable=True)
        centroids = _trimmed_centroids(adata_b.obsm["X_pca"], adata_b.obs[label_key])
        if len(centroids) < 2:
            continue
        pca_dists[batch] = _pairwise_dist_df(centroids)

    if not pca_dists:
        return {"corr_weighted": np.nan, "corr_pca": np.nan, "rank_pca": np.nan}

    # Average across batches
    consensus = pd.concat(pca_dists.values()).groupby(level=0).mean()
    consensus = consensus.loc[consensus.columns, :]
    consensus = consensus.div(consensus.max(axis=0), axis=1)

    # --- Evaluate embedding ---
    emb_centroids = _trimmed_centroids(latent[:len(adata)], labels)
    emb_dist = _pairwise_dist_df(emb_centroids)

    # Align columns
    shared = sorted(set(consensus.columns) & set(emb_dist.columns))
    if len(shared) < 2:
        return {"corr_weighted": np.nan, "corr_pca": np.nan, "rank_pca": np.nan}
    cons = consensus.loc[shared, shared]
    emb = emb_dist.loc[shared, shared]

    # --- Compute correlations (per cell-type column, then average) ---
    rank_pca, corr_pca, corr_w = [], [], []
    for col in shared:
        c, e = cons[col].dropna(), emb[col].dropna()
        common = c.index.intersection(e.index)
        if len(common) < 2:
            continue
        cv, ev = c[common], e[common]
        rank_pca.append(cv.corr(ev, method="spearman"))
        corr_pca.append(cv.corr(ev, method="pearson"))
        # Weighted Pearson (weight = 1/distance from consensus)
        d = cv.values.astype(float)
        w = np.where(d > 0, 1.0 / d, 0.0)
        w /= w.sum() if w.sum() > 0 else 1.0
        mx = np.average(cv, weights=w)
        my = np.average(ev, weights=w)
        cov = np.sum(w * (cv - mx) * (ev - my))
        vx = np.sum(w * (cv - mx) ** 2)
        vy = np.sum(w * (ev - my) ** 2)
        corr_w.append(cov / np.sqrt(vx * vy) if vx * vy > 0 else 0.0)

    return {
        "corr_weighted": float(np.nanmean(corr_w)) if corr_w else np.nan,
        "corr_pca": float(np.nanmean(corr_pca)) if corr_pca else np.nan,
        "rank_pca": float(np.nanmean(rank_pca)) if rank_pca else np.nan,
    }
