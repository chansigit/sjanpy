"""
Dataset classes for loading training data.

Supports two on-disk formats (auto-detected):
  - safetensors: single {split}.safetensors file per split (fast, DMA-friendly)
  - pt_chunks:   multiple chunk_NNNN.pt files per split (legacy)

Three dataset classes for different memory/speed trade-offs:
  - GPUDataset:       Preload entire dataset to GPU. Zero transfer during training.
  - ChunkedDataset:   Load all data to CPU RAM. DataLoader handles GPU transfer.
  - StreamingDataset: mmap safetensors or stream chunks. For datasets that don't fit in RAM.
"""

import json
import threading
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Format detection and loading helpers
# ---------------------------------------------------------------------------

def _find_safetensors(chunks_dir: Path, split: str):
    """Find safetensors file for a split. Returns path or None."""
    # {chunks_dir}/{split}/{split}.safetensors (new layout)
    p = chunks_dir / split / f"{split}.safetensors"
    if p.exists():
        return p
    # {chunks_dir}/{split}.safetensors (flat layout)
    p = chunks_dir / f"{split}.safetensors"
    if p.exists():
        return p
    return None


def _find_chunk_files(chunks_dir: Path, split: str):
    """Return sorted list of .pt chunk files for a given split, or empty list."""
    subdir = chunks_dir / split
    if subdir.is_dir():
        files = sorted(subdir.glob("chunk_*.pt"))
    else:
        files = sorted(chunks_dir.glob(f"{split}_*.pt"))
    return files


def _load_metadata(chunks_dir: Path, split: str) -> dict:
    """Load metadata, supporting per-split and root-level formats."""
    for path in [
        chunks_dir / split / "metadata.json",
        chunks_dir / "metadata.json",
    ]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"No metadata.json found in {chunks_dir}")


def _load_safetensors(path, device=None):
    """Load a safetensors file. Returns (counts, condition, labels)."""
    from safetensors.torch import load_file
    device_str = str(device) if device else "cpu"
    tensors = load_file(str(path), device=device_str)
    return tensors["counts"], tensors["condition"], tensors["labels"]


def _load_safetensors_or_chunks(chunks_dir, split, device=None, dtype=None):
    """Load data from safetensors (preferred) or .pt chunks (fallback).

    Returns (counts, condition, labels) tensors.
    """
    st_path = _find_safetensors(chunks_dir, split)
    if st_path is not None:
        counts, condition, labels = _load_safetensors(st_path, device=device)
        if dtype is not None and counts.dtype != dtype:
            counts = counts.to(dtype)
        return counts, condition, labels

    # Fallback: .pt chunks
    chunk_files = _find_chunk_files(chunks_dir, split)
    if not chunk_files:
        raise FileNotFoundError(
            f"No safetensors or chunk files for split '{split}' in {chunks_dir}"
        )
    chunks = [torch.load(f, weights_only=True) for f in chunk_files]
    counts = torch.cat([c["counts"] for c in chunks])
    condition = torch.cat([c["condition"] for c in chunks])
    labels = torch.cat([c["labels"] for c in chunks])
    if dtype is not None:
        counts = counts.to(dtype)
    if device is not None:
        counts = counts.to(device)
        condition = condition.to(device)
        labels = labels.to(device)
    return counts, condition, labels


def _load_single_chunk(path, dtype=None):
    """Load a single .pt chunk file."""
    c = torch.load(path, weights_only=True)
    counts = c["counts"].to(dtype) if dtype is not None else c["counts"]
    return counts, c["condition"], c["labels"]


# ---------------------------------------------------------------------------
# GPUDataset: entire dataset on GPU, zero transfer during training
# ---------------------------------------------------------------------------

class GPUDataset(torch.utils.data.Dataset):
    """Preload entire dataset to GPU for zero-transfer training.

    Tries safetensors first (DMA load in one call), falls back to .pt chunks.

    Args:
        chunks_dir: Path to data directory.
        split: "train", "val", or "test".
        device: Target device (e.g. "cuda").
        dtype: Optional dtype for counts (e.g. torch.bfloat16). None = keep as-is.
    """

    def __init__(self, chunks_dir, split: str = "train", device="cuda", dtype=None):
        self.chunks_dir = Path(chunks_dir)
        self.metadata = _load_metadata(self.chunks_dir, split)
        self.counts, self.condition, self.labels = _load_safetensors_or_chunks(
            self.chunks_dir, split, device=device, dtype=dtype,
        )

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        return self.counts[idx], self.condition[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# StreamingDataset: mmap safetensors or chunk-level streaming
# ---------------------------------------------------------------------------

class StreamingDataset(torch.utils.data.IterableDataset):
    """Stream data without loading everything into RAM.

    With safetensors: mmap the file, slice row ranges on demand.
    With pt_chunks: load one chunk at a time, prefetch next in background.

    Args:
        chunks_dir: Path to data directory.
        split: "train", "val", or "test".
        dtype: Optional dtype for counts.
        shuffle: Shuffle chunk/row order each epoch.
        seed: Random seed (combined with epoch for determinism).
    """

    def __init__(self, chunks_dir, split: str = "train", dtype=None,
                 shuffle: bool = True, seed: int = 42):
        self.chunks_dir = Path(chunks_dir)
        self.metadata = _load_metadata(self.chunks_dir, split)
        self.split = split
        self.dtype = dtype
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Detect format
        self._st_path = _find_safetensors(self.chunks_dir, split)
        if self._st_path is not None:
            self._mode = "safetensors"
            self._total = self.metadata.get("n_cells", 0)
            if self._total == 0:
                # Fallback: read shape from file
                from safetensors import safe_open
                with safe_open(str(self._st_path), framework="pt") as f:
                    self._total = f.get_tensor("counts").shape[0]
        else:
            self._mode = "pt_chunks"
            self.chunk_files = _find_chunk_files(self.chunks_dir, split)
            if not self.chunk_files:
                raise FileNotFoundError(
                    f"No safetensors or chunk files for split '{split}' in {self.chunks_dir}"
                )
            chunk_sizes = self.metadata.get("chunk_sizes")
            if chunk_sizes is None:
                split_info = self.metadata.get("splits", {}).get(split, {})
                chunk_sizes = split_info.get("chunk_sizes")
            if chunk_sizes is not None:
                self._total = sum(chunk_sizes)
            else:
                self._total = sum(
                    len(torch.load(f, weights_only=True)["counts"])
                    for f in self.chunk_files
                )

    def __len__(self):
        return self._total

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self._mode == "safetensors":
            yield from self._iter_safetensors()
        else:
            yield from self._iter_chunks()

    def _iter_safetensors(self):
        """mmap safetensors file, yield shuffled rows."""
        from safetensors import safe_open
        rng = np.random.default_rng(self.seed + self.epoch)

        with safe_open(str(self._st_path), framework="pt") as f:
            counts = f.get_tensor("counts")
            condition = f.get_tensor("condition")
            labels = f.get_tensor("labels")

        n = len(counts)
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        if self.dtype is not None and counts.dtype != self.dtype:
            counts = counts.to(self.dtype)

        for i in order:
            yield counts[i], condition[i], labels[i]

    def _iter_chunks(self):
        """Stream .pt chunk files with background prefetch."""
        rng = np.random.default_rng(self.seed + self.epoch)

        chunk_order = np.arange(len(self.chunk_files))
        if self.shuffle:
            rng.shuffle(chunk_order)

        # Prefetch first chunk
        prefetch = {}
        self._prefetch_chunk(self.chunk_files[chunk_order[0]], prefetch)

        for i, ci in enumerate(chunk_order):
            counts, condition, labels = prefetch["data"]

            # Prefetch next
            if i + 1 < len(chunk_order):
                prefetch = {}
                thread = threading.Thread(
                    target=self._prefetch_chunk,
                    args=(self.chunk_files[chunk_order[i + 1]], prefetch),
                )
                thread.start()

            n = len(counts)
            row_order = rng.permutation(n) if self.shuffle else np.arange(n)
            for j in row_order:
                yield counts[j], condition[j], labels[j]

            if i + 1 < len(chunk_order):
                thread.join()

    def _prefetch_chunk(self, path, result_holder):
        result_holder["data"] = _load_single_chunk(path, dtype=self.dtype)


