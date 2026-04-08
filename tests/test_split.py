"""Tests for sjanpy.pp.split — stratified train/val/test splitting."""

import numpy as np
import pandas as pd
import pytest

from sjanpy.pp.split import stratified_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(n: int, types: list[str], rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Build a minimal obs DataFrame with *n* cells and given cell types."""
    if rng is None:
        rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"cell_type": rng.choice(types, size=n)},
        index=[f"cell_{i}" for i in range(n)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_correct_columns():
    """Result has cell_index + split columns, length matches input."""
    obs = _make_obs(200, ["A", "B", "C"])
    result = stratified_split(obs, "cell_type")

    assert list(result.columns) == ["cell_index", "split"]
    assert len(result) == len(obs)


def test_split_ratios_approximate():
    """1000 cells, 5 types -> roughly 90/5/5 split."""
    obs = _make_obs(1000, ["A", "B", "C", "D", "E"], rng=np.random.default_rng(7))
    result = stratified_split(obs, "cell_type", val_ratio=0.05, test_ratio=0.05)

    counts = result["split"].value_counts()
    total = len(result)

    assert counts["train"] / total == pytest.approx(0.90, abs=0.03)
    assert counts["val"] / total == pytest.approx(0.05, abs=0.03)
    assert counts["test"] / total == pytest.approx(0.05, abs=0.03)


def test_all_indices_covered():
    """Every cell index from 0..n-1 appears exactly once."""
    obs = _make_obs(500, ["X", "Y"])
    result = stratified_split(obs, "cell_type")

    assert sorted(result["cell_index"].tolist()) == list(range(len(obs)))


def test_rare_types_go_to_train():
    """A cell type with only 1 cell must land in train."""
    types = ["Common"] * 99 + ["Singleton"]
    obs = pd.DataFrame({"cell_type": types}, index=[f"c{i}" for i in range(100)])
    result = stratified_split(obs, "cell_type")

    singleton_row = result.loc[result["cell_index"] == 99]
    assert singleton_row["split"].iloc[0] == "train"


def test_reproducible_with_seed():
    """Same seed produces identical splits."""
    obs = _make_obs(300, ["A", "B", "C"])
    r1 = stratified_split(obs, "cell_type", seed=123)
    r2 = stratified_split(obs, "cell_type", seed=123)

    pd.testing.assert_frame_equal(r1, r2)


def test_all_rare_types():
    """When every type has exactly 1 cell, all go to train."""
    obs = pd.DataFrame(
        {"cell_type": [f"type_{i}" for i in range(10)]},
        index=[f"c{i}" for i in range(10)],
    )
    result = stratified_split(obs, "cell_type")

    assert (result["split"] == "train").all()


def test_zero_ratios_all_train():
    """val_ratio=0, test_ratio=0 should put everything in train."""
    obs = pd.DataFrame({"ct": ["A"] * 50 + ["B"] * 50},
                       index=[f"c{i}" for i in range(100)])
    result = stratified_split(obs, "ct", val_ratio=0, test_ratio=0)
    assert (result["split"] == "train").all()


def test_invalid_ratios_raise():
    """Negative or >= 1.0 ratios should raise ValueError."""
    obs = pd.DataFrame({"ct": ["A"] * 10}, index=[f"c{i}" for i in range(10)])

    with pytest.raises(ValueError):
        stratified_split(obs, "ct", val_ratio=-0.1, test_ratio=0.1)
    with pytest.raises(ValueError):
        stratified_split(obs, "ct", val_ratio=0.6, test_ratio=0.6)
