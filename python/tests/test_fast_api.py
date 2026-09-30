"""Tests for the NumPy fast path. Run: pytest python/tests/test_fast_api.py"""
import numpy as np
import pytest

import rustygraph as rg

RNG = np.random.default_rng(0)


def nvg_ref(y, x=None):
    x = np.arange(len(y), dtype=float) if x is None else x
    return {
        (a, b)
        for a in range(len(y))
        for b in range(a + 1, len(y))
        if all((y[c] - y[b]) * (x[b] - x[a]) < (y[a] - y[b]) * (x[b] - x[c]) for c in range(a + 1, b))
    }


def hvg_ref(y):
    return {
        (a, b)
        for a in range(len(y))
        for b in range(a + 1, len(y))
        if all(y[c] < min(y[a], y[b]) for c in range(a + 1, b))
    }


def as_set(e):
    return set(map(tuple, e.tolist()))


@pytest.mark.parametrize("trial", range(200))
def test_matches_bruteforce(trial):
    n = int(RNG.integers(1, 60))
    y = RNG.integers(0, 4, size=n).astype(float) if trial % 2 else RNG.normal(size=n)
    assert as_set(rg.natural_visibility_edges(y)) == nvg_ref(y)
    assert as_set(rg.horizontal_visibility_edges(y)) == hvg_ref(y)


def test_explicit_positions():
    y = np.array([0.0, 1.0, 3.0])
    assert (0, 2) in as_set(rg.natural_visibility_edges(y, np.array([0.0, 1.0, 2.0])))
    assert (0, 2) not in as_set(rg.natural_visibility_edges(y, np.array([0.0, 0.1, 2.0])))


def test_output_format_and_inputs():
    e = rg.natural_visibility_edges([1, 3, 2, 4])  # list of ints is accepted
    assert e.dtype == np.int64 and e.shape[1] == 2 and (e[:, 0] < e[:, 1]).all()
    assert rg.natural_visibility_edges(np.arange(10.0)[::2]).shape == (4, 2)  # non-contiguous
    assert rg.natural_visibility_edges(np.array([], dtype=float)).shape == (0, 2)


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, np.inf]])
def test_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        rg.natural_visibility_edges(np.array(bad))


def test_rejects_bad_positions():
    with pytest.raises(ValueError):
        rg.natural_visibility_edges(np.ones(3), np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        rg.natural_visibility_edges(np.ones(3), np.array([0.0, 1.0]))


def test_batch_equals_per_row_and_ragged():
    Y = RNG.normal(size=(300, 23))
    e, off = rg.natural_visibility_batch(Y)
    assert len(off) == 301 and off[-1] == len(e)
    for k in range(300):
        assert np.array_equal(e[off[k] : off[k + 1]], rg.natural_visibility_edges(Y[k]))
    ragged = [RNG.normal(size=n) for n in (1, 5, 17)]
    e, off = rg.horizontal_visibility_batch(ragged)
    for k, r in enumerate(ragged):
        assert as_set(e[off[k] : off[k + 1]]) == hvg_ref(r)


def test_large_parallel_path():
    y = RNG.normal(size=100_000).cumsum()
    e = rg.natural_visibility_edges(y)
    assert len(set(map(tuple, e.tolist()))) == len(e)  # no duplicates
    assert (e[:, 0] < e[:, 1]).all()


def test_object_api_consistent():
    y = RNG.normal(size=200)
    g = rg.natural_visibility(y)
    assert {(a, b) for a, b, _ in g.edges()} == as_set(rg.natural_visibility_edges(y))
    A = g.adjacency_matrix()
    assert np.array_equal(A, A.T)
