"""Vector visibility graph tests. Run: pytest python/tests/test_vector_vg.py"""
import numpy as np
import pytest

import rustygraph as rg

RNG = np.random.default_rng(1)


def vvg_ref(X, t=None, horizontal=False):
    """Definition (vector-vis-graph semantics): project on the earlier vector's direction."""
    n = len(X)
    t = np.arange(n, dtype=float) if t is None else t
    E = set()
    for a in range(n):
        p = X @ X[a]  # scaled projections, same visibility as X @ X[a] / |X[a]|
        for b in range(a + 1, n):
            mid = range(a + 1, b)
            if horizontal:
                ok = all(p[k] < min(p[a], p[b]) for k in mid)
            else:
                ok = all((p[k] - p[a]) * (t[b] - t[a]) < (p[b] - p[a]) * (t[k] - t[a]) for k in mid)
            if ok:
                E.add((a, b))
    return E


def as_set(e):
    return set(map(tuple, e.tolist()))


@pytest.mark.parametrize("trial", range(100))
def test_matches_definition(trial):
    n, d = int(RNG.integers(2, 40)), int(RNG.integers(1, 7))
    X = RNG.normal(size=(n, d)) + (2.0 if trial % 2 else 0.0)
    t = np.cumsum(RNG.uniform(0.1, 2.0, size=n))
    assert as_set(rg.natural_vector_visibility_edges(X)) == vvg_ref(X)
    assert as_set(rg.natural_vector_visibility_edges(X, t)) == vvg_ref(X, t)
    assert as_set(rg.horizontal_vector_visibility_edges(X)) == vvg_ref(X, horizontal=True)


def test_against_vector_vis_graph():
    vvg = pytest.importorskip("vector_vis_graph")
    for _ in range(50):
        X = RNG.normal(size=(int(RNG.integers(2, 50)), 5))
        A = vvg.natural_vvg(X, directed=True)
        assert as_set(rg.natural_vector_visibility_edges(X)) == set(zip(*np.nonzero(A)))
    X = RNG.normal(size=(30, 4)) + 1
    A = vvg.natural_vvg(X, weight_method=vvg.WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE, directed=True)
    e, w = rg.natural_vector_visibility_edges(X, weight="time_diff_euclidean_distance")
    B = np.zeros_like(A)
    B[e[:, 0], e[:, 1]] = w
    np.testing.assert_allclose(A, B, atol=1e-12)


def test_batch_and_shapes():
    W = RNG.normal(size=(50, 23, 36))
    e, off = rg.natural_vector_visibility_batch(W)
    assert len(off) == 51
    for k in range(50):
        assert as_set(e[off[k] : off[k + 1]]) == as_set(rg.natural_vector_visibility_edges(W[k]))
    y = np.abs(RNG.normal(size=40)) + 1  # 1-D positive input == univariate NVG
    assert as_set(rg.natural_vector_visibility_edges(y)) == as_set(rg.natural_visibility_edges(y))


def test_errors():
    with pytest.raises(ValueError):
        rg.natural_vector_visibility_edges(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        rg.natural_vector_visibility_edges(np.ones((3, 2)), weight="nope")
    with pytest.raises(ValueError):
        rg.natural_vector_visibility_edges(np.ones((3, 2)), np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        rg.natural_vector_visibility_batch(np.ones((3, 2)))
