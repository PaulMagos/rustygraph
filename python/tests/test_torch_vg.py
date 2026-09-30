"""Tests for rustygraph.torch_vg. Run: pytest python/tests/test_torch_vg.py

Benchmark: python python/tests/test_torch_vg.py
"""
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rustygraph as rg  # noqa: E402
from rustygraph.torch_vg import (  # noqa: E402
    VisibilityLayer,
    edge_index_from_adjacency,
    exact_visibility,
    soft_degree,
    soft_degree_histogram,
    soft_visibility,
    to_undirected,
)

N_CASES = 200
HAS_MPS = torch.backends.mps.is_available()


def edge_set(e):
    return set(map(tuple, np.asarray(e).tolist()))


def adj_set(A):
    return set(map(tuple, torch.nonzero(A.cpu()).tolist()))


def reference(kind, x, t=None):
    """rustygraph exact edges for one series/window."""
    if kind == "natural":
        return edge_set(rg.natural_visibility_edges(x, t))
    if kind == "horizontal":
        return edge_set(rg.horizontal_visibility_edges(x))
    if kind == "vector_natural":
        return edge_set(rg.natural_vector_visibility_edges(x, t))
    return edge_set(rg.horizontal_vector_visibility_edges(x))


def random_case(kind, rng, trial):
    """Mix of float, integer-tie, offset and irregular-position cases."""
    w = int(rng.integers(1, 40))
    vec = kind.startswith("vector")
    shape = (w, int(rng.integers(1, 7))) if vec else (w,)
    style = trial % 4
    if style == 0:
        x = rng.normal(size=shape)
    elif style == 1:
        x = rng.integers(-3, 4, size=shape).astype(float)  # many ties / collinear points
    elif style == 2:
        x = rng.normal(size=shape) + 2.0
    else:
        x = rng.integers(0, 10, size=shape).astype(float)
    t = None
    if kind.endswith("natural") and trial % 2 == 1:
        t = np.cumsum(rng.uniform(0.1, 2.0, size=w))
        if style == 3:
            t = np.cumsum(rng.integers(1, 4, size=w)).astype(float)
    return x, t


# ---------------------------------------------------------------------------
# exact vs rustygraph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["natural", "horizontal", "vector_natural", "vector_horizontal"])
def test_exact_matches_rustygraph(kind):
    rng = np.random.default_rng(len(kind) * 101 + ord(kind[0]))
    bad = []
    for trial in range(N_CASES):
        x, t = random_case(kind, rng, trial)
        A = exact_visibility(torch.from_numpy(x), kind, None if t is None else torch.from_numpy(t))
        assert A.dtype == torch.bool and A.shape == (len(x), len(x))
        if adj_set(A) != reference(kind, x, t):
            bad.append(trial)
    assert not bad, f"{len(bad)}/{N_CASES} mismatches: {bad[:10]}"


@pytest.mark.parametrize("kind", ["natural", "horizontal"])
def test_exact_batch_univariate(kind):
    rng = np.random.default_rng(3)
    Y = np.concatenate([rng.normal(size=(40, 25)), rng.integers(-2, 3, size=(40, 25)).astype(float)])
    batch = rg.natural_visibility_batch if kind == "natural" else rg.horizontal_visibility_batch
    e, off = batch(Y)
    A = exact_visibility(torch.from_numpy(Y), kind, max_elems=1000)  # force many chunks
    for k in range(len(Y)):
        assert adj_set(A[k]) == edge_set(e[off[k]:off[k + 1]])


@pytest.mark.parametrize("kind", ["vector_natural", "vector_horizontal"])
def test_exact_batch_vector(kind):
    rng = np.random.default_rng(4)
    W = rng.normal(size=(60, 23, 36))
    W[0, 5] = 0.0  # zero vector anchor
    batch = rg.natural_vector_visibility_batch if kind == "vector_natural" else rg.horizontal_vector_visibility_batch
    e, off = batch(W)
    A = exact_visibility(torch.from_numpy(W).reshape(3, 20, 23, 36), kind, max_elems=5000)
    assert A.shape == (3, 20, 23, 23)
    A = A.reshape(60, 23, 23)
    for k in range(len(W)):
        assert adj_set(A[k]) == edge_set(e[off[k]:off[k + 1]])
    assert torch.nonzero(A[0, 5]).flatten().tolist() == [6]


def test_exact_batched_positions():
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(8, 15))
    T = np.cumsum(rng.uniform(0.1, 3, size=(8, 15)), axis=1)
    A = exact_visibility(torch.from_numpy(Y), "natural", torch.from_numpy(T))
    for k in range(8):
        assert adj_set(A[k]) == reference("natural", Y[k], T[k])


def test_validation():
    with pytest.raises(ValueError, match="kind"):
        exact_visibility(torch.zeros(5), "bogus")
    with pytest.raises(ValueError, match="NaN"):
        exact_visibility(torch.tensor([1.0, float("nan"), 2.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        exact_visibility(torch.rand(4), t=torch.tensor([0.0, 1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="last dimension"):
        exact_visibility(torch.rand(4), t=torch.arange(5.0))
    with pytest.raises(ValueError, match="2-D"):
        exact_visibility(torch.rand(4), "vector_natural")
    with pytest.raises(ValueError, match="not used"):
        exact_visibility(torch.rand(4), "horizontal", t=torch.arange(4.0))
    with pytest.raises(ValueError, match="tau"):
        soft_visibility(torch.rand(4), tau=0.0)
    with pytest.raises(ValueError, match="mode"):
        VisibilityLayer(mode="fast")


def test_to_undirected():
    A = exact_visibility(torch.randn(3, 12, dtype=torch.float64))
    S = to_undirected(A)
    assert torch.equal(S, S.transpose(-1, -2)) and torch.equal(torch.triu(S, 1), A)


# ---------------------------------------------------------------------------
# MPS
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
@pytest.mark.parametrize("kind", ["natural", "horizontal", "vector_natural", "vector_horizontal"])
def test_mps_agreement(kind):
    rng = np.random.default_rng(6)
    vec = kind.startswith("vector")
    X = rng.normal(size=(200, 30, 8) if vec else (200, 30)).astype(np.float32)
    A_mps = exact_visibility(torch.from_numpy(X).to("mps"), kind).cpu()
    A_cpu32 = exact_visibility(torch.from_numpy(X), kind)
    A_ref = exact_visibility(torch.from_numpy(X.astype(np.float64)), kind)
    upper = torch.triu(torch.ones(30, 30, dtype=torch.bool), 1)
    n_pairs = int(upper.sum()) * len(X)
    mism_ref = int((A_mps != A_ref).sum())
    assert mism_ref / n_pairs < 1e-3, f"{mism_ref}/{n_pairs} pairs differ from float64"
    assert int((A_mps != A_cpu32).sum()) / n_pairs < 1e-4


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_mps_soft_grad():
    x = torch.randn(4, 16, 3, device="mps", requires_grad=True)
    A = soft_visibility(x, "vector_natural", tau=0.1)
    soft_degree_histogram(A, 10).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# soft relaxation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["natural", "horizontal", "vector_natural", "vector_horizontal"])
def test_soft_converges_to_exact(kind):
    g = torch.Generator().manual_seed(7)
    vec = kind.startswith("vector")
    x = torch.randn((50, 16, 4) if vec else (50, 16), generator=g, dtype=torch.float64)
    t = None
    if kind.endswith("natural"):
        t = torch.cumsum(torch.rand(50, 16, generator=g, dtype=torch.float64) + 0.1, dim=-1)
    E = exact_visibility(x, kind, t)
    S = soft_visibility(x, kind, tau=1e-4, t=t)
    assert torch.equal(S > 0.5, E)
    assert (S - E.double()).abs().mean() < 1e-3
    assert torch.equal(torch.triu(S, 1), S)


@pytest.mark.parametrize("kind", ["natural", "horizontal", "vector_natural", "vector_horizontal"])
def test_soft_gradcheck(kind):
    g = torch.Generator().manual_seed(8)
    shape = (2, 6, 3) if kind.startswith("vector") else (2, 6)
    x = torch.randn(shape, generator=g, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: soft_visibility(z, kind, tau=0.3), (x,))
    tau = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s: soft_visibility(x.detach(), kind, tau=s), (tau,))


def test_soft_gradients_finite_degenerate():
    x = torch.zeros(10, 3, dtype=torch.float64)
    x[3:] = torch.randn(7, 3, dtype=torch.float64)
    x.requires_grad_(True)
    for kind in ("vector_natural", "vector_horizontal"):
        soft_visibility(x, kind).sum().backward()
        assert torch.isfinite(x.grad).all()
    y = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64, requires_grad=True)
    soft_visibility(y).sum().backward()
    assert torch.isfinite(y.grad).all()


def test_soft_scale_invariant():
    x = torch.randn(3, 20, dtype=torch.float64)
    assert torch.allclose(soft_visibility(x), soft_visibility(1000 * x))


# ---------------------------------------------------------------------------
# layer
# ---------------------------------------------------------------------------

def test_layer_auto_mode():
    layer = VisibilityLayer("natural", symmetric=True, self_loops=True)
    x = torch.randn(4, 20, dtype=torch.float64)
    A_exact = layer(x)
    assert not A_exact.requires_grad
    assert set(A_exact.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(A_exact, A_exact.transpose(-1, -2)) and bool((A_exact.diagonal(dim1=-2, dim2=-1) == 1).all())
    xg = x.clone().requires_grad_(True)
    A_soft = layer(xg)
    assert A_soft.requires_grad
    assert not set(A_soft.unique().tolist()) <= {0.0, 1.0}
    with torch.no_grad():
        assert not layer(xg).requires_grad


def test_layer_learnable_tau():
    layer = VisibilityLayer("vector_natural", mode="soft", tau=0.2, learn_tau=True)
    assert abs(layer.tau.item() - 0.2) < 1e-6
    x = torch.randn(3, 12, 4)
    soft_degree(layer(x)).mean().backward()
    assert layer.raw_tau.grad is not None and layer.raw_tau.grad.abs().item() > 0
    assert [n for n, _ in layer.named_parameters()] == ["raw_tau"]


# ---------------------------------------------------------------------------
# edge index and statistics
# ---------------------------------------------------------------------------

def test_edge_index_batching():
    rng = np.random.default_rng(9)
    Y = rng.normal(size=(3, 14))
    A = exact_visibility(torch.from_numpy(Y))
    ei = edge_index_from_adjacency(A)
    e, off = rg.natural_visibility_batch(Y)
    expect = np.concatenate([e[off[k]:off[k + 1]] + 14 * k for k in range(3)])
    assert ei.dtype == torch.long and ei.shape == (2, len(expect))
    assert edge_set(ei.T.numpy()) == edge_set(expect)
    e1 = e[off[1]:off[2]]
    e1 = e1[np.lexsort((e1[:, 1], e1[:, 0]))]  # edge_index is (source, target)-sorted
    assert torch.equal(edge_index_from_adjacency(A[1]), torch.from_numpy(e1.T.copy()))
    S = soft_visibility(torch.from_numpy(Y), tau=0.05)
    ei2, wts = edge_index_from_adjacency(S, threshold=0.5, return_weights=True)
    assert torch.equal(ei2, ei) and bool((wts > 0.5).all())
    with pytest.raises(ValueError):
        edge_index_from_adjacency(torch.zeros(3, 4))


def test_degree_and_histogram():
    rng = np.random.default_rng(10)
    Y = torch.from_numpy(rng.normal(size=(5, 30)))
    A = exact_visibility(Y)
    deg = soft_degree(A.double())
    assert torch.equal(deg, soft_degree(to_undirected(A).double()))
    ref = np.bincount(np.asarray(rg.natural_visibility_edges(Y[0].numpy())).ravel(), minlength=30)
    assert np.array_equal(deg[0].numpy(), ref)
    H = soft_degree_histogram(A.double(), max_k=40, bandwidth=0.05)
    assert torch.allclose(H.sum(-1), torch.ones(5, dtype=torch.float64))
    hard = torch.stack([torch.bincount(d.long(), minlength=41)[:41] / 30.0 for d in deg])
    assert torch.allclose(H, hard.double(), atol=1e-6)
    x = Y.clone().requires_grad_(True)
    target = soft_degree_histogram(A.double(), 20)
    loss = (soft_degree_histogram(soft_visibility(x, tau=0.1), 20) - target).abs().sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def _bench(fn, sync=None, reps=3):
    fn()
    if sync:
        sync()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


if __name__ == "__main__":
    B, w, d = 10_000, 23, 36
    W = np.random.default_rng(0).normal(size=(B, w, d))
    t_rg, (e, off) = _bench(lambda: rg.natural_vector_visibility_batch(W))
    print(f"rustygraph natural_vector_visibility_batch  {B}x{w}x{d}: {t_rg * 1e3:8.1f} ms  ({len(e)} edges)")
    X64 = torch.from_numpy(W)
    t_cpu, A_cpu = _bench(lambda: exact_visibility(X64, "vector_natural"))
    print(f"torch exact CPU float64                     : {t_cpu * 1e3:8.1f} ms  ({int(A_cpu.sum())} edges)")
    t_c32, A_c32 = _bench(lambda: exact_visibility(X64.float(), "vector_natural"))
    print(f"torch exact CPU float32                     : {t_c32 * 1e3:8.1f} ms  (mismatch vs f64: {int((A_c32 != A_cpu).sum())})")
    if HAS_MPS:
        Xm = X64.float().to("mps")
        t_mps, A_mps = _bench(lambda: exact_visibility(Xm, "vector_natural"), torch.mps.synchronize)
        print(f"torch exact MPS float32                     : {t_mps * 1e3:8.1f} ms  (mismatch vs f64: {int((A_mps.cpu() != A_cpu).sum())})")
        t_sm, _ = _bench(lambda: soft_visibility(Xm, "vector_natural"), torch.mps.synchronize)
        print(f"torch soft  MPS float32 (fwd)               : {t_sm * 1e3:8.1f} ms")
    ref = np.zeros((B, w, w), dtype=bool)
    k = np.repeat(np.arange(B), np.diff(off))
    ref[k, e[:, 0], e[:, 1]] = True
    print("CPU float64 == rustygraph:", bool((A_cpu.numpy() == ref).all()))
