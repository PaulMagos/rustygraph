"""Tests: hull engine, streaming, motifs, fidelity metrics, auto dispatcher."""
import numpy as np
import pytest

import rustygraph as rg

RNG = np.random.default_rng(7)


def as_set(e):
    return set(map(tuple, np.asarray(e).tolist()))


# --- hybrid / output-sensitive natural VG ------------------------------------

@pytest.mark.parametrize(
    "y",
    [
        np.sqrt(np.arange(20_000.0)),
        np.sqrt(np.arange(20_000.0)) + 1e-3 * RNG.normal(size=20_000),
        (RNG.normal(size=50_000) + 0.05).cumsum(),
        np.exp(-np.arange(5_000) / 1000) + 1e-4 * RNG.normal(size=5_000),
        RNG.normal(size=50_000),
    ],
    ids=["concave", "concave+noise", "walk+drift", "exp-decay", "noise"],
)
def test_all_engines_identical(y):
    ref = as_set(rg.natural_visibility_edges(y, algorithm="scan"))
    assert as_set(rg.natural_visibility_edges(y, algorithm="hull")) == ref
    assert as_set(rg.natural_visibility_edges(y)) == ref


def test_plan_is_dynamic():
    assert rg.natural_visibility_plan(np.sqrt(np.arange(50_000.0)))["algorithm"] == "hull"
    assert rg.natural_visibility_plan(RNG.normal(size=50_000))["algorithm"] == "scan"
    with pytest.raises(ValueError):
        rg.natural_visibility_edges(np.ones(3), algorithm="nope")


# --- streaming ------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["auto", "scan", "indexed"])
def test_stream_natural_equals_batch(mode):
    y = (RNG.normal(size=6_000)).cumsum()
    s = rg.VisibilityStream("natural", mode=mode)
    assert as_set(s.extend(y)) == as_set(rg.natural_visibility_edges(y))
    assert len(s) == len(y)


def test_stream_push_window_and_time():
    y = RNG.normal(size=300)
    t = np.cumsum(RNG.uniform(0.5, 2.0, size=300))
    full = as_set(rg.natural_visibility_edges(y, t))
    s = rg.VisibilityStream("natural", window=25)
    got = set()
    for i in range(300):
        for a in s.push(y[i], t[i]).tolist():
            got.add((a, i))
    assert got == {e for e in full if e[1] - e[0] <= 25}


@pytest.mark.parametrize("kind", ["horizontal", "vector_natural", "vector_horizontal"])
def test_stream_other_kinds(kind):
    if kind == "horizontal":
        y = RNG.integers(0, 5, size=1000).astype(float)
        ref = rg.horizontal_visibility_edges(y)
        s = rg.VisibilityStream(kind)
    else:
        y = RNG.normal(size=(400, 4))
        ref = rg.natural_vector_visibility_edges(y) if kind == "vector_natural" else rg.horizontal_vector_visibility_edges(y)
        s = rg.VisibilityStream(kind, dim=4)
    assert as_set(s.extend(y)) == as_set(ref)


def test_stream_errors():
    with pytest.raises(ValueError):
        rg.VisibilityStream("vector_natural")
    with pytest.raises(ValueError):
        rg.VisibilityStream("natural", window=0)
    s = rg.VisibilityStream("natural")
    s.push(1.0, 3.0)
    with pytest.raises(ValueError):
        s.push(2.0, 3.0)


# --- motifs ------------------------------------------------------------------------

def motifs_from_edges(edges, n):
    e = as_set(edges)
    c = np.zeros(8, int)
    for s in range(n - 3):
        c[((s, s + 2) in e) | ((s + 1, s + 3) in e) << 1 | ((s, s + 3) in e) << 2] += 1
    return c


def test_motifs_match_graph():
    y = RNG.normal(size=500)
    X = RNG.normal(size=(300, 3))
    assert np.array_equal(rg.visibility_motifs(y), motifs_from_edges(rg.natural_visibility_edges(y), 500))
    assert np.array_equal(rg.visibility_motifs(y, "horizontal"), motifs_from_edges(rg.horizontal_visibility_edges(y), 500))
    assert np.array_equal(rg.visibility_motifs(X, "vector_natural"), motifs_from_edges(rg.natural_vector_visibility_edges(X), 300))
    B = rg.visibility_motifs(RNG.normal(size=(10, 50)))
    assert B.shape == (10, 8) and (B.sum(1) == 47).all()


def test_nvg_impossible_motifs_never_appear():
    # codes 3 ([0,2]&[1,3] without [0,3]) and 4 ([0,3] alone) are geometrically impossible
    c = rg.visibility_motifs(RNG.normal(size=100_000))
    assert c[3] == 0 and c[4] == 0


# --- fidelity metrics --------------------------------------------------------------------

def ar_process(T, N, rng):
    L = rng.normal(size=(N, N)) * 0.3 + np.eye(N)
    eps = rng.normal(size=(T, N)) @ L.T
    x = np.zeros((T, N))
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + eps[t]
    return x


def test_fidelity_separates_temporal_structure():
    real = ar_process(4096, 4, np.random.default_rng(1))
    same = ar_process(4096, 4, np.random.default_rng(2))
    shuffled = real[RNG.permutation(len(real))]  # identical marginals
    good = rg.vg_fidelity(real, same, window=64)
    bad = rg.vg_fidelity(real, shuffled, window=64)
    assert good["vg_divergence"] < 3 * good["baseline_vg_divergence"] + 1e-3
    assert bad["vg_divergence"] > 10 * good["vg_divergence"]
    assert bad["synth_hvg_iid_distance"] < good["synth_hvg_iid_distance"]  # shuffled ~ i.i.d.
    assert {"jsd_vvg_motifs", "absdiff_multiplex_mi", "absdiff_irreversibility"} <= set(good)


def test_hvg_iid_null_matches_theory():
    from rustygraph.metrics import hvg_iid_null, jensen_shannon
    # long i.i.d. series: HVG degrees follow the analytic law
    e = rg.horizontal_visibility_edges(RNG.normal(size=200_000))
    deg = np.bincount(e.ravel(), minlength=200_000)[1:-1]
    assert jensen_shannon(np.bincount(np.minimum(deg, 32), minlength=33), hvg_iid_null(32)) < 1e-4
    # finite windows: the shuffle null makes the index ~0 for i.i.d. data at any window
    for w in (32, 200):
        assert rg.vg_descriptors(RNG.normal(size=100_000), window=w)["hvg_iid_distance"] < 2e-3


def test_fidelity_input_validation():
    with pytest.raises(ValueError):
        rg.vg_fidelity(np.ones((100, 2)), np.ones((100, 3)), window=20)
    with pytest.raises(ValueError):
        rg.vg_descriptors(np.array([1.0, np.nan, 2, 3, 4]))


# --- auto dispatcher ---------------------------------------------------------------------

def test_visibility_dispatch_numpy():
    y = RNG.normal(size=1000)
    assert as_set(rg.visibility(y)) == as_set(rg.natural_visibility_edges(y))
    e, off = rg.visibility(RNG.normal(size=(5, 30)))
    assert len(off) == 6
    X = RNG.normal(size=(50, 3))
    assert as_set(rg.visibility(X, "vvg")) == as_set(rg.natural_vector_visibility_edges(X))
    e, w = rg.visibility(X, "vvg", weight="euclidean_distance")
    assert len(w) == len(e)
    A = rg.visibility(y[:40], output="adjacency")
    assert A.shape == (40, 40) and np.array_equal(A, A.T)
    ei = rg.visibility(y[:40], output="edge_index")
    assert ei.shape[0] == 2
    avg = rg.visibility(RNG.normal(size=(60, 4)), "average", output="adjacency")
    assert avg.max() <= 1.0 and np.allclose(avg, avg.T)
    _, plan = rg.visibility(np.sqrt(np.arange(50_000.0)), explain=True)
    assert plan["algorithm"] == "hull"


def test_visibility_dispatch_torch():
    torch = pytest.importorskip("torch")
    x = torch.randn(4, 23, dtype=torch.float64, requires_grad=True)
    A, plan = rg.visibility(x, output="adjacency", explain=True)
    assert plan["engine"] == "torch.soft"
    A.sum().backward()
    assert torch.isfinite(x.grad).all()
    y = torch.randn(23, dtype=torch.float64)
    assert as_set(rg.visibility(y)) == as_set(rg.natural_visibility_edges(y.numpy()))
    with pytest.raises(ValueError):
        rg.visibility(np.ones(5), differentiable=True)
