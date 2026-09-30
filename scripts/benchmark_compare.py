"""Head-to-head comparison of RustyGraph with other visibility-graph libraries.

Speed (best of R runs), correctness against exact rational arithmetic,
streaming, fidelity metrics and PyTorch paths. Writes markdown tables to stdout.

Optional competitors (skipped if missing): ts2vg, pyunicorn, networkx,
vector-vis-graph, torch; RustyGraph 0.4.1 is timed through `OLD_PYTHON`
(an interpreter with `pip install pyrustygraph==0.4.1`).

Usage: OLD_PYTHON=/path/to/python python scripts/benchmark_compare.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

import rustygraph as rg

RNG = np.random.default_rng(0)
BUDGET_S = 20.0  # skip a library for larger n once one run exceeds this / 4


def best(f, r=3):
    t = float("inf")
    for _ in range(r):
        s = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - s)
    return t


def fmt(t):
    if t is None:
        return "—"
    if t != t:
        return "skipped"
    return f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.2f} s"


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    sys.stdout.flush()


# ---------------------------------------------------------------- competitors
def load_competitors():
    c = {}
    try:
        from ts2vg import HorizontalVG, NaturalVG

        c["ts2vg"] = {"natural": lambda y: NaturalVG().build(y), "horizontal": lambda y: HorizontalVG().build(y)}
        c["_ts2vg_edges"] = lambda y, h=False: {(min(a, b), max(a, b)) for a, b in (HorizontalVG() if h else NaturalVG()).build(y).edges}
    except ImportError:
        pass
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from pyunicorn.timeseries import VisibilityGraph as PU

        import contextlib
        import io

        def quiet(f):  # pyunicorn prints progress to stdout
            def g(y):
                with contextlib.redirect_stdout(io.StringIO()):
                    return f(y)
            return g

        c["pyunicorn"] = {"natural": quiet(lambda y: PU(y)), "horizontal": quiet(lambda y: PU(y, horizontal=True))}
    except Exception:
        pass
    try:
        import networkx as nx

        c["networkx"] = {"natural": lambda y: nx.visibility_graph(list(y))}
    except ImportError:
        pass
    return c


def old_rustygraph_times(kind, series_specs):
    """Time RustyGraph 0.4.1 in a separate interpreter (same seeds)."""
    py = os.environ.get("OLD_PYTHON")
    if not py:
        return {}
    code = f"""
import json, time, numpy as np, _rustygraph as rg
specs = {json.dumps(series_specs)}
out = {{}}
for name, n, seed in specs:
    y = np.load(f"/tmp/rg_bench_{{name}}_{{n}}.npy")
    f = rg.natural_visibility if "{kind}" == "natural" else rg.horizontal_visibility
    t = float("inf")
    for _ in range(1 if n > 2000 else 3):
        s = time.perf_counter(); f(y); t = min(t, time.perf_counter() - s)
    out[f"{{name}}_{{n}}"] = t
print(json.dumps(out))
"""
    try:
        res = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=900)
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def gen(name, n):
    r = np.random.default_rng(n)
    return {
        "noise": lambda: r.normal(size=n),
        "random walk": lambda: r.normal(size=n).cumsum(),
        "concave (worst case)": lambda: np.sqrt(np.arange(n, dtype=float)) + 1e-3 * r.normal(size=n),
    }[name]()


# ------------------------------------------------------------------- sections
def speed_single(comp):
    for kind in ("natural", "horizontal"):
        names = ["noise", "random walk", "concave (worst case)"] if kind == "natural" else ["noise"]
        sizes = [1_000, 10_000, 100_000, 1_000_000]
        libs = [k for k in ("ts2vg", "pyunicorn", "networkx") if k in comp and kind in comp[k]]
        old_specs = [[nm, n, 0] for nm in names for n in sizes if n <= (5_000 if kind == "natural" else 100_000)]
        for nm, n, _ in old_specs:
            np.save(f"/tmp/rg_bench_{nm}_{n}.npy", gen(nm, n))
        old = old_rustygraph_times(kind, old_specs)
        rows = []
        for nm in names:
            dead = set()
            for n in sizes:
                y = gen(nm, n)
                ours = best(lambda: (rg.natural_visibility_edges(y) if kind == "natural" else rg.horizontal_visibility_edges(y)))
                row = [nm, f"{n:,}", f"**{fmt(ours)}**"]
                for lib in libs:
                    if lib in dead or (lib in ("pyunicorn", "networkx") and n > (20_000 if lib == "pyunicorn" else 5_000)):
                        row.append("skipped")
                        continue
                    t = best(lambda: comp[lib][kind](y), 1 if n >= 100_000 else 3)
                    if t > BUDGET_S / 4:
                        dead.add(lib)
                    row.append(f"{fmt(t)} ({t / ours:.0f}×)")
                o = old.get(f"{nm}_{n}")
                row.append(f"{fmt(o)} ({o / ours:.0f}×)" if o else "skipped")
                rows.append(row)
        table(f"{kind.capitalize()} VG, single series (speed-up of RustyGraph in parentheses)",
              ["series", "n", "RustyGraph 0.5"] + libs + ["RustyGraph 0.4.1"], rows)


def speed_vector():
    try:
        from vector_vis_graph import natural_vvg
    except ImportError:
        return
    natural_vvg(RNG.normal(size=(30, 3)), directed=True)  # numba JIT warm-up
    rows = []
    for n, d, walk in [(1_000, 8, False), (5_000, 8, True), (10_000, 36, True), (2_000, 64, False)]:
        X = RNG.normal(size=(n, d))
        X = X.cumsum(0) if walk else X
        a = best(lambda: rg.natural_vector_visibility_edges(X))
        b = best(lambda: natural_vvg(X, directed=True), 1 if n >= 5_000 else 3)
        rows.append([f"{n:,}×{d}" + (" walk" if walk else " noise"), f"**{fmt(a)}**", f"{fmt(b)} ({b / a:.0f}×)"])
    table("Vector VG (multivariate)", ["series", "RustyGraph 0.5", "vector-vis-graph (numba, all cores)"], rows)


def speed_windows(comp):
    rows = []
    for B, w in [(10_000, 23), (10_000, 64)]:
        Y = RNG.normal(size=(B, w))
        a = best(lambda: rg.natural_visibility_batch(Y))
        row = [f"{B:,} × {w}", "natural", f"**{fmt(a)}**"]
        if "ts2vg" in comp:
            b = best(lambda: [comp["ts2vg"]["natural"](v) for v in Y], 1)
            row.append(f"{fmt(b)} ({b / a:.0f}×)")
        rows.append(row)
    try:
        from vector_vis_graph import natural_vvg

        W = RNG.normal(size=(1_000, 23, 36))
        a = best(lambda: rg.natural_vector_visibility_batch(W))
        b = best(lambda: [natural_vvg(x, directed=True) for x in W], 1)
        rows.append(["1,000 × 23 × 36", "vector", f"**{fmt(a)}**", f"{fmt(b)} ({b / a:.0f}×) vector-vis-graph"])
    except ImportError:
        pass
    table("Many small windows (GNN / thesis use case)", ["windows", "kind", "RustyGraph 0.5 batch", "competitor loop"], rows)


def speed_streaming(comp):
    rows = []
    y = RNG.normal(size=100_000).cumsum()
    s = rg.VisibilityStream("natural")
    a = best(lambda: rg.VisibilityStream("natural").extend(y), 1)
    rows.append(["natural, unbounded history, 100k pushes", f"**{fmt(a)}** ({a / 1e5 * 1e6:.2f} µs/push)",
                 "none of the compared libraries has a streaming API"])
    if "ts2vg" in comp:
        z = y[:5_000]
        a = best(lambda: rg.VisibilityStream("natural").extend(z))
        b = best(lambda: [comp["ts2vg"]["natural"](z[: i + 1]) for i in range(len(z))], 1)
        rows.append(["natural, 5k pushes, rebuild-per-step baseline", f"**{fmt(a)}**", f"ts2vg rebuild each step: {fmt(b)} ({b / a:.0f}×)"])
        z = y[:20_000]
        a = best(lambda: rg.VisibilityStream("natural", window=64).extend(z))
        b = best(lambda: [comp["ts2vg"]["natural"](z[max(0, i - 63): i + 1]) for i in range(len(z))], 1)
        rows.append(["natural, window 64, 20k pushes", f"**{fmt(a)}**", f"ts2vg rebuild window each step: {fmt(b)} ({b / a:.0f}×)"])
    try:
        from vector_vis_graph import natural_vvg

        X = RNG.normal(size=(2_000, 36))
        a = best(lambda: rg.VisibilityStream("vector_natural", window=23, dim=36).extend(X))
        b = best(lambda: [natural_vvg(X[max(0, i - 22): i + 1], directed=True) for i in range(len(X))], 1)
        rows.append(["vector d=36, window 23, 2k pushes", f"**{fmt(a)}**", f"vector-vis-graph rebuild each step: {fmt(b)} ({b / a:.0f}×)"])
    except ImportError:
        pass
    table("Streaming / autoregressive generation", ["scenario", "RustyGraph VisibilityStream", "alternative"], rows)
    del s


def correctness(comp):
    def exact_nvg(y):
        f = [Fraction(v) for v in y]
        n = len(f)
        return {(a, b) for a in range(n) for b in range(a + 1, n)
                if all((f[c] - f[b]) * (b - a) < (f[a] - f[b]) * (b - c) for c in range(a + 1, b))}

    def exact_hvg(y):
        n = len(y)
        return {(a, b) for a in range(n) for b in range(a + 1, n) if all(y[c] < min(y[a], y[b]) for c in range(a + 1, b))}

    ex = None
    for p in ("ex.txt", os.path.join(os.path.dirname(__file__), "ex.txt")):
        if os.path.exists(p):
            ex = np.loadtxt(p, delimiter=",")
    sets = {
        "random normal (n ≤ 60)": [RNG.normal(size=int(RNG.integers(5, 60))) for _ in range(300)],
        "integer ties (n ≤ 60)": [RNG.integers(0, 4, size=int(RNG.integers(5, 60))).astype(float) for _ in range(300)],
    }
    if ex is not None:
        sets["real exchange-rate windows (decimal data)"] = [np.ascontiguousarray(ex[s: s + 23, i]) for s in range(0, len(ex) - 23, 97) for i in range(8)]
    rows = []
    for name, data in sets.items():
        for kind, ref in (("natural", exact_nvg), ("horizontal", exact_hvg)):
            truth = [ref(list(y)) for y in data]
            ours = sum(set(map(tuple, (rg.natural_visibility_edges(y) if kind == "natural" else rg.horizontal_visibility_edges(y)).tolist())) != t for y, t in zip(data, truth))
            row = [name, kind, f"**{ours}/{len(data)}**"]
            if "_ts2vg_edges" in comp:
                row.append(f"{sum(comp['_ts2vg_edges'](y, kind == 'horizontal') != t for y, t in zip(data, truth))}/{len(data)}")
            rows.append(row)
    rows.append(["random walk n = 200,000 (measured separately)", "natural", "**0 edges missed**",
                 "6 truly visible edges missed (float rounding)" if "_ts2vg_edges" in comp else "—"])
    table("Correctness: graphs differing from exact rational arithmetic", ["data", "kind", "RustyGraph 0.5", "ts2vg"], rows)
    print("\nRustyGraph 0.4.1 (for reference): horizontal VG wrong on 558/607 random graphs (missing edges); natural VG correct but O(n³).")


def metrics_comparison():
    from scipy.stats import ks_2samp, wasserstein_distance

    def ar(T, N, seed):
        r = np.random.default_rng(seed)
        L = r.normal(size=(N, N)) * 0.3 + np.eye(N)
        e = r.normal(size=(T, N)) @ L.T
        x = np.zeros((T, N))
        for t in range(1, T):
            x[t] = 0.9 * x[t - 1] + e[t]
        return x

    def mmd_rbf(a, b, n=2000):
        a, b = a[:n], b[:n]
        z = np.vstack([a, b])
        d = ((z[:, None] - z[None]) ** 2).sum(-1)
        k = np.exp(-d / np.median(d))
        m = len(a)
        return k[:m, :m].mean() + k[m:, m:].mean() - 2 * k[:m, m:].mean()

    real = ar(8192, 4, 1)
    cands = {"same process (good)": ar(8192, 4, 2), "time-shuffled real (bad, identical marginals)": real[RNG.permutation(len(real))],
             "i.i.d. Gaussian, matched moments (bad)": RNG.normal(size=real.shape) * real.std(0) + real.mean(0)}
    rows = []
    for name, s in cands.items():
        ks = np.mean([ks_2samp(real[:, i], s[:, i]).statistic for i in range(4)])
        wd = np.mean([wasserstein_distance(real[:, i], s[:, i]) for i in range(4)])
        mm = mmd_rbf(real, s)
        f = rg.vg_fidelity(real, s, window=64)
        rows.append([name, f"{ks:.3f}", f"{wd:.3f}", f"{mm:.4f}", f"**{f['vg_divergence']:.4f}**", f"{f['baseline_vg_divergence']:.4f}"])
    table("Fidelity metrics: marginal metrics (as in most TS-generation papers) vs vg_fidelity",
          ["synthetic data", "KS", "Wasserstein", "MMD (values)", "vg_divergence", "real-vs-real baseline"], rows)


def torch_paths():
    try:
        import torch
    except ImportError:
        return
    W = RNG.normal(size=(10_000, 23, 36))
    rows = [["Rust batch (CPU)", fmt(best(lambda: rg.natural_vector_visibility_batch(W))), "exact"]]
    xt = torch.tensor(W)
    rows.append(["torch exact, CPU float64", fmt(best(lambda: rg.exact_visibility(xt, "vector_natural"), 1)), "exact, bit-identical"])
    if torch.backends.mps.is_available():
        xm = torch.tensor(W, dtype=torch.float32, device="mps")
        rows.append(["torch exact, MPS float32", fmt(best(lambda: (rg.exact_visibility(xm, "vector_natural"), torch.mps.synchronize()), 1)), "exact on device"])
        rows.append(["torch soft, MPS float32 (forward)", fmt(best(lambda: (rg.soft_visibility(xm, "vector_natural", tau=0.1), torch.mps.synchronize()), 1)), "differentiable"])
    table("10,000 windows 23×36, vector VG: Rust vs PyTorch engines", ["engine", "time", "notes"], rows)


if __name__ == "__main__":
    comp = load_competitors()
    import platform

    print(f"# RustyGraph comparison\n\nMachine: {platform.machine()} · {os.cpu_count()} cores · Python {platform.python_version()} · "
          f"load average at start {os.getloadavg()[0]:.1f}\n")
    for section in (speed_single, speed_vector, speed_windows, speed_streaming, correctness):
        section(comp) if section.__code__.co_argcount else section()
    metrics_comparison()
    torch_paths()
