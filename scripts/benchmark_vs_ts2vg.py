"""RustyGraph vs ts2vg benchmark (pip install ts2vg). Times = best of R runs.

Usage: python scripts/benchmark_vs_ts2vg.py            # all cores
       RAYON_NUM_THREADS=1 python scripts/benchmark_vs_ts2vg.py
"""
import os, sys, time
import numpy as np
import rustygraph as rg
from ts2vg import NaturalVG, HorizontalVG

R = 5
rng = np.random.default_rng(0)


def best(f, r=R):
    t = float("inf")
    for _ in range(r):
        s = time.perf_counter(); f(); t = min(t, time.perf_counter() - s)
    return t


def series(kind, n):
    if kind == "noise":
        return rng.normal(size=n)
    if kind == "walk":
        return rng.normal(size=n).cumsum()
    if kind == "sine+noise":
        return np.sin(np.arange(n) * 0.05) + 0.1 * rng.normal(size=n)
    if kind == "monotone(worst)":
        return np.sqrt(np.arange(n, dtype=float)) + 1e-3 * rng.normal(size=n)


def row(label, t_rg, t_tv):
    print(f"  {label:34s} rustygraph {t_rg*1e3:10.3f} ms   ts2vg {t_tv*1e3:10.3f} ms   speedup {t_tv/t_rg:7.1f}x", flush=True)


threads = os.environ.get("RAYON_NUM_THREADS", "all")
print(f"threads={threads}  cpu={os.cpu_count()}")

print("\n== Natural VG, single series ==")
for kind in ("noise", "walk", "sine+noise"):
    for n in (100, 1_000, 10_000, 100_000, 1_000_000):
        y = series(kind, n)
        r = 3 if n >= 1_000_000 else R
        row(f"{kind:10s} n={n:>9,}", best(lambda: rg.natural_visibility_edges(y), r),
            best(lambda: NaturalVG().build(y), r))

print("\n== Natural VG, worst case (monotone concave) ==")
for n in (1_000, 10_000, 50_000):
    y = series("monotone(worst)", n)
    row(f"monotone n={n:>9,}", best(lambda: rg.natural_visibility_edges(y), 3),
        best(lambda: NaturalVG().build(y), 3))

print("\n== Horizontal VG, single series ==")
for n in (1_000, 100_000, 1_000_000):
    y = series("noise", n)
    row(f"noise n={n:>9,}", best(lambda: rg.horizontal_visibility_edges(y)),
        best(lambda: HorizontalVG().build(y)))

print("\n== Many small windows (thesis use case), 10,000 windows ==")
for w in (15, 23, 64, 256):
    Y = rng.normal(size=(10_000, w))
    t_tv = best(lambda: [NaturalVG().build(v) for v in Y], 3)
    row(f"w={w:<4d} per-call edges()", best(lambda: [rg.natural_visibility_edges(v) for v in Y], 3), t_tv)
    row(f"w={w:<4d} natural_visibility_batch", best(lambda: rg.natural_visibility_batch(Y), 3), t_tv)

print("\n== Object API (VisibilityGraph, hash-map backed) ==")
for n in (1_000, 100_000):
    y = series("noise", n)
    row(f"natural_visibility n={n:>9,}", best(lambda: rg.natural_visibility(y), 3), best(lambda: NaturalVG().build(y), 3))
