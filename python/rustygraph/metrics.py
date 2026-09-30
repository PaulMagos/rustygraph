"""Visibility-graph fidelity metrics for synthetic (multivariate) time series.

Classical generation metrics (moments, Wasserstein, KS, MMD on values) compare
*marginal* distributions only: shuffling a series in time leaves them unchanged.
Visibility-graph descriptors instead capture temporal structure (convexity,
extremes, periodicity, irreversibility) and, for multivariate data, cross-channel
structure. ``vg_fidelity(real, synth)`` compares them.

Descriptors (per graph type):

* degree distribution of natural / horizontal VGs (Lacasa et al. 2008; Luque et al. 2009);
* size-4 sequential motif profile (Iacovacci & Lacasa 2016);
* time irreversibility: KL divergence between out- and in-degree distributions of
  the directed HVG (Lacasa et al. 2012);
* temporal-structure index: JSD between the HVG degree distribution and that of
  the same windows shuffled in time, the exact finite-size i.i.d. null (whose
  infinite-length law is P(k) = (1/3)(2/3)^(k-2), Luque et al. 2009): 0 means
  "no temporal structure";
* multivariate (N > 1): vector-VG degrees and motifs (Ren & Jin 2019), and the
  multiplex edge overlap and interlayer degree mutual information
  (Lacasa, Nicosia & Latora 2015).

All distribution distances are Jensen-Shannon divergences in bits (range [0, 1]).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ._rustygraph import (
    horizontal_visibility_batch,
    natural_vector_visibility_batch,
    natural_visibility_batch,
    visibility_motifs,
)

__all__ = ["as_windows", "vg_descriptors", "vg_fidelity", "hvg_iid_null", "jensen_shannon"]

_EPS = 1e-12


def as_windows(data: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """Return a (B, w, N) float64 array of windows.

    (T,) and (T, N) inputs are cut into non-overlapping windows of length
    ``window`` (default ``min(T, 128)``); (B, w, N) inputs are used as given.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim == 2:
        T = x.shape[0]
        w = min(T, 128) if window is None else int(window)
        if w < 4 or w > T:
            raise ValueError(f"window must be in [4, {T}], got {w}")
        B = T // w
        x = x[: B * w].reshape(B, w, x.shape[1])
    elif x.ndim != 3:
        raise ValueError(f"expected (T,), (T, N) or (B, w, N) data, got shape {x.shape}")
    if x.shape[1] < 4:
        raise ValueError("windows need at least 4 time steps")
    if not np.isfinite(x).all():
        raise ValueError("data contains NaN or inf")
    return np.ascontiguousarray(x)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two (unnormalised) histograms."""
    p = np.asarray(p, float) + _EPS
    q = np.asarray(q, float) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def hvg_iid_null(max_degree: int) -> np.ndarray:
    """P(k) = (1/3)(2/3)^(k-2), k >= 2, for i.i.d. series (tail folded into the last bin)."""
    k = np.arange(max_degree + 1)
    p = np.where(k >= 2, (1 / 3) * (2 / 3) ** (k - 2.0), 0.0)
    p[-1] += 1.0 - p.sum()
    return p


def _degrees(edges: np.ndarray, offsets: np.ndarray, rows: int, w: int):
    row = np.repeat(np.arange(rows), np.diff(offsets))
    src = row * w + edges[:, 0]
    dst = row * w + edges[:, 1]
    out_deg = np.bincount(src, minlength=rows * w)
    in_deg = np.bincount(dst, minlength=rows * w)
    return (in_deg + out_deg).reshape(rows, w), out_deg.reshape(rows, w), in_deg.reshape(rows, w)


def _hist(deg: np.ndarray, max_degree: int) -> np.ndarray:
    return np.bincount(np.minimum(deg.ravel(), max_degree), minlength=max_degree + 1).astype(float)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p + _EPS
    q = q + _EPS
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _multiplex(edges: np.ndarray, offsets: np.ndarray, deg: np.ndarray, B: int, N: int, w: int, max_degree: int):
    """Edge overlap and mean interlayer degree mutual information; layers = variables."""
    rows = B * N
    row = np.repeat(np.arange(rows), np.diff(offsets))
    window = row // N
    keys = (window * w + edges[:, 0]) * w + edges[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    overlap = float(counts.sum() / (N * max(len(counts), 1)))
    # interlayer MI of degrees, pooled over windows: deg is (B*N, w) -> (N, B*w)
    d = np.minimum(deg.reshape(B, N, w).transpose(1, 0, 2).reshape(N, B * w), max_degree)
    K = max_degree + 1
    mis = []
    for a in range(N):
        for b in range(a + 1, N):
            joint = np.bincount(d[a] * K + d[b], minlength=K * K).reshape(K, K).astype(float)
            joint /= joint.sum()
            pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
            nz = joint > 0
            mis.append(float(np.sum(joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz]))))
    return overlap, float(np.mean(mis)) if mis else 0.0


def vg_descriptors(data: np.ndarray, window: Optional[int] = None, max_degree: int = 32, seed: int = 0) -> Dict[str, object]:
    """Visibility-graph descriptors of a dataset (see module docstring)."""
    x = as_windows(data, window)
    B, w, N = x.shape
    uni = np.ascontiguousarray(x.transpose(0, 2, 1).reshape(B * N, w))  # one row per (window, variable)
    out: Dict[str, object] = {"shape": (B, w, N)}

    e, off = natural_visibility_batch(uni)
    deg, _, _ = _degrees(e, off, B * N, w)
    out["nvg_degree"] = _hist(deg, max_degree)
    out["nvg_motifs"] = visibility_motifs(uni, "natural").sum(0).astype(float)
    if N > 1:
        out["multiplex_overlap"], out["multiplex_mi"] = _multiplex(e, off, deg, B, N, w, max_degree)

    e, off = horizontal_visibility_batch(uni)
    deg, out_deg, in_deg = _degrees(e, off, B * N, w)
    out["hvg_degree"] = _hist(deg, max_degree)
    out["hvg_motifs"] = visibility_motifs(uni, "horizontal").sum(0).astype(float)
    inner = slice(1, w - 1)  # window edges truncate degrees
    out["irreversibility"] = _kl(_hist(out_deg[:, inner], max_degree), _hist(in_deg[:, inner], max_degree))
    # Temporal-structure index: distance to the exact finite-size i.i.d. null,
    # obtained by shuffling every window in time (same values, same length).
    # hvg_iid_null() gives the infinite-length law P(k) = (1/3)(2/3)^(k-2).
    rng = np.random.default_rng(seed)
    shuffled = np.take_along_axis(uni, rng.random(uni.shape).argsort(axis=1), axis=1)
    e0, off0 = horizontal_visibility_batch(np.ascontiguousarray(shuffled))
    deg0, _, _ = _degrees(e0, off0, B * N, w)
    out["hvg_iid_distance"] = jensen_shannon(_hist(deg, max_degree), _hist(deg0, max_degree))

    if N > 1:
        e, off = natural_vector_visibility_batch(x)
        deg, _, _ = _degrees(e, off, B, w)
        out["vvg_degree"] = _hist(deg, max_degree)
        out["vvg_motifs"] = visibility_motifs(x, "vector_natural").sum(0).astype(float)
    return out


_DISTS = ("nvg_degree", "nvg_motifs", "hvg_degree", "hvg_motifs", "vvg_degree", "vvg_motifs")
_SCALARS = ("irreversibility", "hvg_iid_distance", "multiplex_overlap", "multiplex_mi")


def _compare(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for k in _DISTS:
        if k in a and k in b:
            res[f"jsd_{k}"] = jensen_shannon(a[k], b[k])  # type: ignore[arg-type]
    for k in _SCALARS:
        if k in a and k in b:
            res[f"absdiff_{k}"] = abs(float(a[k]) - float(b[k]))  # type: ignore[arg-type]
    jsd = [v for k, v in res.items() if k.startswith("jsd_")]
    res["vg_divergence"] = float(np.mean(jsd))
    return res


def vg_fidelity(
    real: np.ndarray,
    synth: np.ndarray,
    window: Optional[int] = None,
    max_degree: int = 32,
    baseline: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare visibility-graph descriptors of real vs synthetic data.

    Returns JSD (bits, lower is better) per distribution descriptor, absolute
    differences of scalar descriptors, their raw values, and ``vg_divergence``
    (mean JSD). With ``baseline=True`` the same comparison between two random
    halves of the real windows is reported as ``baseline_*``: differences at or
    below the baseline are within sampling noise.
    """
    xr, xs = as_windows(real, window), as_windows(synth, window)
    if xr.shape[1:] != xs.shape[1:]:
        raise ValueError(f"window shape mismatch: real {xr.shape[1:]} vs synth {xs.shape[1:]}")
    dr = vg_descriptors(xr, max_degree=max_degree, seed=seed)
    ds = vg_descriptors(xs, max_degree=max_degree, seed=seed)
    res = _compare(dr, ds)
    for k in _SCALARS:
        if k in dr:
            res[f"real_{k}"] = float(dr[k])  # type: ignore[arg-type]
            res[f"synth_{k}"] = float(ds[k])  # type: ignore[arg-type]
    if baseline and xr.shape[0] >= 4:
        idx = np.random.default_rng(seed).permutation(xr.shape[0])
        h = len(idx) // 2
        base = _compare(vg_descriptors(xr[idx[:h]], max_degree=max_degree), vg_descriptors(xr[idx[h:]], max_degree=max_degree))
        res.update({f"baseline_{k}": v for k, v in base.items()})
    return res
