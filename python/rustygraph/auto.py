"""One entry point that picks the right visibility-graph engine for the scenario.

``visibility(data, kind=...)`` looks at the input and chooses:

========================================  =====================================================
Scenario                                  Engine
========================================  =====================================================
torch tensor that requires grad /         differentiable soft VG (``torch_vg.soft_visibility``)
``differentiable=True``
torch tensor on CUDA/MPS, small windows   exact VG on device (``torch_vg.exact_visibility``)
1-D series                                Rust kernel; natural VG measures its exact scan cost
                                          and switches to output-sensitive hull queries when
                                          scanning would be super-linear
2-D, univariate kind                      rows are independent series -> parallel batch
2-D, vector kind                          one multivariate series (time, features)
3-D, vector kind                          many multivariate windows -> parallel batch
kind="multiplex"                          one layer per variable of a (time, features) series
kind="average"                            multiplex layers averaged into one weighted graph
========================================  =====================================================

Use :class:`rustygraph.VisibilityStream` for sample-by-sample (autoregressive) use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from . import _rustygraph as _rg

__all__ = ["visibility", "UNIVARIATE_KINDS", "VECTOR_KINDS"]

UNIVARIATE_KINDS = ("natural", "horizontal")
VECTOR_KINDS = ("vector_natural", "vector_horizontal")
_ALIASES = {"nvg": "natural", "hvg": "horizontal", "vector": "vector_natural", "vvg": "vector_natural", "hvvg": "vector_horizontal"}
_TORCH_DEVICE_MAX_WINDOW = 128


def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _dense(edges: np.ndarray, n: int, weights: Optional[np.ndarray] = None, symmetric: bool = True) -> np.ndarray:
    A = np.zeros((n, n))
    A[edges[:, 0], edges[:, 1]] = 1.0 if weights is None else weights
    return A + A.T if symmetric else A


def _format(edges: np.ndarray, n: int, output: str, weights: Optional[np.ndarray] = None):
    if output == "edges":
        return edges if weights is None else (edges, weights)
    if output == "edge_index":
        ei = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
        return ei if weights is None else (ei, np.concatenate([weights, weights]))
    if output == "adjacency":
        return _dense(edges, n, weights)
    raise ValueError(f"output must be 'edges', 'edge_index' or 'adjacency', got {output!r}")


def _format_batch(edges: np.ndarray, offsets: np.ndarray, n: int, output: str):
    if output == "edges":
        return edges, offsets
    if output == "edge_index":  # PyG-style block-diagonal batch
        row = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        g = edges + (row * n)[:, None]
        return np.concatenate([g.T, g[:, ::-1].T], axis=1), row
    if output == "adjacency":
        B = len(offsets) - 1
        A = np.zeros((B, n, n))
        row = np.repeat(np.arange(B), np.diff(offsets))
        A[row, edges[:, 0], edges[:, 1]] = 1.0
        return A + A.transpose(0, 2, 1)
    raise ValueError(f"output must be 'edges', 'edge_index' or 'adjacency', got {output!r}")


def _torch_path(data, kind, t, differentiable, tau, output):
    import importlib

    torch_vg = importlib.import_module(".torch_vg", __package__)  # requires torch

    x = data
    w = x.shape[-2] if kind in VECTOR_KINDS else x.shape[-1]
    wants_grad = differentiable if differentiable is not None else bool(getattr(x, "requires_grad", False))
    if wants_grad:
        A = torch_vg.soft_visibility(x, kind=kind, tau=tau, t=t)
        plan = {"engine": "torch.soft", "device": str(x.device), "tau": tau}
    elif x.device.type != "cpu" and w <= _TORCH_DEVICE_MAX_WINDOW:
        A = torch_vg.exact_visibility(x, kind=kind, t=t)
        plan = {"engine": "torch.exact", "device": str(x.device)}
    else:
        return None
    if output != "adjacency":
        raise ValueError("torch engines return dense adjacency; use output='adjacency'")
    return torch_vg.to_undirected(A), plan


def visibility(
    data: Any,
    kind: str = "natural",
    *,
    t: Optional[Any] = None,
    weight: Optional[str] = None,
    output: str = "edges",
    differentiable: Optional[bool] = None,
    tau: float = 0.1,
    algorithm: str = "auto",
    explain: bool = False,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """Build a visibility graph with the engine best suited to ``data``.

    Args:
        data: NumPy array / list / torch tensor. See the module table for shapes.
        kind: "natural", "horizontal", "vector_natural", "vector_horizontal",
            "multiplex" or "average" (aliases: nvg, hvg, vvg, vector, hvvg).
        t: optional strictly increasing sample positions (natural kinds).
        weight: vector kinds only, a vector-vis-graph weight method name.
        output: "edges" (default), "edge_index" (PyG style, both directions) or
            "adjacency" (dense, symmetric).
        differentiable: force (True) / forbid (False) the soft torch engine;
            default: soft iff a torch input requires grad.
        tau: temperature of the soft engine.
        algorithm: natural 1-D engine, "auto" | "scan" | "hull".
        explain: also return a dict describing the chosen engine.
    """
    kind = _ALIASES.get(kind, kind)

    if _is_torch(data):
        if kind not in UNIVARIATE_KINDS + VECTOR_KINDS:
            raise ValueError(f"torch inputs support kinds {UNIVARIATE_KINDS + VECTOR_KINDS}")
        res = _torch_path(data, kind, t, differentiable, tau, output)
        if res is not None:
            return res if explain else res[0]
        data = data.detach().cpu().numpy()  # CPU tensor, no grad: Rust is fastest
    elif differentiable:
        raise ValueError("differentiable=True needs a torch tensor input")

    x = np.asarray(data, dtype=np.float64)
    plan: Dict[str, Any] = {"kind": kind, "shape": x.shape}

    if kind in ("multiplex", "average"):
        if x.ndim != 2:
            raise ValueError("multiplex/average expect a (time, features) array")
        T, N = x.shape
        e, off = _rg.natural_visibility_batch(np.ascontiguousarray(x.T))
        plan.update(engine="rust.batch(layers)", layers=N)
        if kind == "multiplex":
            res = _format_batch(e, off, T, output)
        else:
            row = np.repeat(np.arange(N), np.diff(off))
            keys = e[:, 0] * T + e[:, 1]
            uniq, counts = np.unique(keys, return_counts=True)
            edges = np.stack([uniq // T, uniq % T], axis=1)
            res = _format(edges, T, output, counts / N)
            del row
        return (res, plan) if explain else res

    if kind in UNIVARIATE_KINDS:
        if x.ndim == 1:
            if kind == "natural":
                plan.update(engine="rust", **_rg.natural_visibility_plan(x))
                if algorithm != "auto":
                    plan["algorithm"] = algorithm
                edges = _rg.natural_visibility_edges(x, t, algorithm)
            else:
                plan.update(engine="rust", algorithm="stack")
                edges = _rg.horizontal_visibility_edges(x)
            res = _format(edges, len(x), output)
        elif x.ndim == 2:
            if t is not None:
                raise ValueError("t is not supported for batched univariate input")
            fn = _rg.natural_visibility_batch if kind == "natural" else _rg.horizontal_visibility_batch
            plan.update(engine="rust.batch", series=x.shape[0])
            res = _format_batch(*fn(np.ascontiguousarray(x)), x.shape[1], output)
        else:
            raise ValueError(f"{kind}: expected (n,) or (B, n) input, got {x.shape}")
        return (res, plan) if explain else res

    if kind in VECTOR_KINDS:
        if x.ndim in (1, 2):
            fn = _rg.natural_vector_visibility_edges if kind == "vector_natural" else None
            plan.update(engine="rust.vector")
            if fn is not None:
                out = fn(x, t, weight) if weight else fn(x, t)
            else:
                if t is not None:
                    raise ValueError("t is not used by horizontal visibility")
                out = _rg.horizontal_vector_visibility_edges(x, weight) if weight else _rg.horizontal_vector_visibility_edges(x)
            edges, weights = out if isinstance(out, tuple) else (out, None)
            res = _format(edges, x.shape[0], output, weights)
        elif x.ndim == 3:
            if t is not None or weight is not None:
                raise ValueError("t / weight are not supported for batched windows")
            fn = _rg.natural_vector_visibility_batch if kind == "vector_natural" else _rg.horizontal_vector_visibility_batch
            plan.update(engine="rust.vector.batch", windows=x.shape[0])
            res = _format_batch(*fn(np.ascontiguousarray(x)), x.shape[1], output)
        else:
            raise ValueError(f"{kind}: expected (n, d) or (B, n, d) input, got {x.shape}")
        return (res, plan) if explain else res

    raise ValueError(f"unknown kind {kind!r}")
