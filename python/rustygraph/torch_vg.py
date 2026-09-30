"""PyTorch visibility graphs: exact batched adjacency, soft relaxation, layer, statistics.

Semantics mirror rustygraph's Rust kernels:

* edges ``(a, b)`` with ``a < b``; adjacent pairs are always visible;
* visibility is strict: an intermediate point exactly on the sight line blocks;
* natural: ``(y_k - y_a)(t_b - t_a) < (y_b - y_a)(t_k - t_a)`` for all ``a < k < b``;
* horizontal: ``y_k < min(y_a, y_b)`` for all ``a < k < b``;
* vector kinds (``X`` of shape ``(w, d)``): for anchor ``a`` use the scaled
  projections ``q_k = X_k . X_a`` and apply the univariate criterion on ``q``.

Adjacency matrices are upper triangular and directed: ``A[..., a, b]`` is set for
``a < b`` visible. Use :func:`to_undirected` for the symmetric version.

Requires ``torch`` (optional dependency of rustygraph).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "KINDS",
    "exact_visibility",
    "soft_visibility",
    "to_undirected",
    "VisibilityLayer",
    "edge_index_from_adjacency",
    "soft_degree",
    "soft_degree_histogram",
]

KINDS = ("natural", "horizontal", "vector_natural", "vector_horizontal")
MODES = ("auto", "exact", "soft")
DEFAULT_MAX_ELEMS = 1 << 26
EPS = 1e-8

ArrayLike = Any  # Tensor, numpy array or nested sequence


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def _check_kind(kind: str) -> bool:
    """Validate ``kind``; return True for the vector kinds."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind.startswith("vector")


def _as_float(x: ArrayLike) -> Tensor:
    x = torch.as_tensor(x)
    if x.is_complex():
        raise ValueError("complex input is not supported")
    if not x.is_floating_point():
        x = x.to(torch.float32 if x.device.type == "mps" else torch.float64)
    return x


def _prepare(
    x: ArrayLike, kind: str, t: Optional[ArrayLike]
) -> Tuple[Tensor, Tensor, Tuple[int, ...], int, bool]:
    """Validate and flatten input.

    Returns ``(xf, T, lead, w, vec)`` with ``xf`` of shape ``(N, w)`` or
    ``(N, w, d)``, positions ``T`` of shape ``(N, w)``, and the leading shape.
    """
    vec = _check_kind(kind)
    x = _as_float(x)
    min_dim = 2 if vec else 1
    if x.dim() < min_dim:
        raise ValueError(f"kind={kind!r} needs at least {min_dim}-D input, got shape {tuple(x.shape)}")
    w = x.shape[-2] if vec else x.shape[-1]
    if w < 1 or (vec and x.shape[-1] < 1):
        raise ValueError(f"empty series/window: shape {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("input contains NaN or infinite values")
    lead = tuple(x.shape[:-2] if vec else x.shape[:-1])
    n = math.prod(lead)
    xf = x.reshape(n, w, x.shape[-1]) if vec else x.reshape(n, w)
    return xf, _positions(t, kind, lead, w, x), lead, w, vec


def _positions(t: Optional[ArrayLike], kind: str, lead: Tuple[int, ...], w: int, x: Tensor) -> Tensor:
    n = math.prod(lead)
    if t is None:
        return torch.arange(w, dtype=x.dtype, device=x.device).expand(n, w)
    if kind.endswith("horizontal"):
        raise ValueError(f"positions t are not used by kind={kind!r}; pass t=None")
    t = torch.as_tensor(t, device=x.device).to(x.dtype)
    if t.dim() < 1 or t.shape[-1] != w:
        raise ValueError(f"t must have last dimension {w}, got shape {tuple(t.shape)}")
    try:
        ok = torch.broadcast_shapes(tuple(t.shape[:-1]), lead) == lead
    except RuntimeError:
        ok = False
    if not ok:
        raise ValueError(f"t shape {tuple(t.shape)} does not broadcast to {lead + (w,)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("t contains NaN or infinite values")
    if w > 1 and not bool((t.diff(dim=-1) > 0).all()):
        raise ValueError("t must be strictly increasing")
    return t.expand(*lead, w).reshape(n, w)


def _row_chunks(n_rows: int, w: int, max_elems: int):
    """Yield ``(start, stop)`` ranges over flattened (series, anchor) rows."""
    if max_elems < 1:
        raise ValueError(f"max_elems must be positive, got {max_elems}")
    step = max(1, max_elems // max(1, w * w))
    for s in range(0, n_rows, step):
        yield s, min(n_rows, s + step)


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------

def _exact_profiles(xf: Tensor, bi: Tensor, a: Tensor, vec: bool) -> Tensor:
    """Per-row profile ``P[r, k]`` (series values or ``X_k . X_a``), shape ``(R, w)``.

    Dot products are accumulated sequentially over ``d`` (no BLAS / FMA) so that
    float64 results are bit-identical to the Rust kernel.
    """
    if not vec:
        return xf[bi]
    p = None
    for j in range(xf.shape[-1]):
        col = xf[..., j]
        term = col[bi] * col[bi, a][:, None]
        p = term if p is None else p + term
    return p


def _exact_rows(P: Tensor, T: Tensor, a: Tensor, horizontal: bool) -> Tensor:
    """Exact visibility rows ``vis[r, b]`` for anchors ``a`` (shape ``(R,)``)."""
    w = P.shape[-1]
    ks = torch.arange(w, device=P.device)
    after = ks[None, :] > a[:, None]  # (R, w): b > a
    pa = P.gather(1, a[:, None])
    if horizontal:
        # max over k in (a, b) of P_k via shifted cummax (exact).
        masked = P.masked_fill(~after, -math.inf)
        cm = masked.cummax(dim=-1).values
        m = torch.cat([torch.full_like(cm[:, :1], -math.inf), cm[:, :-1]], dim=-1)
        return after & (m < torch.minimum(pa, P))
    dp = P - pa
    dt = T - T.gather(1, a[:, None])
    # index [r, b, k]
    blocked = ~(dp[:, None, :] * dt[:, :, None] < dp[:, :, None] * dt[:, None, :])
    between = after[:, None, :] & (ks[None, None, :] < ks[None, :, None])
    return after & ~(blocked & between).any(dim=-1)


@torch.no_grad()
def exact_visibility(
    x: ArrayLike,
    kind: str = "natural",
    t: Optional[ArrayLike] = None,
    max_elems: int = DEFAULT_MAX_ELEMS,
) -> Tensor:
    """Exact batched visibility adjacency on any device.

    Args:
        x: ``(..., w)`` for ``"natural"``/``"horizontal"``, ``(..., w, d)`` for
            ``"vector_natural"``/``"vector_horizontal"``.
        kind: one of :data:`KINDS`.
        t: optional strictly increasing positions ``(w,)`` or ``(..., w)``
            (natural kinds only); defaults to ``arange(w)``.
        max_elems: bound on elements of each ``(rows, w, w)`` intermediate.

    Returns:
        Bool tensor ``(..., w, w)``, upper triangular: ``A[..., a, b]`` iff
        ``a < b`` and ``b`` is visible from ``a``.
    """
    xf, T, lead, w, vec = _prepare(x, kind, t)
    n = xf.shape[0]
    out = torch.zeros(n * w, w, dtype=torch.bool, device=xf.device)
    horizontal = kind.endswith("horizontal")
    for s, e in _row_chunks(n * w, w, max_elems):
        rows = torch.arange(s, e, device=xf.device)
        bi, a = rows // w, rows % w
        P = _exact_profiles(xf, bi, a, vec)
        out[s:e] = _exact_rows(P, T[bi], a, horizontal)
    return out.reshape(*lead, w, w)


def to_undirected(A: Tensor) -> Tensor:
    """Symmetrise an upper-triangular adjacency: ``A | A^T`` (bool) or ``A + A^T``."""
    At = A.transpose(-1, -2)
    return A | At if A.dtype == torch.bool else A + At


# ---------------------------------------------------------------------------
# soft
# ---------------------------------------------------------------------------

def _soft_profiles(xf: Tensor, vec: bool, normalize: bool) -> Tuple[Tensor, Tensor]:
    """Profiles ``(N, w, w)`` indexed ``[series, anchor, k]`` and per-series scale ``(N,)``."""
    if vec:
        norms = torch.sqrt((xf * xf).sum(-1) + EPS)  # safe norm, finite grad at 0
        q = (xf @ xf.transpose(-1, -2)) / norms[..., :, None]
        scale = xf.flatten(1).std(dim=-1, correction=0) if normalize else None
    else:
        q = xf[:, None, :]
        scale = xf.std(dim=-1, correction=0) if normalize else None
    if scale is None:
        scale = torch.ones(xf.shape[0], dtype=xf.dtype, device=xf.device)
    return q, scale + EPS


def _soft_rows(P: Tensor, T: Tensor, a: Tensor, denom: Tensor, horizontal: bool) -> Tensor:
    """Soft visibility rows for anchors ``a``; ``denom`` = ``tau * scale`` per row."""
    w = P.shape[-1]
    ks = torch.arange(w, device=P.device)
    after = ks[None, :] > a[:, None]
    pa = P.gather(1, a[:, None])
    if horizontal:
        m = torch.minimum(pa, P)[:, :, None] - P[:, None, :]
    else:
        dp = P - pa
        dt = T - T.gather(1, a[:, None])
        safe_dt = torch.where(after, dt, torch.ones_like(dt))
        # margin[r, b, k] = line height at t_k minus P_k
        m = (dp[:, :, None] * dt[:, None, :] - dp[:, None, :] * dt[:, :, None]) / safe_dt[:, :, None]
    between = after[:, None, :] & (ks[None, None, :] < ks[None, :, None])
    logp = F.logsigmoid(m / denom[:, None, None])
    total = torch.where(between, logp, torch.zeros_like(logp)).sum(dim=-1)
    return torch.exp(total) * after.to(P.dtype)


def soft_visibility(
    x: ArrayLike,
    kind: str = "natural",
    tau: Union[float, Tensor] = 0.1,
    t: Optional[ArrayLike] = None,
    normalize: bool = True,
    max_elems: int = DEFAULT_MAX_ELEMS,
) -> Tensor:
    """Differentiable visibility relaxation.

    ``A[a, b] = exp(sum_{a<k<b} logsigmoid(m_k / (tau * s)))`` for ``a < b``,
    where ``m_k > 0`` iff ``k`` does not block: natural ``m_k`` is the sight line
    height at ``t_k`` minus ``y_k``; horizontal ``m_k = min(y_a, y_b) - y_k``.
    Vector kinds use projections ``X_k . X_a / |X_a|``. ``s`` is the series
    (population) std when ``normalize`` else 1, making ``tau`` scale free.
    Converges to :func:`exact_visibility` as ``tau -> 0`` on non-degenerate data.

    Args:
        x, kind, t, max_elems: as in :func:`exact_visibility`.
        tau: positive temperature (float or tensor, may require grad).
        normalize: divide margins by per-series std.

    Returns:
        Float tensor ``(..., w, w)``, upper triangular, entries in ``[0, 1]``.
    """
    if isinstance(tau, (int, float)) and not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")
    xf, T, lead, w, vec = _prepare(x, kind, t)
    n = xf.shape[0]
    q, scale = _soft_profiles(xf, vec, normalize)
    tau_t = torch.as_tensor(tau, dtype=xf.dtype, device=xf.device)
    horizontal = kind.endswith("horizontal")
    parts = []
    for s, e in _row_chunks(n * w, w, max_elems):
        rows = torch.arange(s, e, device=xf.device)
        bi, a = rows // w, rows % w
        P = q[bi, a] if vec else q[bi, 0]
        parts.append(_soft_rows(P, T[bi], a, tau_t * scale[bi], horizontal))
    return torch.cat(parts, dim=0).reshape(*lead, w, w)


# ---------------------------------------------------------------------------
# layer
# ---------------------------------------------------------------------------

class VisibilityLayer(nn.Module):
    """Visibility graph layer returning a dense float adjacency ``(..., w, w)``.

    Args:
        kind: one of :data:`KINDS`.
        mode: ``"exact"``, ``"soft"`` or ``"auto"`` (soft iff grad mode is on and
            the input requires grad, else exact).
        tau: soft temperature (> 0).
        learn_tau: learn ``tau = softplus(raw_tau)``.
        normalize: see :func:`soft_visibility`.
        symmetric: return ``A + A^T`` instead of the upper-triangular ``A``.
        self_loops: add the identity.
        max_elems: chunk size bound.
    """

    def __init__(
        self,
        kind: str = "natural",
        mode: str = "auto",
        tau: float = 0.1,
        learn_tau: bool = False,
        normalize: bool = True,
        symmetric: bool = False,
        self_loops: bool = False,
        max_elems: int = DEFAULT_MAX_ELEMS,
    ) -> None:
        super().__init__()
        _check_kind(kind)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not tau > 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.kind, self.mode, self.normalize = kind, mode, normalize
        self.symmetric, self.self_loops, self.max_elems = symmetric, self_loops, max_elems
        self.learn_tau = learn_tau
        if learn_tau:
            # inverse softplus; expm1 overflows for large tau, where softplus(x) ~ x
            raw = tau if tau > 20 else math.log(math.expm1(tau))
            self.raw_tau = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("_tau", torch.tensor(float(tau)), persistent=False)

    @property
    def tau(self) -> Tensor:
        """Current temperature."""
        return F.softplus(self.raw_tau) if self.learn_tau else self._tau

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Adjacency of ``x`` (shapes as in :func:`exact_visibility`)."""
        x = _as_float(x)
        soft = self.mode == "soft" or (
            self.mode == "auto" and torch.is_grad_enabled() and x.requires_grad
        )
        if soft:
            A = soft_visibility(x, self.kind, self.tau.to(x.dtype), t, self.normalize, self.max_elems)
        else:
            A = exact_visibility(x, self.kind, t, self.max_elems).to(x.dtype)
        if self.symmetric:
            A = A + A.transpose(-1, -2)
        if self.self_loops:
            A = A + torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        return A

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}, mode={self.mode!r}, learn_tau={self.learn_tau}, symmetric={self.symmetric}"


# ---------------------------------------------------------------------------
# graph utilities and differentiable statistics
# ---------------------------------------------------------------------------

def edge_index_from_adjacency(
    A: Tensor, threshold: float = 0.0, return_weights: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """PyG-style ``(2, E)`` long edge index from ``(w, w)`` or ``(B, w, w)`` adjacency.

    Batched input is block-diagonal: node ``i`` of graph ``g`` gets index
    ``g * w + i`` (PyG batching). Entries ``> threshold`` (or ``True``) are edges,
    ordered by (graph, source, target).

    Returns:
        ``edge_index``, or ``(edge_index, weights)`` if ``return_weights``.
    """
    if A.dim() not in (2, 3) or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be (w, w) or (B, w, w), got shape {tuple(A.shape)}")
    A3 = A if A.dim() == 3 else A[None]
    mask = A3 if A3.dtype == torch.bool else A3 > threshold
    g, i, j = mask.nonzero(as_tuple=True)
    w = A3.shape[-1]
    ei = torch.stack([g * w + i, g * w + j]).long()
    return (ei, A3[g, i, j]) if return_weights else ei


def soft_degree(A: Tensor) -> Tensor:
    """Node degrees ``(..., w)`` of the undirected graph of ``A`` (self loops ignored).

    Uses the strict upper triangle, so it accepts upper-triangular or symmetric
    adjacency, bool or (soft) float. Differentiable for float input.
    """
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be (..., w, w), got shape {tuple(A.shape)}")
    if A.dtype == torch.bool:
        A = A.to(torch.get_default_dtype())
    U = torch.triu(A, diagonal=1)
    return (U + U.transpose(-1, -2)).sum(dim=-1)


def soft_degree_histogram(A: Tensor, max_k: int, bandwidth: float = 0.5) -> Tensor:
    """Smooth degree distribution ``(..., max_k + 1)`` via a Gaussian kernel.

    Each node spreads unit mass over bins ``0..max_k`` with weights
    ``softmax(-(deg - k)^2 / (2 bandwidth^2))`` (degrees beyond ``max_k`` land in
    the last bin); bins are averaged over nodes, so each histogram sums to 1.
    Suitable for L1/MMD losses between real and generated graphs.
    """
    if not isinstance(max_k, int) or max_k < 0:
        raise ValueError(f"max_k must be a non-negative int, got {max_k!r}")
    if not bandwidth > 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    deg = soft_degree(A)
    centers = torch.arange(max_k + 1, dtype=deg.dtype, device=deg.device)
    logits = -0.5 * ((deg[..., None] - centers) / bandwidth) ** 2
    return torch.softmax(logits, dim=-1).mean(dim=-2)
