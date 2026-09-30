//! Output-sensitive natural visibility via a segment tree of upper convex hulls.
//!
//! Seen from point `t` looking left, the visible points are the successive
//! records of the slope `σ_j = (y_j - y_t) / (x_t - x_j)`. The next record after
//! the current best slope `s = bdy / bdx` is the nearest `j` with
//!
//! ```text
//! bdx·y_j + bdy·x_j  >  bdx·y_t + bdy·x_t          (looking left)
//! bdx·y_j - bdy·x_j  >  bdx·y_t - bdy·x_t          (looking right)
//! ```
//!
//! i.e. a "nearest index whose linear functional exceeds C" query. The maximum
//! of a linear functional `a·y + b·x` (`a > 0`) over a range is attained on the
//! range's upper convex hull, so a segment tree whose nodes store upper hulls
//! answers each query in O(log² n). Visible points cost O(log² n) each and
//! blocked points cost nothing: total O((n + E) log² n) worst case, versus the
//! Θ(n²) worst case of scan-based algorithms (e.g. on monotone concave series).
//!
//! Hulls are only used to *prune* subtrees (with a conservative tolerance);
//! the visibility decision at the leaves uses exactly the same
//! cross-multiplication test as the scan kernel, so both engines return
//! identical edges.

use super::fast::{Edge, Positions};

/// Leaves of this many points are scanned directly.
const LEAF: usize = 16;

/// Segment tree of upper hulls over `(x_i, y_i)`.
pub struct HullTree<'a> {
    y: std::borrow::Cow<'a, [f64]>,
    x: Vec<f64>,
    n: usize,
    /// Number of leaf blocks (power of two).
    blocks: usize,
    /// Hull vertices of node `v` are `verts[off[v]..off[v + 1]]`.
    off: Vec<u32>,
    verts: Vec<u32>,
}

#[inline(always)]
fn cross(x: &[f64], y: &[f64], o: u32, a: u32, b: u32) -> f64 {
    let (o, a, b) = (o as usize, a as usize, b as usize);
    (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])
}

fn upper_hull(x: &[f64], y: &[f64], pts: impl Iterator<Item = u32>, out: &mut Vec<u32>) {
    let start = out.len();
    for p in pts {
        while out.len() >= start + 2
            && cross(x, y, out[out.len() - 2], out[out.len() - 1], p) >= 0.0
        {
            out.pop();
        }
        out.push(p);
    }
}

impl<'a> HullTree<'a> {
    /// Builds the tree in O(n log n) time.
    pub fn new<P: Positions>(y: &'a [f64], pos: &P) -> Self {
        let x: Vec<f64> = (0..y.len()).map(|i| pos.at(i)).collect();
        Self::build(std::borrow::Cow::Borrowed(y), x)
    }

    /// Builds a tree that owns its data (used by streaming blocks).
    pub fn owned(y: Vec<f64>, x: Vec<f64>) -> HullTree<'static> {
        HullTree::build(std::borrow::Cow::Owned(y), x)
    }

    fn build(y: std::borrow::Cow<'a, [f64]>, x: Vec<f64>) -> Self {
        let n = y.len();
        let nblocks = n.div_ceil(LEAF).max(1);
        let blocks = nblocks.next_power_of_two();
        let nodes = 2 * blocks;
        let mut hulls: Vec<Vec<u32>> = vec![Vec::new(); nodes];
        for b in 0..nblocks {
            let (lo, hi) = (b * LEAF, ((b + 1) * LEAF).min(n));
            let mut h = Vec::with_capacity(hi - lo);
            upper_hull(&x, &y, (lo as u32)..(hi as u32), &mut h);
            hulls[blocks + b] = h;
        }
        for v in (1..blocks).rev() {
            let (l, r) = (&hulls[2 * v], &hulls[2 * v + 1]);
            let mut h = Vec::with_capacity(l.len() + r.len());
            upper_hull(&x, &y, l.iter().chain(r.iter()).copied(), &mut h);
            hulls[v] = h;
        }
        let mut off = Vec::with_capacity(nodes + 1);
        let total: usize = hulls.iter().map(Vec::len).sum();
        let mut verts = Vec::with_capacity(total);
        for h in &hulls {
            off.push(verts.len() as u32);
            verts.extend_from_slice(h);
        }
        off.push(verts.len() as u32);
        Self {
            y,
            x,
            n,
            blocks,
            off,
            verts,
        }
    }

    #[inline(always)]
    fn f(&self, i: u32, a: f64, b: f64) -> f64 {
        a * self.y[i as usize] + b * self.x[i as usize]
    }

    /// Max of `a·y + b·x` over node `v` (a > 0): binary search on the hull.
    fn node_max(&self, v: usize, a: f64, b: f64) -> f64 {
        let h = &self.verts[self.off[v] as usize..self.off[v + 1] as usize];
        if h.is_empty() {
            return f64::NEG_INFINITY;
        }
        let (mut lo, mut hi) = (0usize, h.len() - 1);
        while lo < hi {
            let m = (lo + hi) / 2;
            if self.f(h[m], a, b) <= self.f(h[m + 1], a, b) {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        // Guard against float non-unimodality near flat directions.
        let mut best = self.f(h[lo], a, b);
        if lo > 0 {
            best = best.max(self.f(h[lo - 1], a, b));
        }
        if lo + 1 < h.len() {
            best = best.max(self.f(h[lo + 1], a, b));
        }
        best
    }

    #[inline(always)]
    fn node_range(&self, v: usize) -> (usize, usize) {
        let depth = usize::BITS - 1 - v.leading_zeros();
        let span = self.blocks >> depth; // blocks covered by v
        let first = (v - (1 << depth)) * span;
        (
            (first * LEAF).min(self.n),
            ((first + span) * LEAF).min(self.n),
        )
    }

    /// Rightmost `j` in `[lo, r)` accepted by `pred`, pruning nodes whose max of
    /// `a·y + b·x` is below `c - tol`.
    #[allow(clippy::too_many_arguments)]
    fn rightmost(
        &self,
        v: usize,
        lo: usize,
        r: usize,
        a: f64,
        b: f64,
        c: f64,
        tol: f64,
        pred: &impl Fn(usize) -> bool,
        work: &mut usize,
    ) -> Option<usize> {
        let (s, e) = self.node_range(v);
        if s >= r || e <= lo || s >= e {
            return None;
        }
        *work += 8; // node visit + hull binary search, in scan-step units
        if s >= lo && e <= r && self.node_max(v, a, b) <= c - tol {
            return None;
        }
        if v >= self.blocks {
            let (l, h) = (s.max(lo), e.min(r));
            *work += h - l;
            return (l..h).rev().find(|&j| pred(j));
        }
        self.rightmost(2 * v + 1, lo, r, a, b, c, tol, pred, work)
            .or_else(|| self.rightmost(2 * v, lo, r, a, b, c, tol, pred, work))
    }

    /// Leftmost `j` in `[l, hi)` accepted by `pred` (see [`Self::rightmost`]).
    #[allow(clippy::too_many_arguments)]
    fn leftmost(
        &self,
        v: usize,
        l: usize,
        hi: usize,
        a: f64,
        b: f64,
        c: f64,
        tol: f64,
        pred: &impl Fn(usize) -> bool,
        work: &mut usize,
    ) -> Option<usize> {
        let (s, e) = self.node_range(v);
        if s >= hi || e <= l || s >= e {
            return None;
        }
        *work += 8;
        if s >= l && e <= hi && self.node_max(v, a, b) <= c - tol {
            return None;
        }
        if v >= self.blocks {
            let (a0, h0) = (s.max(l), e.min(hi));
            *work += h0 - a0;
            return (a0..h0).find(|&j| pred(j));
        }
        self.leftmost(2 * v, l, hi, a, b, c, tol, pred, work)
            .or_else(|| self.leftmost(2 * v + 1, l, hi, a, b, c, tol, pred, work))
    }

    /// Continue the left record-scan of `i` from record `j` (best slope
    /// `bdy/bdx`) over `[lo, j)` with hull queries.
    ///
    /// With `handback`, returns the current state as soon as queries cost more
    /// than the linear scan they replace (dense near-visible regions defeat the
    /// hull bound), so the caller can resume scanning. Returns `None` when done.
    #[allow(clippy::too_many_arguments)]
    pub fn left_from(
        &self,
        i: usize,
        lo: usize,
        mut j: usize,
        mut bdy: f64,
        mut bdx: f64,
        handback: bool,
        out: &mut Vec<Edge>,
    ) -> Option<(usize, f64, f64)> {
        let (y, x) = (&self.y[..], &self.x);
        let (yi, xi) = (y[i], x[i]);
        let (mut spent, mut saved) = (0usize, 0usize);
        loop {
            let (a, b) = (bdx, bdy);
            let c = a * yi + b * xi;
            let tol = tolerance(a, b, yi, xi);
            let pred = |k: usize| (y[k] - yi) * bdx > bdy * (xi - x[k]);
            let mut work = 0;
            let k = self.rightmost(1, lo, j, a, b, c, tol, &pred, &mut work)?;
            out.push((k as u32, i as u32));
            spent += work;
            saved += j - k;
            j = k;
            bdy = y[k] - yi;
            bdx = xi - x[k];
            if handback && spent > 2 * saved + 64 {
                return Some((j, bdy, bdx));
            }
        }
    }

    /// Right-side counterpart of [`Self::left_from`] over `(j, hi]`.
    #[allow(clippy::too_many_arguments)]
    pub fn right_from(
        &self,
        i: usize,
        hi: usize,
        mut j: usize,
        mut bdy: f64,
        mut bdx: f64,
        handback: bool,
        out: &mut Vec<Edge>,
    ) -> Option<(usize, f64, f64)> {
        let (y, x) = (&self.y[..], &self.x);
        let (yi, xi) = (y[i], x[i]);
        let (mut spent, mut saved) = (0usize, 0usize);
        while bdy < 0.0 {
            let (a, b) = (bdx, -bdy);
            let c = a * yi + b * xi;
            let tol = tolerance(a, b, yi, xi);
            let pred = |k: usize| (y[k] - yi) * bdx > bdy * (x[k] - xi);
            let mut work = 0;
            let k = self.leftmost(1, j + 1, hi + 1, a, b, c, tol, &pred, &mut work)?;
            out.push((i as u32, k as u32));
            spent += work;
            saved += k - j;
            j = k;
            bdy = y[k] - yi;
            bdx = x[k] - xi;
            if handback && spent > 2 * saved + 64 {
                return Some((j, bdy, bdx));
            }
        }
        None
    }

    /// Natural-visibility edges incident to `i` within its segment `[lo, hi]`,
    /// found entirely by record queries.
    pub fn scan_point(&self, i: usize, lo: usize, hi: usize, out: &mut Vec<Edge>) {
        let (y, x) = (&self.y[..], &self.x);
        let (yi, xi) = (y[i], x[i]);
        if i > lo {
            out.push(((i - 1) as u32, i as u32));
            self.left_from(i, lo, i - 1, y[i - 1] - yi, xi - x[i - 1], false, out);
        }
        if i < hi {
            out.push((i as u32, (i + 1) as u32));
            self.right_from(i, hi, i + 1, y[i + 1] - yi, x[i + 1] - xi, false, out);
        }
    }

    /// Rightmost local index `j` in `[lo, r)` with `pred(j)`, pruning subtrees
    /// whose max of `a·y + b·x` (a > 0) is at most `c - tol`. Adds the query
    /// cost (in scan-step units) to `work`.
    #[allow(clippy::too_many_arguments)]
    pub fn rightmost_where(
        &self,
        lo: usize,
        r: usize,
        a: f64,
        b: f64,
        c: f64,
        tol: f64,
        pred: &impl Fn(usize) -> bool,
        work: &mut usize,
    ) -> Option<usize> {
        self.rightmost(1, lo, r, a, b, c, tol, pred, work)
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the tree holds no points.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Positions of the samples (as used by the tree).
    pub fn x(&self) -> &[f64] {
        &self.x
    }
}

/// Conservative pruning tolerance for the functional `a·y + b·x` around `(xi, yi)`.
#[inline(always)]
pub fn tolerance(a: f64, b: f64, yi: f64, xi: f64) -> f64 {
    1e-12 * (a.abs() * (yi.abs() + 1.0) + b.abs() * (xi.abs() + 1.0))
}
