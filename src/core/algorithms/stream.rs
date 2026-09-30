//! Streaming (online) visibility graphs: push one sample at a time and get the
//! edges from the past to the new point.
//!
//! Visibility between `a < b` depends only on the samples between them, so a
//! prefix's graph is an induced subgraph of the full graph and pushing a sample
//! only *adds* edges to the new node. A sliding window is therefore just a
//! restriction of the sources to the last `window` samples.
//!
//! Engines are chosen dynamically:
//!
//! * [`NaturalStream`]: backward record-scan from the new point. For long
//!   histories it maintains a logarithmic set of hull trees (binary counter of
//!   power-of-two blocks, amortised O(log² n) upkeep per push) and hands long
//!   sparse scans to hull queries, with a cost-model hand-back to scanning.
//! * [`HorizontalStream`]: monotone stack, O(1) amortised per push.
//! * [`VectorStream`]: per-anchor record state (the projection axis depends on
//!   the earlier point), O(active · d) per push; horizontal anchors are retired
//!   as soon as they can never see again.

use std::collections::VecDeque;

use super::fast::{InputError, ADAPT_CHUNK, ADAPT_DENSITY};
use super::hull::{tolerance, HullTree};

/// Engine policy for [`NaturalStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamMode {
    /// Scan for short windows / histories, index long histories (default).
    #[default]
    Auto,
    /// Always scan backwards.
    Scan,
    /// Always maintain and use the hull index.
    Indexed,
}

/// History length above which `Auto` starts maintaining the hull index.
pub const STREAM_INDEX_THRESHOLD: usize = 4096;

struct Block {
    start: usize,
    tree: HullTree<'static>,
}

fn check_time(x: &[f64], t: Option<f64>) -> Result<f64, InputError> {
    let n = x.len();
    let t = t.unwrap_or(n as f64);
    if !t.is_finite() || x.last().is_some_and(|&last| t <= last) {
        return Err(InputError::NotIncreasing(n));
    }
    Ok(t)
}

/// Online natural visibility graph.
pub struct NaturalStream {
    y: Vec<f64>,
    x: Vec<f64>,
    window: Option<usize>,
    mode: StreamMode,
    blocks: Vec<Block>,
    indexed_upto: usize,
}

impl NaturalStream {
    /// New stream; `window` limits sources to the last `window` samples.
    pub fn new(window: Option<usize>, mode: StreamMode) -> Self {
        Self {
            y: Vec::new(),
            x: Vec::new(),
            window,
            mode,
            blocks: Vec::new(),
            indexed_upto: 0,
        }
    }

    /// Number of samples pushed.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// True if nothing was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    fn start(&self) -> usize {
        self.window.map_or(0, |w| self.y.len().saturating_sub(w))
    }

    fn wants_index(&self) -> bool {
        let span = self.y.len() - self.start();
        match self.mode {
            StreamMode::Scan => false,
            StreamMode::Indexed => true,
            StreamMode::Auto => span > STREAM_INDEX_THRESHOLD,
        }
    }

    /// Binary-counter maintenance: index every sample in power-of-two blocks.
    fn update_index(&mut self) {
        while self.indexed_upto < self.y.len() {
            let i = self.indexed_upto;
            self.blocks.push(Block {
                start: i,
                tree: HullTree::owned(vec![self.y[i]], vec![self.x[i]]),
            });
            self.indexed_upto += 1;
            while self.blocks.len() >= 2 {
                let k = self.blocks.len();
                let (a, b) = (&self.blocks[k - 2], &self.blocks[k - 1]);
                if a.tree.len() != b.tree.len() {
                    break;
                }
                let (s, e) = (a.start, b.start + b.tree.len());
                self.blocks.truncate(k - 2);
                self.blocks.push(Block {
                    start: s,
                    tree: HullTree::owned(self.y[s..e].to_vec(), self.x[s..e].to_vec()),
                });
            }
        }
        // Drop blocks that fell entirely out of the window.
        let start = self.start();
        self.blocks.retain(|b| b.start + b.tree.len() > start);
    }

    /// Rightmost `k` in `[lo, r)` whose slope from `i` beats `bdy/bdx` (hull queries).
    #[allow(clippy::too_many_arguments)]
    fn query_left(
        &self,
        i: usize,
        lo: usize,
        r: usize,
        bdy: f64,
        bdx: f64,
        yi: f64,
        xi: f64,
        work: &mut usize,
    ) -> Option<usize> {
        let (y, x) = (&self.y, &self.x);
        let (a, b) = (bdx, bdy);
        let c = a * yi + b * xi;
        let tol = tolerance(a, b, yi, xi);
        for blk in self.blocks.iter().rev() {
            let (bs, be) = (blk.start, blk.start + blk.tree.len());
            if bs >= r {
                continue;
            }
            if be <= lo {
                break;
            }
            let (l, h) = (lo.max(bs) - bs, r.min(be) - bs);
            let pred = |k: usize| (y[bs + k] - yi) * bdx > bdy * (xi - x[bs + k]);
            if let Some(k) = blk.tree.rightmost_where(l, h, a, b, c, tol, &pred, work) {
                return Some(bs + k);
            }
        }
        let _ = i;
        None
    }

    /// Pushes a sample (position `t`, default = its index) and returns the
    /// sources of the new edges `(source, new)`, nearest first.
    pub fn push(&mut self, value: f64, t: Option<f64>) -> Result<Vec<usize>, InputError> {
        if !value.is_finite() {
            return Err(InputError::NonFinite(self.y.len()));
        }
        let xi = check_time(&self.x, t)?;
        let i = self.y.len();
        let lo = self.start();
        let indexed = self.wants_index();
        if indexed {
            self.update_index();
        }
        let yi = value;
        let mut out = Vec::new();
        if i > lo {
            let (y, x) = (&self.y, &self.x);
            let mut j = i - 1;
            out.push(j);
            let (mut bdy, mut bdx) = (y[j] - yi, xi - x[j]);
            let (mut k, mut steps, mut seen) = (j, 0usize, 0usize);
            while k > lo {
                if indexed && steps == ADAPT_CHUNK {
                    if seen * ADAPT_DENSITY < steps && k - lo > ADAPT_CHUNK {
                        // Hull queries until they stop paying off.
                        let (mut spent, mut saved) = (0usize, 0usize);
                        loop {
                            let mut work = 0;
                            match self.query_left(i, lo, j, bdy, bdx, yi, xi, &mut work) {
                                Some(q) => {
                                    out.push(q);
                                    spent += work;
                                    saved += j - q;
                                    j = q;
                                    bdy = self.y[q] - yi;
                                    bdx = xi - self.x[q];
                                    if spent > 2 * saved + 64 {
                                        break;
                                    }
                                }
                                None => {
                                    j = lo;
                                    break;
                                }
                            }
                        }
                        k = j;
                        if k <= lo {
                            break;
                        }
                    }
                    steps = 0;
                    seen = 0;
                }
                k -= 1;
                steps += 1;
                let dy = self.y[k] - yi;
                let dx = xi - self.x[k];
                if dy * bdx > bdy * dx {
                    out.push(k);
                    j = k;
                    bdy = dy;
                    bdx = dx;
                    seen += 1;
                }
            }
        }
        self.y.push(yi);
        self.x.push(xi);
        Ok(out)
    }
}

/// Online horizontal visibility graph (monotone stack).
pub struct HorizontalStream {
    y: Vec<f64>,
    stack: VecDeque<usize>,
    window: Option<usize>,
}

impl HorizontalStream {
    /// New stream; `window` limits sources to the last `window` samples.
    pub fn new(window: Option<usize>) -> Self {
        Self {
            y: Vec::new(),
            stack: VecDeque::new(),
            window,
        }
    }

    /// Number of samples pushed.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// True if nothing was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Pushes a sample; returns the sources of the new edges, nearest first.
    pub fn push(&mut self, value: f64) -> Result<Vec<usize>, InputError> {
        if !value.is_finite() {
            return Err(InputError::NonFinite(self.y.len()));
        }
        let i = self.y.len();
        if let Some(w) = self.window {
            let start = i.saturating_sub(w);
            while self.stack.front().is_some_and(|&f| f < start) {
                self.stack.pop_front();
            }
        }
        let mut out = Vec::new();
        while let Some(&t) = self.stack.back() {
            out.push(t);
            if self.y[t] < value {
                self.stack.pop_back();
            } else {
                if self.y[t] == value {
                    self.stack.pop_back();
                }
                break;
            }
        }
        self.stack.push_back(i);
        self.y.push(value);
        Ok(out)
    }
}

struct Anchor {
    index: usize,
    /// q_aa = ‖x_a‖²
    qaa: f64,
    /// natural: best (dy, dt) record; horizontal: running max of projections
    bdy: f64,
    bdt: f64,
}

/// Online vector visibility graph (natural or horizontal).
pub struct VectorStream {
    d: usize,
    data: Vec<f64>,
    x: Vec<f64>,
    anchors: VecDeque<Anchor>,
    window: Option<usize>,
    horizontal: bool,
}

impl VectorStream {
    /// New stream of `d`-dimensional samples.
    pub fn new(d: usize, window: Option<usize>, horizontal: bool) -> Result<Self, InputError> {
        if d == 0 {
            return Err(InputError::LengthMismatch);
        }
        Ok(Self {
            d,
            data: Vec::new(),
            x: Vec::new(),
            anchors: VecDeque::new(),
            window,
            horizontal,
        })
    }

    /// Number of samples pushed.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True if nothing was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Pushes a `d`-dimensional sample; returns the sources of new edges (ascending).
    pub fn push(&mut self, v: &[f64], t: Option<f64>) -> Result<Vec<usize>, InputError> {
        let i = self.x.len();
        if v.len() != self.d {
            return Err(InputError::LengthMismatch);
        }
        if v.iter().any(|z| !z.is_finite()) {
            return Err(InputError::NonFinite(i));
        }
        let ti = check_time(&self.x, t)?;
        if let Some(w) = self.window {
            let start = i.saturating_sub(w);
            while self.anchors.front().is_some_and(|a| a.index < start) {
                self.anchors.pop_front();
            }
        }
        let d = self.d;
        let mut out = Vec::new();
        let (data, x, horizontal) = (&self.data, &self.x, self.horizontal);
        self.anchors.retain_mut(|a| {
            let xa = &data[a.index * d..(a.index + 1) * d];
            let q: f64 = xa.iter().zip(v).map(|(p, r)| p * r).sum();
            if horizontal {
                // bdy holds the running max of projections strictly between a and now.
                if a.index + 1 == i || a.bdy < a.qaa.min(q) {
                    out.push(a.index);
                }
                a.bdy = a.bdy.max(q);
                a.bdy < a.qaa // retire once something at least as high as x_a was seen
            } else {
                let (dy, dt) = (q - a.qaa, ti - x[a.index]);
                if a.index + 1 == i || dy * a.bdt > a.bdy * dt {
                    out.push(a.index);
                    a.bdy = dy;
                    a.bdt = dt;
                }
                true
            }
        });
        let qaa: f64 = v.iter().map(|z| z * z).sum();
        self.anchors.push_back(Anchor {
            index: i,
            qaa,
            bdy: f64::NEG_INFINITY,
            bdt: 1.0,
        });
        self.data.extend_from_slice(v);
        self.x.push(ti);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::fast::{self, Explicit, Uniform};
    use super::super::vector::{self, Rows};
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn batch_edges(e: Vec<(u32, u32)>) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = e
            .into_iter()
            .map(|(a, b)| (a as usize, b as usize))
            .collect();
        v.sort_unstable();
        v
    }

    fn stream_edges(mut push: impl FnMut(usize) -> Vec<usize>, n: usize) -> Vec<(usize, usize)> {
        let mut v = vec![];
        for i in 0..n {
            for s in push(i) {
                v.push((s, i));
            }
        }
        v.sort_unstable();
        v
    }

    #[test]
    fn natural_stream_equals_batch_all_modes() {
        let mut r = Lcg(2);
        for (n, kind) in [(300, 0), (6000, 1), (6000, 2), (3000, 3)] {
            let mut acc = 0.0;
            let y: Vec<f64> = (0..n)
                .map(|i| match kind {
                    0 => (r.next() % 5) as f64,
                    1 => {
                        acc += (r.next() % 201) as f64 - 100.0;
                        acc
                    }
                    2 => (i as f64).sqrt(),
                    _ => (r.next() % 1000) as f64,
                })
                .collect();
            let want = batch_edges(fast::natural_edges_seq(&y, &Uniform));
            for mode in [StreamMode::Scan, StreamMode::Indexed, StreamMode::Auto] {
                let mut s = NaturalStream::new(None, mode);
                assert_eq!(
                    stream_edges(|i| s.push(y[i], None).unwrap(), n),
                    want,
                    "mode {mode:?} kind {kind}"
                );
            }
            // sliding window == batch edges restricted to span < window
            let w = 50;
            let mut s = NaturalStream::new(Some(w), StreamMode::Auto);
            let got = stream_edges(|i| s.push(y[i], None).unwrap(), n);
            let want_w: Vec<_> = want.iter().copied().filter(|&(a, b)| b - a <= w).collect();
            assert_eq!(got, want_w);
        }
    }

    #[test]
    fn natural_stream_irregular_time() {
        let mut r = Lcg(5);
        let n = 500;
        let y: Vec<f64> = (0..n).map(|_| (r.next() % 100) as f64).collect();
        let mut t = 0.0;
        let x: Vec<f64> = (0..n)
            .map(|_| {
                t += 1.0 + (r.next() % 10) as f64 / 3.0;
                t
            })
            .collect();
        let want = batch_edges(fast::natural_edges_seq(&y, &Explicit(&x)));
        let mut s = NaturalStream::new(None, StreamMode::Indexed);
        assert_eq!(stream_edges(|i| s.push(y[i], Some(x[i])).unwrap(), n), want);
    }

    #[test]
    fn horizontal_stream_equals_batch() {
        let mut r = Lcg(3);
        let y: Vec<f64> = (0..2000).map(|_| (r.next() % 7) as f64).collect();
        let want = batch_edges(fast::horizontal_edges(&y));
        let mut s = HorizontalStream::new(None);
        assert_eq!(stream_edges(|i| s.push(y[i]).unwrap(), y.len()), want);
        let mut s = HorizontalStream::new(Some(10));
        let want_w: Vec<_> = want.iter().copied().filter(|&(a, b)| b - a <= 10).collect();
        assert_eq!(stream_edges(|i| s.push(y[i]).unwrap(), y.len()), want_w);
    }

    #[test]
    fn vector_stream_equals_batch() {
        let mut r = Lcg(4);
        let (n, d) = (400, 5);
        let data: Vec<f64> = (0..n * d).map(|_| (r.next() % 9) as f64 - 4.0).collect();
        let rows = Rows::new(&data, d).unwrap();
        for horizontal in [false, true] {
            let want = batch_edges(if horizontal {
                vector::horizontal_vector_edges(&rows, false)
            } else {
                vector::natural_vector_edges(&rows, &Uniform, false)
            });
            let mut s = VectorStream::new(d, None, horizontal).unwrap();
            assert_eq!(
                stream_edges(|i| s.push(&data[i * d..(i + 1) * d], None).unwrap(), n),
                want,
                "h={horizontal}"
            );
            let mut s = VectorStream::new(d, Some(12), horizontal).unwrap();
            let want_w: Vec<_> = want.iter().copied().filter(|&(a, b)| b - a <= 12).collect();
            assert_eq!(
                stream_edges(|i| s.push(&data[i * d..(i + 1) * d], None).unwrap(), n),
                want_w,
                "hw={horizontal}"
            );
        }
    }

    #[test]
    fn stream_validation() {
        let mut s = NaturalStream::new(None, StreamMode::Auto);
        assert!(s.push(f64::NAN, None).is_err());
        s.push(1.0, Some(5.0)).unwrap();
        assert!(s.push(2.0, Some(5.0)).is_err());
        let mut v = VectorStream::new(2, None, false).unwrap();
        assert!(v.push(&[1.0], None).is_err());
    }
}
