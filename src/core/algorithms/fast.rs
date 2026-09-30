//! Fast visibility-graph kernels on plain `f64` slices.
//!
//! # Natural visibility
//!
//! Divide & conquer on the maximum (Lan et al., 2015) without recursion:
//! in the segment where point `i` is the (leftmost) maximum, `i` is the only
//! point that can see across itself, so every edge is discovered exactly once
//! from its taller endpoint. That segment is
//!
//! ```text
//! lo(i) = 1 + previous index with y >= y_i
//! hi(i) = next index with y > y_i  - 1
//! ```
//!
//! Both bounds are computed in O(n) with monotone stacks. Point `i` then scans
//! outwards over its segment keeping the running maximum slope: `j` is visible
//! from `i` iff its slope is strictly larger than every slope in between.
//! Total work is `Σ |segment|` = O(n log n) on average (O(n²) worst case for
//! monotone series, like every known exact algorithm), and all scans are
//! independent, so the parallel version needs no synchronisation.
//!
//! # Horizontal visibility
//!
//! Classic O(n) monotone stack.
//!
//! # Conventions
//!
//! * Edges are `(a, b)` with `a < b` (earlier → later).
//! * Visibility is strict: an intermediate point lying exactly on the
//!   line of sight (natural) or at the lower endpoint height (horizontal) blocks.
//! * Slopes are compared by cross-multiplication (no division).
//! * Non-finite values (NaN/±inf) are rejected by the callers in this module's
//!   public wrappers; kernels assume finite input.

/// Fast non-cryptographic hasher for integer edge keys (FxHash scheme).
///
/// Edge keys are node indices produced by this crate, so HashDoS resistance
/// (the reason `std` defaults to SipHash) is not needed.
#[derive(Default, Clone, Copy)]
pub struct EdgeHasher(u64);

impl std::hash::Hasher for EdgeHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// Edge map `(src, dst) -> weight` used by [`crate::VisibilityGraph`].
pub type EdgeMap =
    std::collections::HashMap<(usize, usize), f64, std::hash::BuildHasherDefault<EdgeHasher>>;

/// Edge as `(earlier, later)` node indices.
pub type Edge = (u32, u32);

/// Series length above which the natural-visibility kernel goes parallel.
pub const PARALLEL_THRESHOLD: usize = 1 << 15;

/// Positions of the samples on the time axis.
pub trait Positions: Sync {
    /// Position of sample `i`.
    fn at(&self, i: usize) -> f64;
}

/// Unit-spaced positions `0, 1, 2, …`.
pub struct Uniform;

impl Positions for Uniform {
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        i as f64
    }
}

/// Explicit, strictly increasing positions.
pub struct Explicit<'a>(pub &'a [f64]);

impl Positions for Explicit<'_> {
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.0[i]
    }
}

/// Segment bounds `(lo, hi)` (inclusive) of every point in the max-Cartesian tree.
pub(crate) fn segments(y: &[f64]) -> (Vec<u32>, Vec<u32>) {
    let n = y.len();
    let mut lo = vec![0u32; n];
    let mut hi = vec![0u32; n];
    let mut stack: Vec<u32> = Vec::with_capacity(64);

    // previous index with y >= y_i
    for i in 0..n {
        while let Some(&t) = stack.last() {
            if y[t as usize] < y[i] {
                stack.pop();
            } else {
                break;
            }
        }
        lo[i] = stack.last().map_or(0, |&t| t + 1);
        stack.push(i as u32);
    }

    // next index with y > y_i
    stack.clear();
    for i in (0..n).rev() {
        while let Some(&t) = stack.last() {
            if y[t as usize] <= y[i] {
                stack.pop();
            } else {
                break;
            }
        }
        hi[i] = stack.last().map_or(n as u32 - 1, |&t| t - 1);
        stack.push(i as u32);
    }
    (lo, hi)
}

/// Emits all natural-visibility edges incident to `i` inside its segment.
#[inline(always)]
fn scan_point<P: Positions>(
    y: &[f64],
    pos: &P,
    i: usize,
    lo: usize,
    hi: usize,
    out: &mut Vec<Edge>,
) {
    let yi = y[i];
    let xi = pos.at(i);

    // Left: all y_j < y_i, slopes (y_j - y_i)/(x_i - x_j) < 0.
    if i > lo {
        let mut j = i - 1;
        out.push((j as u32, i as u32));
        let (mut bdy, mut bdx) = (y[j] - yi, xi - pos.at(j));
        while j > lo {
            j -= 1;
            let dy = y[j] - yi;
            let dx = xi - pos.at(j);
            // slope_j > slope_best  <=>  dy * bdx > bdy * dx   (dx, bdx > 0)
            if dy * bdx > bdy * dx {
                out.push((j as u32, i as u32));
                bdy = dy;
                bdx = dx;
            }
        }
    }

    // Right: all y_j <= y_i, slopes <= 0; a zero slope ends the scan.
    if i < hi {
        let mut j = i + 1;
        out.push((i as u32, j as u32));
        let (mut bdy, mut bdx) = (y[j] - yi, pos.at(j) - xi);
        while bdy < 0.0 && j < hi {
            j += 1;
            let dy = y[j] - yi;
            let dx = pos.at(j) - xi;
            if dy * bdx > bdy * dx {
                out.push((i as u32, j as u32));
                bdy = dy;
                bdx = dx;
            }
        }
    }
}

/// Scan chunk after which the adaptive engine re-evaluates the visible density.
pub const ADAPT_CHUNK: usize = 256;
/// Switch to hull queries when fewer than 1 in `ADAPT_DENSITY` scanned points is visible.
pub const ADAPT_DENSITY: usize = 64;

/// Maximum scan→hull switches per side of a point (bounds oscillation).
const MAX_SWITCHES: usize = 4;

/// Like [`scan_point`] but hands long, sparse scans over to hull queries, and
/// takes them back when the hull bound stops pruning (cost-model driven).
fn scan_point_adaptive<P: Positions>(
    y: &[f64],
    pos: &P,
    tree: &super::hull::HullTree<'_>,
    i: usize,
    lo: usize,
    hi: usize,
    out: &mut Vec<Edge>,
) {
    let yi = y[i];
    let xi = pos.at(i);
    if i > lo {
        let mut j = i - 1;
        out.push((j as u32, i as u32));
        let (mut bdy, mut bdx) = (y[j] - yi, xi - pos.at(j));
        let (mut k, mut steps, mut seen, mut switches) = (j, 0usize, 0usize, 0usize);
        while k > lo {
            if steps == ADAPT_CHUNK {
                if switches < MAX_SWITCHES && seen * ADAPT_DENSITY < steps && k - lo > ADAPT_CHUNK {
                    switches += 1;
                    match tree.left_from(i, lo, j, bdy, bdx, true, out) {
                        None => break,
                        Some((nj, ny, nx)) => {
                            j = nj;
                            k = nj;
                            bdy = ny;
                            bdx = nx;
                        }
                    }
                }
                steps = 0;
                seen = 0;
                if k <= lo {
                    break;
                }
            }
            k -= 1;
            steps += 1;
            let dy = y[k] - yi;
            let dx = xi - pos.at(k);
            if dy * bdx > bdy * dx {
                out.push((k as u32, i as u32));
                j = k;
                bdy = dy;
                bdx = dx;
                seen += 1;
            }
        }
    }
    if i < hi {
        let mut j = i + 1;
        out.push((i as u32, j as u32));
        let (mut bdy, mut bdx) = (y[j] - yi, pos.at(j) - xi);
        let (mut k, mut steps, mut seen, mut switches) = (j, 0usize, 0usize, 0usize);
        while bdy < 0.0 && k < hi {
            if steps == ADAPT_CHUNK {
                if switches < MAX_SWITCHES && seen * ADAPT_DENSITY < steps && hi - k > ADAPT_CHUNK {
                    switches += 1;
                    match tree.right_from(i, hi, j, bdy, bdx, true, out) {
                        None => break,
                        Some((nj, ny, nx)) => {
                            j = nj;
                            k = nj;
                            bdy = ny;
                            bdx = nx;
                        }
                    }
                }
                steps = 0;
                seen = 0;
                if bdy >= 0.0 || k >= hi {
                    break;
                }
            }
            k += 1;
            steps += 1;
            let dy = y[k] - yi;
            let dx = pos.at(k) - xi;
            if dy * bdx > bdy * dx {
                out.push((i as u32, k as u32));
                j = k;
                bdy = dy;
                bdx = dx;
                seen += 1;
            }
        }
    }
}

/// Natural visibility edges, sequential.
pub fn natural_edges_seq<P: Positions>(y: &[f64], pos: &P) -> Vec<Edge> {
    let n = y.len();
    let mut out = Vec::with_capacity(n * 3);
    if n < 2 {
        return out;
    }
    let (lo, hi) = segments(y);
    for i in 0..n {
        scan_point(y, pos, i, lo[i] as usize, hi[i] as usize, &mut out);
    }
    out
}

/// Natural visibility edges, parallel over points (deterministic order).
#[cfg(feature = "parallel")]
pub fn natural_edges_par<P: Positions>(y: &[f64], pos: &P) -> Vec<Edge> {
    use rayon::prelude::*;
    let n = y.len();
    if n < 2 {
        return Vec::new();
    }
    let (lo, hi) = segments(y);
    let chunk = (n / (rayon::current_num_threads() * 8)).max(4096);
    let parts: Vec<Vec<Edge>> = (0..n)
        .into_par_iter()
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk).min(n);
            let mut out = Vec::with_capacity((end - start) * 3);
            for i in start..end {
                scan_point(y, pos, i, lo[i] as usize, hi[i] as usize, &mut out);
            }
            out
        })
        .collect();
    let total = parts.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend_from_slice(&p);
    }
    out
}

/// Natural-visibility engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Measure the exact scan cost first and pick the cheaper engine (default).
    #[default]
    Auto,
    /// Segment scan (divide & conquer on the maximum): O(n log n) average, O(n²) worst.
    Scan,
    /// Output-sensitive hull queries: O((n + E) log² n) worst case.
    Hull,
}

impl std::str::FromStr for Algorithm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "scan" | "dc" | "divide_and_conquer" => Ok(Self::Scan),
            "hull" | "output_sensitive" => Ok(Self::Hull),
            other => Err(format!(
                "unknown algorithm '{other}' (expected auto, scan or hull)"
            )),
        }
    }
}

/// `Auto` switches to hull queries when the total scan work exceeds
/// `AUTO_WORK_FACTOR · n · log2 n` (random data sits well below it).
pub const AUTO_WORK_FACTOR: f64 = 30.0;
/// In hull mode, segments up to this length are scanned directly; longer ones
/// use the adaptive scan that may hand over to hull queries.
pub const HULL_MIN_SEGMENT: usize = 256;

/// Plan chosen by [`plan_natural`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Engine that will run.
    pub algorithm: Algorithm,
    /// Exact total scan work `Σ |segment|` of the scan engine.
    pub scan_work: u64,
    /// Whether the parallel path is used.
    pub parallel: bool,
}

fn scan_work(lo: &[u32], hi: &[u32]) -> u64 {
    lo.iter().zip(hi).map(|(&l, &h)| (h - l) as u64).sum()
}

fn choose(n: usize, work: u64, algo: Algorithm) -> Algorithm {
    match algo {
        Algorithm::Auto => {
            let budget = AUTO_WORK_FACTOR * n as f64 * (n.max(2) as f64).log2();
            if n >= 4 * HULL_MIN_SEGMENT && work as f64 > budget {
                Algorithm::Hull
            } else {
                Algorithm::Scan
            }
        }
        a => a,
    }
}

/// Reports which engine [`natural_edges_with`] would use for `y` (O(n)).
pub fn plan_natural(y: &[f64], algo: Algorithm) -> Plan {
    let (lo, hi) = segments(y);
    let work = scan_work(&lo, &hi);
    Plan {
        algorithm: choose(y.len(), work, algo),
        scan_work: work,
        parallel: cfg!(feature = "parallel") && y.len() >= PARALLEL_THRESHOLD,
    }
}

/// Natural visibility edges with explicit engine choice.
pub fn natural_edges_with<P: Positions>(
    y: &[f64],
    pos: &P,
    algo: Algorithm,
    parallel: bool,
) -> Vec<Edge> {
    let n = y.len();
    if n < 2 {
        return Vec::new();
    }
    let (lo, hi) = segments(y);
    let algo = choose(n, scan_work(&lo, &hi), algo);
    // Hull: build the tree once; long sparse scans switch to record queries.
    // Small inputs forced to Hull route every point through the tree (tests).
    let tree = matches!(algo, Algorithm::Hull).then(|| super::hull::HullTree::new(y, pos));
    let force_all = n < 4 * HULL_MIN_SEGMENT;
    let point = |i: usize, out: &mut Vec<Edge>| {
        let (l, h) = (lo[i] as usize, hi[i] as usize);
        match &tree {
            Some(t) if force_all => t.scan_point(i, l, h, out),
            Some(t) if h - l > HULL_MIN_SEGMENT => scan_point_adaptive(y, pos, t, i, l, h, out),
            _ => scan_point(y, pos, i, l, h, out),
        }
    };

    #[cfg(feature = "parallel")]
    {
        if parallel && n >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            let chunk = (n / (rayon::current_num_threads() * 8)).max(4096);
            let parts: Vec<Vec<Edge>> = (0..n)
                .into_par_iter()
                .step_by(chunk)
                .map(|start| {
                    let mut out = Vec::with_capacity(chunk * 3);
                    for i in start..(start + chunk).min(n) {
                        point(i, &mut out);
                    }
                    out
                })
                .collect();
            return parts.concat();
        }
    }
    let _ = parallel;
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        point(i, &mut out);
    }
    out
}

/// Natural visibility edges; picks the engine automatically ([`Algorithm::Auto`])
/// and goes parallel above [`PARALLEL_THRESHOLD`] when available.
pub fn natural_edges<P: Positions>(y: &[f64], pos: &P) -> Vec<Edge> {
    natural_edges_with(y, pos, Algorithm::Auto, true)
}

/// Horizontal visibility edges, O(n).
pub fn horizontal_edges(y: &[f64]) -> Vec<Edge> {
    let n = y.len();
    let mut out = Vec::with_capacity(n * 2);
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    for i in 0..n {
        let yi = y[i];
        while let Some(&t) = stack.last() {
            let yt = y[t as usize];
            out.push((t, i as u32));
            if yt < yi {
                stack.pop();
            } else {
                if yt == yi {
                    stack.pop();
                }
                break;
            }
        }
        stack.push(i as u32);
    }
    out
}

/// Error for invalid kernel input.
#[derive(Debug, Clone, PartialEq)]
pub enum InputError {
    /// A value is NaN or infinite.
    NonFinite(usize),
    /// Positions length differs from values length.
    LengthMismatch,
    /// Positions are not strictly increasing (or not finite).
    NotIncreasing(usize),
    /// Series longer than `u32::MAX` points.
    TooLong,
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputError::NonFinite(i) => write!(f, "non-finite value at index {i}"),
            InputError::LengthMismatch => write!(f, "x and y must have the same length"),
            InputError::NotIncreasing(i) => {
                write!(f, "x must be finite and strictly increasing (index {i})")
            }
            InputError::TooLong => write!(f, "series longer than u32::MAX points"),
        }
    }
}

impl std::error::Error for InputError {}

/// Validates values (and optional positions).
pub fn validate(y: &[f64], x: Option<&[f64]>) -> Result<(), InputError> {
    if y.len() > u32::MAX as usize {
        return Err(InputError::TooLong);
    }
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(InputError::NonFinite(i));
    }
    if let Some(x) = x {
        if x.len() != y.len() {
            return Err(InputError::LengthMismatch);
        }
        if let Some(i) = x.iter().position(|v| !v.is_finite()) {
            return Err(InputError::NotIncreasing(i));
        }
        if let Some(i) = x.windows(2).position(|w| w[1] <= w[0]) {
            return Err(InputError::NotIncreasing(i + 1));
        }
    }
    Ok(())
}

/// Natural visibility with optional explicit positions (validated, automatic engine).
pub fn natural(y: &[f64], x: Option<&[f64]>) -> Result<Vec<Edge>, InputError> {
    natural_with(y, x, Algorithm::Auto)
}

/// Natural visibility with optional positions and explicit engine choice (validated).
pub fn natural_with(
    y: &[f64],
    x: Option<&[f64]>,
    algo: Algorithm,
) -> Result<Vec<Edge>, InputError> {
    validate(y, x)?;
    Ok(match x {
        Some(x) => natural_edges_with(y, &Explicit(x), algo, true),
        None => natural_edges_with(y, &Uniform, algo, true),
    })
}

/// Horizontal visibility (validated).
pub fn horizontal(y: &[f64]) -> Result<Vec<Edge>, InputError> {
    validate(y, None)?;
    Ok(horizontal_edges(y))
}

/// Many independent series (e.g. sliding windows), parallel over series.
#[cfg(feature = "parallel")]
pub fn natural_batch(rows: &[&[f64]]) -> Result<Vec<Vec<Edge>>, InputError> {
    use rayon::prelude::*;
    for r in rows {
        validate(r, None)?;
    }
    Ok(rows
        .par_iter()
        .map(|r| natural_edges_seq(r, &Uniform))
        .collect())
}

/// Many independent series, parallel over series.
#[cfg(feature = "parallel")]
pub fn horizontal_batch(rows: &[&[f64]]) -> Result<Vec<Vec<Edge>>, InputError> {
    use rayon::prelude::*;
    for r in rows {
        validate(r, None)?;
    }
    Ok(rows.par_iter().map(|r| horizontal_edges(r)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut e: Vec<Edge>) -> Vec<Edge> {
        e.sort_unstable();
        e
    }

    /// Exact brute force on integer data (i128, no rounding).
    fn nvg_exact(y: &[i64]) -> Vec<Edge> {
        let mut e = vec![];
        for a in 0..y.len() {
            for b in a + 1..y.len() {
                let ok = (a + 1..b).all(|c| {
                    let (ya, yb, yc) = (y[a] as i128, y[b] as i128, y[c] as i128);
                    (yc - yb) * ((b - a) as i128) < (ya - yb) * ((b - c) as i128)
                });
                if ok {
                    e.push((a as u32, b as u32));
                }
            }
        }
        e
    }

    fn hvg_exact(y: &[i64]) -> Vec<Edge> {
        let mut e = vec![];
        for a in 0..y.len() {
            for b in a + 1..y.len() {
                if (a + 1..b).all(|c| y[c] < y[a].min(y[b])) {
                    e.push((a as u32, b as u32));
                }
            }
        }
        e
    }

    /// Tiny deterministic LCG so the test needs no extra dependency.
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

    fn check_int(y: &[i64]) {
        let yf: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        assert_eq!(
            sorted(natural_edges_seq(&yf, &Uniform)),
            nvg_exact(y),
            "NVG {y:?}"
        );
        assert_eq!(sorted(horizontal_edges(&yf)), hvg_exact(y), "HVG {y:?}");
        #[cfg(feature = "parallel")]
        assert_eq!(
            sorted(natural_edges_par(&yf, &Uniform)),
            nvg_exact(y),
            "NVG par {y:?}"
        );
        assert_eq!(
            sorted(natural_edges_with(&yf, &Uniform, Algorithm::Hull, false)),
            nvg_exact(y),
            "NVG hull {y:?}"
        );
    }

    #[test]
    fn matches_exact_bruteforce_random_ties_and_degenerate() {
        let mut r = Lcg(42);
        for trial in 0..3000 {
            let n = 1 + (r.next() % 70) as usize;
            let range = [2, 4, 10, 1000, 1 << 20][trial % 5];
            let y: Vec<i64> = (0..n)
                .map(|_| (r.next() % range) as i64 - (range as i64) / 2)
                .collect();
            check_int(&y);
        }
        for y in [
            vec![],
            vec![7],
            vec![1, 1],
            vec![3; 40],
            (0..40).collect(),
            (0..40).rev().collect(),
            vec![0, 1, 2, 3, 2, 1, 0],
            vec![1, 3, 3, 3, 1, 2, 3],
            vec![5, 1, 3, 2, 4],
            (0..40).map(|i| i * i).collect(),
            (0..40).map(|i| -(i * i)).collect(),
            (0..40).map(|i| (i % 7) * (i % 3)).collect(),
        ] {
            check_int(&y);
        }
    }

    #[test]
    fn hull_engine_matches_scan_on_large_and_adversarial_inputs() {
        let mut r = Lcg(21);
        let mut acc = 0.0;
        let walk: Vec<f64> = (0..20_000)
            .map(|_| {
                acc += (r.next() % 2001) as f64 - 1000.0;
                acc
            })
            .collect();
        let concave: Vec<f64> = (0..5_000).map(|i| (i as f64).sqrt()).collect();
        let convex: Vec<f64> = (0..3_000).map(|i| (i as f64) * (i as f64)).collect();
        let noise: Vec<f64> = (0..20_000).map(|_| r.next() as f64).collect();
        let mut xs = Vec::new();
        let mut t = 0.0;
        for _ in 0..20_000 {
            t += 0.5 + (r.next() % 100) as f64 / 50.0;
            xs.push(t);
        }
        for y in [&walk, &concave, &convex, &noise] {
            let a = sorted(natural_edges_with(y, &Uniform, Algorithm::Scan, false));
            assert_eq!(
                sorted(natural_edges_with(y, &Uniform, Algorithm::Hull, false)),
                a
            );
            assert_eq!(
                sorted(natural_edges_with(y, &Uniform, Algorithm::Hull, true)),
                a
            );
            assert_eq!(
                sorted(natural_edges_with(y, &Uniform, Algorithm::Auto, true)),
                a
            );
        }
        let a = sorted(natural_edges_with(
            &walk,
            &Explicit(&xs),
            Algorithm::Scan,
            false,
        ));
        assert_eq!(
            sorted(natural_edges_with(
                &walk,
                &Explicit(&xs),
                Algorithm::Hull,
                false
            )),
            a
        );
    }

    #[test]
    fn auto_plan_picks_hull_for_monotone_and_scan_for_noise() {
        let concave: Vec<f64> = (0..50_000).map(|i| (i as f64).sqrt()).collect();
        assert_eq!(
            plan_natural(&concave, Algorithm::Auto).algorithm,
            Algorithm::Hull
        );
        let mut r = Lcg(4);
        let noise: Vec<f64> = (0..50_000).map(|_| r.next() as f64).collect();
        assert_eq!(
            plan_natural(&noise, Algorithm::Auto).algorithm,
            Algorithm::Scan
        );
    }

    #[test]
    fn explicit_unit_positions_equal_uniform() {
        let mut r = Lcg(7);
        let y: Vec<f64> = (0..500).map(|_| r.next() as f64 / 1e9).collect();
        let x: Vec<f64> = (0..500).map(|i| i as f64).collect();
        assert_eq!(
            natural_edges_seq(&y, &Uniform),
            natural_edges_seq(&y, &Explicit(&x))
        );
    }

    #[test]
    fn prefix_edges_are_stable() {
        let mut r = Lcg(9);
        let y: Vec<f64> = (0..300).map(|_| r.next() as f64).collect();
        let full = sorted(natural_edges_seq(&y, &Uniform));
        for k in [2, 17, 150, 299] {
            let pre: Vec<Edge> = full
                .iter()
                .copied()
                .filter(|e| (e.1 as usize) < k)
                .collect();
            assert_eq!(pre, sorted(natural_edges_seq(&y[..k], &Uniform)));
        }
    }

    #[test]
    fn validation() {
        assert_eq!(
            natural(&[1.0, f64::NAN], None),
            Err(InputError::NonFinite(1))
        );
        assert_eq!(
            natural(&[1.0, 2.0], Some(&[0.0])),
            Err(InputError::LengthMismatch)
        );
        assert_eq!(
            natural(&[1.0, 2.0, 3.0], Some(&[0.0, 1.0, 1.0])),
            Err(InputError::NotIncreasing(2))
        );
        assert_eq!(natural(&[], None), Ok(vec![]));
    }
}
