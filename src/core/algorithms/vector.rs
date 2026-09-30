//! Vector visibility graphs (VVG) for multivariate time series.
//!
//! Ren & Jin (2019), "Vector visibility graph from multivariate time series: a
//! new method for characterizing nonlinear dynamic behavior in two-phase flow",
//! Nonlinear Dynamics 97. Semantics follow the `vector-vis-graph` Python package:
//! for a pair `a < b`, every vector is projected on the direction of the earlier
//! vector,
//!
//! ```text
//! p_k = x_k · x_a / ‖x_a‖
//! ```
//!
//! and the univariate natural (horizontal) visibility criterion is applied to
//! `p` between `a` and `b`. Visibility is strict (points on the line block).
//!
//! # Algorithm
//!
//! For each `a`, scan `b = a+1, a+2, …` keeping the running maximum slope of
//! `p` seen from `a` (natural) or the running maximum of `p` (horizontal);
//! `b` is visible iff it beats it. Projections are computed lazily during the
//! scan. The scan stops as soon as no later point can become visible: by
//! Cauchy–Schwarz `p_k ≤ ‖x_k‖ ≤ M_b` where `M_b` is the suffix maximum of the
//! norms, which bounds every remaining slope. The bound is applied with a
//! safety margin, so pruning never changes the result. Worst case
//! O(n² · d), typically far less; rows are independent and run in parallel.
//!
//! Internally the scaled projection `q_k = x_k · x_a = ‖x_a‖ p_k` is used:
//! visibility is invariant to positive scaling, and this avoids a division
//! (exact on integer data).
//!
//! Zero vectors have an undefined direction; their projections are taken as
//! 0 (a flat series, so such a point only sees its successor). The Python
//! package produces NaN there and connects the point to everything.

use super::fast::{Edge, InputError, Positions};

/// Row-major `(n, d)` matrix view.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    data: &'a [f64],
    d: usize,
}

impl<'a> Rows<'a> {
    /// Wraps a row-major buffer of `n * d` values.
    pub fn new(data: &'a [f64], d: usize) -> Result<Self, InputError> {
        #[allow(clippy::manual_is_multiple_of)] // keep MSRV below 1.87
        if d == 0 || data.len() % d != 0 {
            return Err(InputError::LengthMismatch);
        }
        if data.len() / d > u32::MAX as usize {
            return Err(InputError::TooLong);
        }
        if let Some(i) = data.iter().position(|v| !v.is_finite()) {
            return Err(InputError::NonFinite(i / d));
        }
        Ok(Self { data, d })
    }

    /// Number of time steps.
    pub fn len(&self) -> usize {
        self.data.len() / self.d
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &'a [f64] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Relative safety margin for the Cauchy–Schwarz pruning bound.
const PRUNE_EPS: f64 = 1e-9;

struct Prepared {
    norms: Vec<f64>,
    /// suffix_max[k] = max(norms[k..])
    suffix_max: Vec<f64>,
}

fn prepare(x: &Rows<'_>) -> Prepared {
    let n = x.len();
    let norms: Vec<f64> = (0..n).map(|i| dot(x.row(i), x.row(i)).sqrt()).collect();
    let mut suffix_max = norms.clone();
    for k in (0..n.saturating_sub(1)).rev() {
        suffix_max[k] = suffix_max[k].max(suffix_max[k + 1]);
    }
    Prepared { norms, suffix_max }
}

/// Natural-VVG edges from point `a` to later points.
#[inline]
fn natural_from<P: Positions>(
    x: &Rows<'_>,
    pos: &P,
    prep: &Prepared,
    a: usize,
    out: &mut Vec<Edge>,
) {
    let n = x.len();
    if a + 1 >= n {
        return;
    }
    // Visibility is invariant to positive scaling of the projected series, so
    // use q_k = x_k · x_a (= ‖x_a‖ p_k): no division, exact for integer data.
    let na = prep.norms[a];
    let xa = x.row(a);
    let proj = |k: usize| dot(x.row(k), xa);
    let pa = dot(xa, xa);
    let ta = pos.at(a);
    let t_last = pos.at(n - 1);

    let mut b = a + 1;
    out.push((a as u32, b as u32));
    let (mut bdy, mut bdt) = (proj(b) - pa, pos.at(b) - ta);

    while b + 1 < n {
        b += 1;
        let dt = pos.at(b) - ta;
        // Prune: q_k <= ‖x_k‖‖x_a‖ <= M_b ‖x_a‖, so every k >= b has
        // slope <= (M_b ‖x_a‖ - q_a) / dt_k.
        let m = (prep.suffix_max[b] * na) * (1.0 + PRUNE_EPS) + PRUNE_EPS;
        let num = m - pa;
        let (bound_num, bound_dt) = if num > 0.0 {
            (num, dt)
        } else {
            (num, t_last - ta)
        };
        // stop if bound <= best, i.e. bound_num * bdt <= bdy * bound_dt
        if bound_num * bdt <= bdy * bound_dt {
            break;
        }
        let dy = proj(b) - pa;
        if dy * bdt > bdy * dt {
            out.push((a as u32, b as u32));
            bdy = dy;
            bdt = dt;
        }
    }
}

/// Horizontal-VVG edges from point `a` to later points.
#[inline]
fn horizontal_from(x: &Rows<'_>, prep: &Prepared, a: usize, out: &mut Vec<Edge>) {
    let n = x.len();
    if a + 1 >= n {
        return;
    }
    let na = prep.norms[a];
    let xa = x.row(a);
    let proj = |k: usize| dot(x.row(k), xa); // scaled projection, see natural_from
    let pa = dot(xa, xa);

    out.push((a as u32, a as u32 + 1));
    let mut running_max = proj(a + 1);
    let mut b = a + 1;
    while b + 1 < n && running_max < pa {
        b += 1;
        // Prune: no later p_k can exceed running_max.
        if (prep.suffix_max[b] * na) * (1.0 + PRUNE_EPS) + PRUNE_EPS <= running_max {
            break;
        }
        let pb = proj(b);
        if running_max < pb {
            out.push((a as u32, b as u32));
            running_max = pb;
        }
    }
}

fn run<F>(n: usize, parallel: bool, f: F) -> Vec<Edge>
where
    F: Fn(usize, &mut Vec<Edge>) + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if parallel && n >= 512 {
            use rayon::prelude::*;
            let chunk = (n / (rayon::current_num_threads() * 16)).max(32);
            let parts: Vec<Vec<Edge>> = (0..n)
                .into_par_iter()
                .step_by(chunk)
                .map(|s| {
                    let mut out = Vec::new();
                    for a in s..(s + chunk).min(n) {
                        f(a, &mut out);
                    }
                    out
                })
                .collect();
            return parts.concat();
        }
    }
    let _ = parallel;
    let mut out = Vec::with_capacity(n * 4);
    for a in 0..n {
        f(a, &mut out);
    }
    out
}

/// Natural vector visibility edges `(a, b)`, `a < b`.
pub fn natural_vector_edges<P: Positions>(x: &Rows<'_>, pos: &P, parallel: bool) -> Vec<Edge> {
    let prep = prepare(x);
    run(x.len(), parallel, |a, out| {
        natural_from(x, pos, &prep, a, out)
    })
}

/// Horizontal vector visibility edges `(a, b)`, `a < b`.
pub fn horizontal_vector_edges(x: &Rows<'_>, parallel: bool) -> Vec<Edge> {
    let prep = prepare(x);
    run(x.len(), parallel, |a, out| {
        horizontal_from(x, &prep, a, out)
    })
}

/// Edge weight methods (same names and formulas as `vector-vis-graph`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorWeight {
    /// `x_a · x_b / (‖x_a‖ ‖x_b‖)`
    CosineSimilarity,
    /// cosine similarity divided by `|t_b - t_a|`
    TimeDiffCosineSimilarity,
    /// `‖x_b - x_a‖`
    EuclideanDistance,
    /// Euclidean distance divided by `|t_b - t_a|`
    TimeDiffEuclideanDistance,
    /// `‖x_b - x_a‖ / (‖x_a‖ + ‖x_b‖)`
    NormalizedEuclideanDistance,
    /// normalized Euclidean distance divided by `|t_b - t_a|`
    TimeDiffNormalizedEuclideanDistance,
}

impl std::str::FromStr for VectorWeight {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "cosine_similarity" => Self::CosineSimilarity,
            "time_diff_cosine_similarity" => Self::TimeDiffCosineSimilarity,
            "euclidean_distance" => Self::EuclideanDistance,
            "time_diff_euclidean_distance" => Self::TimeDiffEuclideanDistance,
            "normalized_euclidean_distance" => Self::NormalizedEuclideanDistance,
            "time_diff_normalized_euclidean_distance" => Self::TimeDiffNormalizedEuclideanDistance,
            other => return Err(format!("unknown weight method '{other}'")),
        })
    }
}

/// Weights for `edges` (NaN where undefined, e.g. cosine with a zero vector).
pub fn vector_weights<P: Positions>(
    x: &Rows<'_>,
    pos: &P,
    edges: &[Edge],
    method: VectorWeight,
) -> Vec<f64> {
    use VectorWeight::*;
    edges
        .iter()
        .map(|&(a, b)| {
            let (a, b) = (a as usize, b as usize);
            let (xa, xb) = (x.row(a), x.row(b));
            let dt = (pos.at(b) - pos.at(a)).abs();
            let (na, nb) = (dot(xa, xa).sqrt(), dot(xb, xb).sqrt());
            let dist = || {
                xa.iter()
                    .zip(xb)
                    .map(|(p, q)| (q - p) * (q - p))
                    .sum::<f64>()
                    .sqrt()
            };
            let cos = || dot(xa, xb) / (na * nb);
            match method {
                CosineSimilarity => cos(),
                TimeDiffCosineSimilarity => cos() / dt,
                EuclideanDistance => dist(),
                TimeDiffEuclideanDistance => dist() / dt,
                NormalizedEuclideanDistance => dist() / (na + nb),
                TimeDiffNormalizedEuclideanDistance => dist() / (na + nb) / dt,
            }
        })
        .collect()
}

/// Many independent multivariate windows, parallel over windows.
#[cfg(feature = "parallel")]
pub fn natural_vector_batch(windows: &[Rows<'_>]) -> Vec<Vec<Edge>> {
    use rayon::prelude::*;
    windows
        .par_iter()
        .map(|w| natural_vector_edges(w, &super::fast::Uniform, false))
        .collect()
}

/// Many independent multivariate windows (horizontal), parallel over windows.
#[cfg(feature = "parallel")]
pub fn horizontal_vector_batch(windows: &[Rows<'_>]) -> Vec<Vec<Edge>> {
    use rayon::prelude::*;
    windows
        .par_iter()
        .map(|w| horizontal_vector_edges(w, false))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::fast::{self, Explicit, Uniform};
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
        fn f(&mut self) -> f64 {
            self.next() as f64 / (1u64 << 31) as f64 - 0.5
        }
    }

    fn sorted(mut e: Vec<Edge>) -> Vec<Edge> {
        e.sort_unstable();
        e
    }

    /// Direct transcription of the definition (no pruning, no running max).
    fn nvvg_ref(x: &Rows<'_>, t: &[f64]) -> Vec<Edge> {
        let n = x.len();
        let mut e = vec![];
        for a in 0..n {
            // scaled projections q_k = x_k · x_a (same visibility as p_k = q_k / ‖x_a‖)
            let p: Vec<f64> = (0..n).map(|k| dot(x.row(k), x.row(a))).collect();
            let pa = p[a];
            for b in a + 1..n {
                // slope(a->k) < slope(a->b) for all k in (a, b), cross-multiplied
                if (a + 1..b).all(|k| (p[k] - pa) * (t[b] - t[a]) < (p[b] - pa) * (t[k] - t[a])) {
                    e.push((a as u32, b as u32));
                }
            }
        }
        e
    }

    fn hvvg_ref(x: &Rows<'_>) -> Vec<Edge> {
        let n = x.len();
        let mut e = vec![];
        for a in 0..n {
            // scaled projections q_k = x_k · x_a (same visibility as p_k = q_k / ‖x_a‖)
            let p: Vec<f64> = (0..n).map(|k| dot(x.row(k), x.row(a))).collect();
            let pa = p[a];
            for b in a + 1..n {
                if (a + 1..b).all(|k| p[k] < pa.min(p[b])) {
                    e.push((a as u32, b as u32));
                }
            }
        }
        e
    }

    #[test]
    fn matches_definition_random() {
        let mut r = Lcg(11);
        for trial in 0..600 {
            let n = 1 + (r.next() % 50) as usize;
            let d = 1 + (r.next() % 6) as usize;
            let data: Vec<f64> = (0..n * d)
                .map(|_| {
                    if trial % 3 == 0 {
                        (r.next() % 4) as f64
                    } else {
                        r.f() + if trial % 3 == 1 { 3.0 } else { 0.0 }
                    }
                })
                .collect();
            let x = Rows::new(&data, d).unwrap();
            let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
            assert_eq!(
                sorted(natural_vector_edges(&x, &Uniform, false)),
                nvvg_ref(&x, &t),
                "trial {trial}"
            );
            assert_eq!(
                sorted(natural_vector_edges(&x, &Uniform, true)),
                nvvg_ref(&x, &t),
                "par {trial}"
            );
            assert_eq!(
                sorted(horizontal_vector_edges(&x, false)),
                hvvg_ref(&x),
                "hvvg {trial}"
            );
            // irregular timeline
            let mut acc = 0.0;
            let tt: Vec<f64> = (0..n)
                .map(|_| {
                    acc += 0.1 + r.f().abs();
                    acc
                })
                .collect();
            assert_eq!(
                sorted(natural_vector_edges(&x, &Explicit(&tt), false)),
                nvvg_ref(&x, &tt),
                "irregular {trial}"
            );
        }
    }

    #[test]
    fn one_dimensional_positive_equals_univariate_nvg() {
        // d = 1 with positive values: p_k = x_k, so VVG == NVG.
        let mut r = Lcg(3);
        for _ in 0..200 {
            let n = 2 + (r.next() % 80) as usize;
            let y: Vec<f64> = (0..n).map(|_| r.f() + 1.0).collect();
            let x = Rows::new(&y, 1).unwrap();
            let mut nvg = fast::natural_edges_seq(&y, &Uniform);
            nvg.sort_unstable();
            assert_eq!(sorted(natural_vector_edges(&x, &Uniform, false)), nvg);
            let mut hvg = fast::horizontal_edges(&y);
            hvg.sort_unstable();
            assert_eq!(sorted(horizontal_vector_edges(&x, false)), hvg);
        }
    }

    #[test]
    fn large_parallel_equals_sequential() {
        let mut r = Lcg(8);
        let (n, d) = (3000, 8);
        let data: Vec<f64> = (0..n * d).map(|_| r.f()).collect();
        let x = Rows::new(&data, d).unwrap();
        assert_eq!(
            sorted(natural_vector_edges(&x, &Uniform, true)),
            sorted(natural_vector_edges(&x, &Uniform, false))
        );
    }

    #[test]
    fn zero_vector_sees_only_successor() {
        let data = [0.0, 0.0, 1.0, 1.0, 2.0, 0.0];
        let x = Rows::new(&data, 2).unwrap();
        let e = sorted(natural_vector_edges(&x, &Uniform, false));
        assert!(e.contains(&(0, 1)) && !e.contains(&(0, 2)));
    }

    #[test]
    fn validation() {
        assert!(Rows::new(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(Rows::new(&[1.0, f64::NAN], 1).is_err());
        assert!(Rows::new(&[1.0], 0).is_err());
    }
}
