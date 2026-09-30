//! Sequential visibility-graph motifs (Iacovacci & Lacasa, PRE 93, 042309, 2016).
//!
//! For every run of 4 consecutive samples, the induced subgraph always contains
//! the path (0,1),(1,2),(2,3); the motif is which of the optional edges
//! (0,2), (1,3), (0,3) are present, encoded as
//! `code = [0,2] | [1,3] << 1 | [0,3] << 2` (8 codes, some geometrically
//! impossible for a given graph type). Because visibility between two samples
//! depends only on the samples between them, each code is a function of those 4
//! samples alone: the profile costs O(n) and never builds the graph.
//!
//! The motif profile is a robust, length-independent fingerprint of temporal
//! structure (periodic, chaotic and random series separate cleanly), which makes
//! it a good fidelity descriptor for synthetic time series.

use super::fast::InputError;

/// Number of size-4 sequential motif codes.
pub const MOTIFS: usize = 8;

/// Graph type for motif extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotifKind {
    /// Natural visibility.
    Natural,
    /// Horizontal visibility.
    Horizontal,
}

#[inline(always)]
fn nvis(y: &[f64], a: usize, b: usize) -> bool {
    // uniform spacing; same cross-multiplied criterion as the kernels
    (a + 1..b).all(|c| (y[c] - y[a]) * ((b - a) as f64) < (y[b] - y[a]) * ((c - a) as f64))
}

#[inline(always)]
fn hvis(y: &[f64], a: usize, b: usize) -> bool {
    (a + 1..b).all(|c| y[c] < y[a].min(y[b]))
}

#[inline(always)]
fn code(vis: impl Fn(usize, usize) -> bool) -> usize {
    (vis(0, 2) as usize) | (vis(1, 3) as usize) << 1 | (vis(0, 3) as usize) << 2
}

/// Motif counts of a univariate series.
pub fn motif_counts(y: &[f64], kind: MotifKind) -> Result<[u64; MOTIFS], InputError> {
    super::fast::validate(y, None)?;
    let mut c = [0u64; MOTIFS];
    for w in y.windows(4) {
        let k = match kind {
            MotifKind::Natural => code(|a, b| nvis(w, a, b)),
            MotifKind::Horizontal => code(|a, b| hvis(w, a, b)),
        };
        c[k] += 1;
    }
    Ok(c)
}

/// Motif counts of a vector (multivariate) series, row-major `(n, d)`.
pub fn vector_motif_counts(
    data: &[f64],
    d: usize,
    kind: MotifKind,
) -> Result<[u64; MOTIFS], InputError> {
    let rows = super::vector::Rows::new(data, d)?;
    let n = rows.len();
    let mut c = [0u64; MOTIFS];
    let dot = |i: usize, j: usize| -> f64 {
        data[i * d..(i + 1) * d]
            .iter()
            .zip(&data[j * d..(j + 1) * d])
            .map(|(p, q)| p * q)
            .sum()
    };
    for s in 0..n.saturating_sub(3) {
        // projections on each possible anchor a ∈ {s, s+1}, as in the VVG definition
        let vis = |a: usize, b: usize| -> bool {
            let (a, b) = (s + a, s + b);
            let q: Vec<f64> = (a..=b).map(|k| dot(k, a)).collect();
            let m = b - a;
            match kind {
                MotifKind::Natural => {
                    (1..m).all(|c| (q[c] - q[0]) * (m as f64) < (q[m] - q[0]) * (c as f64))
                }
                MotifKind::Horizontal => (1..m).all(|c| q[c] < q[0].min(q[m])),
            }
        };
        c[code(vis)] += 1;
    }
    Ok(c)
}

/// Motif counts for many univariate series, parallel over series.
#[cfg(feature = "parallel")]
pub fn motif_counts_batch(
    rows: &[&[f64]],
    kind: MotifKind,
) -> Result<Vec<[u64; MOTIFS]>, InputError> {
    use rayon::prelude::*;
    rows.par_iter().map(|r| motif_counts(r, kind)).collect()
}

/// Motif counts for many vector series of dimension `d`, parallel over series.
#[cfg(feature = "parallel")]
pub fn vector_motif_counts_batch(
    rows: &[&[f64]],
    d: usize,
    kind: MotifKind,
) -> Result<Vec<[u64; MOTIFS]>, InputError> {
    use rayon::prelude::*;
    rows.par_iter()
        .map(|r| vector_motif_counts(r, d, kind))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::fast::{horizontal_edges, natural_edges_seq, Uniform};
    use super::super::vector::{horizontal_vector_edges, natural_vector_edges, Rows};
    use super::*;
    use std::collections::HashSet;

    /// Motifs read off the full graph must equal the local computation.
    fn from_graph(edges: &[(u32, u32)], n: usize) -> [u64; MOTIFS] {
        let e: HashSet<(u32, u32)> = edges.iter().copied().collect();
        let mut c = [0u64; MOTIFS];
        for s in 0..n.saturating_sub(3) as u32 {
            let k = (e.contains(&(s, s + 2)) as usize)
                | (e.contains(&(s + 1, s + 3)) as usize) << 1
                | (e.contains(&(s, s + 3)) as usize) << 2;
            c[k] += 1;
        }
        c
    }

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

    #[test]
    fn local_motifs_equal_graph_motifs() {
        let mut r = Lcg(1);
        for trial in 0..300 {
            let n = 4 + (r.next() % 80) as usize;
            let y: Vec<f64> = (0..n)
                .map(|_| (r.next() % if trial % 2 == 0 { 5 } else { 1 << 20 }) as f64)
                .collect();
            assert_eq!(
                motif_counts(&y, MotifKind::Natural).unwrap(),
                from_graph(&natural_edges_seq(&y, &Uniform), n)
            );
            assert_eq!(
                motif_counts(&y, MotifKind::Horizontal).unwrap(),
                from_graph(&horizontal_edges(&y), n)
            );
            let d = 1 + (r.next() % 4) as usize;
            let data: Vec<f64> = (0..n * d).map(|_| (r.next() % 7) as f64 - 3.0).collect();
            let rows = Rows::new(&data, d).unwrap();
            assert_eq!(
                vector_motif_counts(&data, d, MotifKind::Natural).unwrap(),
                from_graph(&natural_vector_edges(&rows, &Uniform, false), n)
            );
            assert_eq!(
                vector_motif_counts(&data, d, MotifKind::Horizontal).unwrap(),
                from_graph(&horizontal_vector_edges(&rows, false), n)
            );
        }
    }

    #[test]
    fn short_series_have_no_motifs() {
        assert_eq!(
            motif_counts(&[1.0, 2.0, 3.0], MotifKind::Natural).unwrap(),
            [0; MOTIFS]
        );
    }
}
