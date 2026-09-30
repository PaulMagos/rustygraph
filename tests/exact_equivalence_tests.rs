//! Public-API regression tests: exact equivalence with brute-force definitions.

use rustygraph::{TimeSeries, VisibilityGraph};
use std::collections::BTreeSet;

type Edges = BTreeSet<(usize, usize)>;

/// Natural visibility, brute force, exact on integer-valued data.
fn nvg_bruteforce(x: &[f64], y: &[f64]) -> Edges {
    let mut e = Edges::new();
    for a in 0..y.len() {
        for b in a + 1..y.len() {
            if (a + 1..b).all(|c| (y[c] - y[b]) * (x[b] - x[a]) < (y[a] - y[b]) * (x[b] - x[c])) {
                e.insert((a, b));
            }
        }
    }
    e
}

fn hvg_bruteforce(y: &[f64]) -> Edges {
    let mut e = Edges::new();
    for a in 0..y.len() {
        for b in a + 1..y.len() {
            if (a + 1..b).all(|c| y[c] < y[a].min(y[b])) {
                e.insert((a, b));
            }
        }
    }
    e
}

fn keys(g: &VisibilityGraph<f64>) -> Edges {
    g.edges().keys().copied().collect()
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
fn natural_and_horizontal_match_bruteforce() {
    let mut r = Lcg(1);
    for trial in 0..1500 {
        let n = 2 + (r.next() % 60) as usize;
        let range = [3u64, 10, 1 << 16][trial % 3];
        let y: Vec<f64> = (0..n).map(|_| (r.next() % range) as f64).collect();
        let idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let s = TimeSeries::from_raw(y.clone()).unwrap();
        let nvg = VisibilityGraph::from_series(&s)
            .natural_visibility()
            .unwrap();
        let hvg = VisibilityGraph::from_series(&s)
            .horizontal_visibility()
            .unwrap();
        assert_eq!(keys(&nvg), nvg_bruteforce(&idx, &y), "NVG {y:?}");
        assert_eq!(keys(&hvg), hvg_bruteforce(&y), "HVG {y:?}");
    }
}

#[test]
fn horizontal_regression_non_monotone_blocking() {
    // Old implementation stopped at the first invisible point and missed (0, 4).
    let s = TimeSeries::from_raw(vec![5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
    let g = VisibilityGraph::from_series(&s)
        .horizontal_visibility()
        .unwrap();
    assert!(g.has_edge(0, 4));
    assert_eq!(keys(&g), hvg_bruteforce(&[5.0, 1.0, 3.0, 2.0, 4.0]));
}

#[test]
fn timestamps_are_used_as_positions() {
    // With uneven spacing the middle point blocks only for some timestamps.
    let y = vec![Some(0.0), Some(1.0), Some(3.0)];
    let even = TimeSeries::new(vec![0.0, 1.0, 2.0], y.clone()).unwrap();
    let skew = TimeSeries::new(vec![0.0, 0.1, 2.0], y).unwrap();
    let ge = VisibilityGraph::from_series(&even)
        .natural_visibility()
        .unwrap();
    let gs = VisibilityGraph::from_series(&skew)
        .natural_visibility()
        .unwrap();
    assert!(ge.has_edge(0, 2)); // 1 < 1.5
    assert!(!gs.has_edge(0, 2)); // 1 > 0.15
    assert_eq!(
        keys(&gs),
        nvg_bruteforce(&[0.0, 0.1, 2.0], &[0.0, 1.0, 3.0])
    );
}

#[test]
fn missing_values_are_absent_not_blocking() {
    // Point 1 is missing: 0 and 2 see each other, node 1 has no edges.
    let s = TimeSeries::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![Some(1.0), None, Some(1.0), Some(0.5)],
    )
    .unwrap();
    for g in [
        VisibilityGraph::from_series(&s)
            .natural_visibility()
            .unwrap(),
        VisibilityGraph::from_series(&s)
            .horizontal_visibility()
            .unwrap(),
    ] {
        assert!(g.has_edge(0, 2));
        assert_eq!(g.degree(1), Some(0));
        assert_eq!(g.node_count, 4);
    }
}

#[test]
fn undirected_adjacency_matrix_is_symmetric() {
    let s = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0, 5.0]).unwrap();
    let g = VisibilityGraph::from_series(&s)
        .natural_visibility()
        .unwrap();
    let m = g.to_adjacency_matrix();
    for i in 0..m.len() {
        for j in 0..m.len() {
            assert_eq!(m[i][j], m[j][i]);
        }
    }
    let deg: Vec<usize> = m
        .iter()
        .map(|r| r.iter().filter(|&&w| w != 0.0).count())
        .collect();
    assert_eq!(deg, g.degree_sequence());
}

#[test]
fn large_series_parallel_equals_sequential() {
    use rustygraph::algorithms::fast::{natural_edges, natural_edges_seq, Uniform};
    let mut r = Lcg(5);
    let mut acc = 0.0;
    let y: Vec<f64> = (0..200_000)
        .map(|_| {
            acc += (r.next() % 2001) as f64 - 1000.0;
            acc
        })
        .collect();
    let mut a = natural_edges_seq(&y, &Uniform);
    let mut b = natural_edges(&y, &Uniform);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}
