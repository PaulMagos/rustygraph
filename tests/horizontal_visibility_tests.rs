// Integration tests for horizontal visibility algorithm

use rustygraph::algorithms::horizontal::{compute_edges, compute_edges_weighted};

#[test]
fn test_simple_series() {
    let series = vec![1.0, 2.0, 1.0];
    let edges = compute_edges(&series);
    // Expected: (0,1), (1,2) at minimum
    assert!(edges.len() >= 2);
}

#[test]
fn test_weighted_edges() {
    let series = vec![1.0, 3.0, 2.0];

    // Weight by temporal distance
    let edges = compute_edges_weighted(&series, |i, j, _, _| {
        (j - i) as f64
    });

    assert!(edges.len() >= 2);
    // Check that weights are computed correctly
    for (i, j, weight) in edges {
        assert_eq!(weight, (j - i) as f64);
        println!("Edge ({}, {}) has weight {}", i, j, weight);
    }
}

#[test]
fn test_value_based_weights() {
    let series = vec![1.0, 2.0, 3.0];

    // Weight by sum of values
    let edges = compute_edges_weighted(&series, |_, _, vi, vj| {
        vi + vj
    });

    for (i, j, weight) in &edges {
        let expected = series[*i] + series[*j];
        assert_eq!(*weight, expected);
    }
}

#[test]
fn test_monotonic_increasing() {
    let series = vec![1.0, 2.0, 3.0, 4.0];
    let edges = compute_edges(&series);
    // Should have at least adjacent connections
    assert!(edges.len() >= 3);
}

#[test]
fn test_monotonic_decreasing() {
    let series = vec![4.0, 3.0, 2.0, 1.0];
    let edges = compute_edges(&series);
    // Should have at least adjacent connections
    assert!(edges.len() >= 3);
}

#[test]
fn test_flat_series() {
    let series = vec![2.0, 2.0, 2.0, 2.0];
    let edges = compute_edges(&series);
    // All points at same height should see each other
    assert!(edges.len() >= 3);
}

#[test]
fn test_empty_series() {
    let series: Vec<f64> = vec![];
    let edges = compute_edges(&series);
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_single_point() {
    let series = vec![1.0];
    let edges = compute_edges(&series);
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_two_points() {
    let series = vec![1.0, 2.0];
    let edges = compute_edges(&series);
    // Should have one edge connecting them
    assert!(!edges.is_empty());
}

#[test]
fn test_value_difference_weights() {
    let series: Vec<f64> = vec![1.0, 3.0, 2.0];

    // Weight by absolute value difference
    let edges = compute_edges_weighted(&series, |_, _, vi, vj| {
        (vj - vi).abs()
    });

    for (i, j, weight) in &edges {
        let expected = (series[*j] - series[*i]).abs();
        assert_eq!(*weight, expected);
    }
}

