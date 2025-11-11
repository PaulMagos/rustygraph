// Integration tests for natural visibility algorithm

use rustygraph::algorithms::natural_visibility;
use rustygraph::algorithms::visibility_weighted;

#[test]
fn test_simple_series() {
    let series = vec![1.0, 2.0, 1.0];
    let edges = natural_visibility(&series);
    // Expected: (0,1), (1,2), (0,2)
    assert!(edges.len() >= 2);
}

#[test]
fn test_weighted_edges() {
    let series = vec![1.0_f64, 3.0_f64, 2.0_f64];
    
    // Weight by value difference
    let edges = visibility_weighted(&series, natural_visibility, |_, _, vi, vj| {
        (vj - vi).abs()
    });
    
    assert!(edges.len() >= 2);
    // Check that weights are computed
    for (i, j, weight) in edges {
        assert!(weight >= 0.0);
        println!("Edge ({}, {}) has weight {}", i, j, weight);
    }
}

#[test]
fn test_constant_weights() {
    let series = vec![1.0, 2.0, 3.0];
    
    // Constant weight function
    let edges = visibility_weighted(&series, natural_visibility, |_, _, _, _| 1.0);
    
    for (_, _, weight) in edges {
        assert_eq!(weight, 1.0);
    }
}

#[test]
fn test_monotonic_increasing() {
    let series = vec![1.0, 2.0, 3.0, 4.0];
    let edges = natural_visibility(&series);
    // All points should see all other points
    assert!(edges.len() >= 3);
}

#[test]
fn test_monotonic_decreasing() {
    let series = vec![4.0, 3.0, 2.0, 1.0];
    let edges = natural_visibility(&series);
    // All points should see all other points
    assert!(edges.len() >= 3);
}

#[test]
fn test_single_peak() {
    let series = vec![1.0, 3.0, 2.0];
    let edges = natural_visibility(&series);
    // All three points should be connected
    assert!(edges.len() >= 2);
}

#[test]
fn test_empty_series() {
    let series: Vec<f64> = vec![];
    let edges = natural_visibility(&series);
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_single_point() {
    let series = vec![1.0];
    let edges = natural_visibility(&series);
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_two_points() {
    let series = vec![1.0, 2.0];
    let edges = natural_visibility(&series);
    // Should have one edge connecting them
    assert!(!edges.is_empty());
}

