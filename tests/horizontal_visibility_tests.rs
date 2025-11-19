// Integration tests for horizontal visibility algorithm

use rustygraph::algorithms::{horizontal_visibility, visibility_weighted, VisibilityType};
use rustygraph::TimeSeries;

#[test]
fn test_simple_series() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 1.0]).unwrap();
    let edges = horizontal_visibility(&series);
    // Expected: (0,1), (1,2) at minimum
    assert!(edges.len() >= 2);
}

#[test]
fn test_weighted_edges() {
    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();

    // Weight by temporal distance
    let edges = visibility_weighted(&series, VisibilityType::Horizontal, |i, j, _, _| {
        (j - i) as f64
    });

    assert!(edges.len() >= 2);
    // Check that weights are computed correctly
    for ((i, j), weight) in edges {
        assert_eq!(weight, (j - i) as f64);
        println!("Edge ({}, {}) has weight {}", i, j, weight);
    }
}

#[test]
fn test_value_based_weights() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();

    // Weight by sum of values
    let edges = visibility_weighted(&series, VisibilityType::Horizontal, |_, _, vi, vj| {
        vi + vj
    });

    for ((i, j), weight) in &edges {
        let vi: f64 = series.values[*i].unwrap().into();
        let vj: f64 = series.values[*j].unwrap().into();
        let expected = vi + vj;
        assert_eq!(*weight, expected);
    }
}

#[test]
fn test_monotonic_increasing() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let edges = horizontal_visibility(&series);
    // Should have at least adjacent connections
    assert!(edges.len() >= 3);
}

#[test]
fn test_monotonic_decreasing() {
    let series = TimeSeries::from_raw(vec![4.0, 3.0, 2.0, 1.0]).unwrap();
    let edges = horizontal_visibility(&series);
    // Should have at least adjacent connections
    assert!(edges.len() >= 3);
}

#[test]
fn test_flat_series() {
    let series = TimeSeries::from_raw(vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let edges = horizontal_visibility(&series);
    // All points at same height should see each other
    assert!(edges.len() >= 3);
}

#[test]
fn test_empty_series() {
    // Empty series returns error from from_raw
    let result = TimeSeries::<f64>::from_raw(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_single_point() {
    let series = TimeSeries::from_raw(vec![1.0]).unwrap();
    let edges = horizontal_visibility(&series);
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_two_points() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0]).unwrap();
    let edges = horizontal_visibility(&series);
    // Should have one edge connecting them
    assert!(!edges.is_empty());
}

#[test]
fn test_value_difference_weights() {
    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();

    // Weight by absolute value difference
    let edges = visibility_weighted(&series, VisibilityType::Horizontal, |_, _, vi: f64, vj: f64| {
        (vj - vi).abs()
    });

    for ((i, j), weight) in &edges {
        let vi: f64 = series.values[*i].unwrap().into();
        let vj: f64 = series.values[*j].unwrap().into();
        let expected = (vj - vi).abs();
        assert_eq!(*weight, expected);
    }
}

