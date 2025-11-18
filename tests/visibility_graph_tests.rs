// Integration tests for VisibilityGraph

use rustygraph::{TimeSeries, VisibilityGraph};

#[test]
fn test_graph_builder_basic() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    let _builder = VisibilityGraph::from_series(&series);
    // Just verify builder is created successfully
    // Implementation will be tested once algorithms are implemented
}

#[test]
fn test_empty_series() {
    // from_raw returns Err for empty series, so we expect it to fail
    let result = TimeSeries::<f64>::from_raw(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_single_point() {
    let series = TimeSeries::from_raw(vec![1.0]).unwrap();
    let _builder = VisibilityGraph::from_series(&series);
    // Single point should create builder successfully
}

// Note: Most graph tests require the algorithms to be implemented
// These are placeholder tests that verify the API structure

