// Integration tests for VisibilityGraph

use rustygraph::{TimeSeries, VisibilityGraph};

#[test]
fn test_graph_builder_basic() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
    let _builder = VisibilityGraph::from_series(&series);
    // Just verify builder is created successfully
    // Implementation will be tested once algorithms are implemented
}

#[test]
fn test_empty_series() {
    let series: TimeSeries<f64> = TimeSeries::from_raw(vec![]);
    let _builder = VisibilityGraph::from_series(&series);
    // Builder should be created even for empty series
    // Error should occur when trying to build the graph
}

#[test]
fn test_single_point() {
    let series = TimeSeries::from_raw(vec![1.0]);
    let _builder = VisibilityGraph::from_series(&series);
    // Single point should create builder successfully
}

// Note: Most graph tests require the algorithms to be implemented
// These are placeholder tests that verify the API structure

