// Integration tests for the full pipeline
// These tests verify that all components work together

use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};

#[test]
fn test_basic_pipeline() {
    // Create time series
    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();

    // Verify series creation
    assert_eq!(series.len(), 4);
    assert!(!series.is_empty());

    // Create graph builder
    let _builder = VisibilityGraph::from_series(&series);

    // This will work once algorithms are implemented
    // let graph = builder.natural_visibility().unwrap();
    // assert_eq!(graph.node_count, 4);
}

#[test]
fn test_pipeline_with_features() {
    // Create time series
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();

    // Create feature set
    let features = FeatureSet::new()
        .add_builtin(BuiltinFeature::DeltaForward);

    // Verify feature set creation
    assert_eq!(features.len(), 1);

    // Create graph with features
    let _builder = VisibilityGraph::from_series(&series)
        .with_features(features);

    // This will work once algorithms and features are implemented
    // let graph = builder.natural_visibility().unwrap();
    // assert!(graph.node_features(0).is_some());
}

#[test]
fn test_pipeline_with_missing_data() {
    // Create time series with missing data
    let result = TimeSeries::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![Some(1.0), None, Some(3.0), Some(2.0)]
    );

    assert!(result.is_ok());
    let series = result.unwrap();
    assert_eq!(series.len(), 4);

    // This will work once missing data handling is implemented
    // let cleaned = series.handle_missing(MissingDataStrategy::LinearInterpolation).unwrap();
    // let graph = VisibilityGraph::from_series(&cleaned).natural_visibility().unwrap();
}

#[test]
fn test_weighted_graph_pipeline() {
    use rustygraph::algorithms::{visibility_weighted, VisibilityType};

    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();

    // Create weighted edges
    let edges = visibility_weighted(&series, VisibilityType::Natural, |i, j, _, _| {
        (j - i) as f64
    });

    // Verify edges are computed
    assert!(edges.len() > 0);
}

#[test]
fn test_multiple_features() {
    let features: FeatureSet<f64> = FeatureSet::new()
        .add_builtin(BuiltinFeature::DeltaForward)
        .add_builtin(BuiltinFeature::DeltaBackward)
        .add_builtin(BuiltinFeature::LocalSlope);

    assert_eq!(features.len(), 3);
}

#[test]
fn test_custom_feature_function() {
    let features: FeatureSet<f64> = FeatureSet::new()
        .add_function("custom", |series, idx| {
            series[idx].map(|v: f64| v * 2.0)
        });

    assert_eq!(features.len(), 1);
}

