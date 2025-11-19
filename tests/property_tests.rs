//! Property-based tests for visibility graph algorithms.
//!
//! These tests use proptest to verify properties that should hold for all inputs.

use proptest::prelude::*;
use rustygraph::*;

// Property: Natural visibility graph should always be connected for monotonic sequences
proptest! {
    #[test]
    fn monotonic_natural_always_connected(values in prop::collection::vec(0.0f64..100.0, 5..50)) {
        // Create monotonically increasing sequence
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let series = TimeSeries::from_raw(sorted).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        // Property: Graph should be connected
        prop_assert!(graph.is_connected());
    }
}

// Property: Number of edges should never exceed maximum possible
proptest! {
    #[test]
    fn edge_count_bounded(values in prop::collection::vec(-100.0f64..100.0, 2..100)) {
        let series = TimeSeries::from_raw(values).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let n = graph.node_count;
        let max_edges = n * (n - 1) / 2;

        // Property: Edge count should not exceed maximum
        prop_assert!(graph.edges().len() <= max_edges);
    }
}

// Property: Both algorithms should produce valid graphs
proptest! {
    #[test]
    fn both_algorithms_valid(values in prop::collection::vec(-10.0f64..10.0, 5..30)) {
        let series = TimeSeries::from_raw(values).unwrap();

        let natural = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let horizontal = VisibilityGraph::from_series(&series)
            .horizontal_visibility()
            .unwrap();

        // Property: Both should have same node count
        prop_assert_eq!(natural.node_count, horizontal.node_count);

        // Property: Both should have valid edge counts
        let n = natural.node_count;
        let max_edges = n * (n - 1) / 2;
        prop_assert!(natural.edges().len() <= max_edges);
        prop_assert!(horizontal.edges().len() <= max_edges);
    }
}

// Property: Adding a constant doesn't change graph structure
proptest! {
    #[test]
    fn translation_invariance(
        values in prop::collection::vec(-10.0f64..10.0, 5..30),
        offset in -100.0f64..100.0
    ) {
        let series1 = TimeSeries::from_raw(values.clone()).unwrap();
        let values2: Vec<f64> = values.iter().map(|v| v + offset).collect();
        let series2 = TimeSeries::from_raw(values2).unwrap();

        let graph1 = VisibilityGraph::from_series(&series1)
            .natural_visibility()
            .unwrap();

        let graph2 = VisibilityGraph::from_series(&series2)
            .natural_visibility()
            .unwrap();

        // Property: Same number of edges after translation
        prop_assert_eq!(graph1.edges().len(), graph2.edges().len());
    }
}

// Property: Degree of each node should be at least 1 (except for isolated series)
proptest! {
    #[test]
    fn minimum_degree(values in prop::collection::vec(-10.0f64..10.0, 3..50)) {
        let series = TimeSeries::from_raw(values).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let degrees = graph.degree_sequence();

        // Property: Most nodes should have degree >= 1 (connected to neighbors)
        // For natural visibility, at least adjacent nodes see each other
        let zero_degree_count = degrees.iter().filter(|&&d| d == 0).count();
        prop_assert!(zero_degree_count == 0 || graph.node_count < 2);
    }
}

// Property: Missing data handling should preserve length
proptest! {
    #[test]
    fn missing_data_preserves_length(
        values in prop::collection::vec(prop::option::of(-10.0f64..10.0), 5..50)
    ) {
        let timestamps: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        let series = TimeSeries::new(timestamps, values.clone()).unwrap();

        if let Ok(cleaned) = series.handle_missing(MissingDataStrategy::LinearInterpolation) {
            // Property: Length should be preserved
            prop_assert_eq!(cleaned.len(), values.len());
        }
    }
}

// Property: Feature computation should not fail for valid inputs
proptest! {
    #[test]
    fn features_always_computable(values in prop::collection::vec(-10.0f64..10.0, 5..30)) {
        let series = TimeSeries::from_raw(values).unwrap();

        let result = VisibilityGraph::from_series(&series)
            .with_features(
                FeatureSet::new()
                    .add_builtin(BuiltinFeature::DeltaForward)
                    .add_builtin(BuiltinFeature::LocalSlope)
            )
            .natural_visibility();

        // Property: Should always succeed for valid input
        prop_assert!(result.is_ok());
    }
}

// Property: Graph density should be between 0 and 1
proptest! {
    #[test]
    fn density_bounded(values in prop::collection::vec(-10.0f64..10.0, 3..50)) {
        let series = TimeSeries::from_raw(values).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let density = graph.density();

        // Property: Density should be in [0, 1]
        prop_assert!((0.0..=1.0).contains(&density));
    }
}

// Property: Clustering coefficient should be between 0 and 1
proptest! {
    #[test]
    fn clustering_bounded(values in prop::collection::vec(-10.0f64..10.0, 5..30)) {
        let series = TimeSeries::from_raw(values).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let cc = graph.average_clustering_coefficient();

        // Property: Clustering coefficient should be in [0, 1]
        prop_assert!((0.0..=1.0).contains(&cc));
    }
}

// Property: Community modularity should be between -1 and 1
proptest! {
    #[test]
    fn modularity_bounded(values in prop::collection::vec(-10.0f64..10.0, 5..30)) {
        let series = TimeSeries::from_raw(values).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let communities = graph.detect_communities();

        // Property: Modularity should be in [-1, 1]
        prop_assert!(communities.modularity >= -1.0 && communities.modularity <= 1.0);
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    #[test]
    fn same_input_same_output() {
        // Test that the same input always produces the same output
        let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];

        let series = TimeSeries::from_raw(values.clone()).unwrap();
        let graph1 = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let series = TimeSeries::from_raw(values).unwrap();
        let graph2 = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        assert_eq!(graph1.edges().len(), graph2.edges().len());
        assert_eq!(graph1.node_count, graph2.node_count);
    }
}

