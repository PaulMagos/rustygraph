//! Advanced example with node features and missing data handling.
//!
//! This example demonstrates:
//! - Handling missing data with imputation strategies
//! - Computing built-in node features
//! - Adding custom feature functions
//! - Inspecting feature values

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Advanced Features Example ===\n");

    // Create time series with missing data
    println!("Creating time series with missing data...");
    let series = TimeSeries::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![Some(1.0), None, Some(3.0), Some(2.0), Some(4.0), Some(1.0)]
    )?;
    println!("Time series length: {}", series.len());

    // Handle missing data with fallback strategy
    println!("\nHandling missing data...");
    let cleaned = series.handle_missing(
        MissingDataStrategy::LinearInterpolation
            .with_fallback(MissingDataStrategy::ForwardFill)
    )?;
    println!("Missing data imputed successfully");

    // Build feature set
    println!("\nConfiguring features...");
    let features = FeatureSet::new()
        .add_builtin(BuiltinFeature::DeltaForward)
        .add_builtin(BuiltinFeature::DeltaBackward)
        .add_builtin(BuiltinFeature::LocalSlope)
        .add_builtin(BuiltinFeature::Acceleration)
        .add_builtin(BuiltinFeature::IsLocalMax)
        .add_builtin(BuiltinFeature::IsLocalMin)
        .add_function("squared", |series, idx| {
            series[idx].map(|v| v * v)
        })
        .add_function("log_abs", |series, idx| {
            series[idx].map(|v| v.abs().ln())
        })
        .with_missing_data_strategy(MissingDataStrategy::LinearInterpolation);

    println!("Configured {} features", features.len());

    // Build graph with features
    println!("\nBuilding visibility graph with features...");
    let graph = VisibilityGraph::from_series(&cleaned)
        .with_features(features)
        .natural_visibility()?;

    println!("Graph constructed:");
    println!("  Nodes: {}", graph.node_count);
    println!("  Edges: {}", graph.edges().len());

    // Display features for each node
    println!("\nNode features:");
    for i in 0..graph.node_count {
        println!("\n  Node {}:", i);
        if let Some(features) = graph.node_features(i) {
            for (name, value) in features {
                println!("    {}: {:.4}", name, value);
            }
        }
    }

    // Analyze degree distribution
    println!("\nDegree distribution:");
    let degrees = graph.degree_sequence();
    let max_degree = degrees.iter().max().unwrap_or(&0);
    let min_degree = degrees.iter().min().unwrap_or(&0);
    let avg_degree: f64 = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;

    println!("  Min degree: {}", min_degree);
    println!("  Max degree: {}", max_degree);
    println!("  Average degree: {:.2}", avg_degree);

    // Alternative: Build horizontal visibility graph
    println!("\n--- Horizontal Visibility Graph ---");
    let h_graph = VisibilityGraph::from_series(&cleaned)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::LocalMean)
        )
        .horizontal_visibility()?;

    println!("Horizontal graph:");
    println!("  Nodes: {}", h_graph.node_count);
    println!("  Edges: {}", h_graph.edges().len());

    Ok(())
}

