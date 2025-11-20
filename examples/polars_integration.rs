//! Example demonstrating Polars DataFrame integration with RustyGraph.
//!
//! This example shows how to:
//! - Load time series data from Polars DataFrames
//! - Build visibility graphs from DataFrame data
//! - Export graph properties back to DataFrames
//! - Batch process multiple time series from a grouped DataFrame
//!
//! Run with: cargo run --example polars_integration --features polars-integration

use rustygraph::{TimeSeries, VisibilityGraph};

#[cfg(feature = "polars-integration")]
use rustygraph::integrations::polars::*;

#[cfg(feature = "polars-integration")]
use polars::prelude::*;

#[cfg(feature = "polars-integration")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 RustyGraph + Polars Integration Example\n");

    // ========================================================================
    // Example 1: Basic DataFrame to TimeSeries Conversion
    // ========================================================================
    println!("📊 Example 1: Basic DataFrame Conversion");
    println!("{}", "=".repeat(60));

    let df = df! {
        "time" => &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "value" => &[1.0, 3.0, 2.0, 4.0, 1.0, 3.0],
    }?;

    println!("Input DataFrame:");
    println!("{}", df);

    // Convert to TimeSeries
    let series = TimeSeries::<f64>::from_polars_df(&df, "time", "value")?;
    println!("\n✅ Converted to TimeSeries with {} points", series.len());

    // Build visibility graph
    let graph = VisibilityGraph::from_series(&series).natural_visibility()?;

    println!("📈 Natural Visibility Graph:");
    println!("   Nodes: {}", graph.node_count);
    println!("   Edges: {}", graph.edges().len());
    println!("   Density: {:.4}", graph.density());

    // ========================================================================
    // Example 2: Export Graph Properties to DataFrame
    // ========================================================================
    println!("\n📊 Example 2: Export Graph to DataFrame");
    println!("{}", "=".repeat(60));

    let graph_df = graph.to_polars_df()?;
    println!("Graph properties as DataFrame:");
    println!("{}", graph_df);

    let edges_df = graph.edges_to_polars_df()?;
    println!("\nEdges as DataFrame:");
    println!("{}", edges_df);

    // ========================================================================
    // Example 3: Batch Processing Multiple Time Series
    // ========================================================================
    println!("\n📊 Example 3: Batch Processing");
    println!("{}", "=".repeat(60));

    // Create DataFrame with multiple sensors
    let multi_df = df! {
        "timestamp" => &[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        "reading" => &[1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0],
        "sensor" => &["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
    }?;

    println!("Multi-sensor DataFrame:");
    println!("{}", multi_df);

    // Batch process all sensors
    let batch_processor = BatchProcessor::from_polars_df(
        &multi_df,
        "timestamp",
        "reading",
        "sensor",
    )?;

    let results = batch_processor.process_natural()?;

    println!("\n✅ Processed {} time series", results.len());
    println!("   Sensors: {:?}", results.keys());

    // Analyze each sensor's graph
    for key in results.keys() {
        if let Some(graph) = results.get(key) {
            println!("\n🔹 Sensor {}:", key);
            println!("   Nodes: {}", graph.node_count);
            println!("   Edges: {}", graph.edges().len());
            println!("   Avg Degree: {:.2}", graph.degree_sequence().iter().sum::<usize>() as f64 / graph.node_count as f64);
        }
    }

    // Export batch results to DataFrame
    let results_df = results.to_polars_df()?;
    println!("\n📊 Batch results as DataFrame:");
    println!("{}", results_df);

    // ========================================================================
    // Example 4: Roundtrip - DataFrame → TimeSeries → Graph → DataFrame
    // ========================================================================
    println!("\n📊 Example 4: Complete Roundtrip");
    println!("{}", "=".repeat(60));

    // Start with a DataFrame
    let input_df = df! {
        "t" => &[0.0, 1.0, 2.0, 3.0],
        "y" => &[1.0, 4.0, 2.0, 3.0],
    }?;

    println!("1. Input DataFrame:");
    println!("{}", input_df);

    // Convert to TimeSeries
    let ts = TimeSeries::<f64>::from_polars_df(&input_df, "t", "y")?;
    println!("\n2. ✅ Converted to TimeSeries");

    // Build graph
    let g = VisibilityGraph::from_series(&ts).horizontal_visibility()?;
    println!("3. ✅ Built Horizontal Visibility Graph");

    // Export back to DataFrame
    let output_df = g.to_polars_df()?;
    println!("\n4. Output DataFrame with graph properties:");
    println!("{}", output_df);

    // ========================================================================
    // Example 5: Advanced - Lazy Evaluation
    // ========================================================================
    println!("\n📊 Example 5: Lazy Evaluation (Efficient Pipeline)");
    println!("{}", "=".repeat(60));

    // Create a lazy DataFrame
    let lazy_df = df! {
        "time" => &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "value" => &[1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0],
    }?
    .lazy()
    .select([
        col("time"),
        col("value").alias("original_value"),
    ])
    .collect()?;

    println!("Processed DataFrame (lazy evaluation):");
    println!("{}", lazy_df);

    let lazy_series = TimeSeries::<f64>::from_polars_df(&lazy_df, "time", "original_value")?;
    let lazy_graph = VisibilityGraph::from_series(&lazy_series).natural_visibility()?;

    println!("\n✅ Graph from lazy DataFrame:");
    println!("   Nodes: {}", lazy_graph.node_count);
    println!("   Edges: {}", lazy_graph.edges().len());

    println!("\n🎉 All Polars integration examples completed successfully!");
    Ok(())
}

#[cfg(not(feature = "polars-integration"))]
fn main() {
    eprintln!("❌ This example requires the 'polars-integration' feature.");
    eprintln!("Run with: cargo run --example polars_integration --features polars-integration");
    std::process::exit(1);
}

