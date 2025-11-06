//! Basic visibility graph usage example.
//!
//! This example demonstrates the basic functionality of creating
//! visibility graphs from time series data.

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Basic Usage Example ===\n");

    // Create a simple time series
    println!("Creating time series...");
    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
    println!("Time series length: {}", series.len());

    // Build a natural visibility graph
    println!("\nBuilding natural visibility graph...");
    let natural_graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;

    println!("Number of nodes: {}", natural_graph.node_count);
    println!("Number of edges: {}", natural_graph.edges().len());
    println!("Degree sequence: {:?}", natural_graph.degree_sequence());

    // Display edges
    println!("\nEdges:");
    for (src, dst) in natural_graph.edges() {
        println!("  {} -> {}", src, dst);
    }

    // Build a horizontal visibility graph
    println!("\nBuilding horizontal visibility graph...");
    let horizontal_graph = VisibilityGraph::from_series(&series)
        .horizontal_visibility()?;

    println!("Number of nodes: {}", horizontal_graph.node_count);
    println!("Number of edges: {}", horizontal_graph.edges().len());
    println!("Degree sequence: {:?}", horizontal_graph.degree_sequence());

    // Display adjacency information
    println!("\nNode neighbors (horizontal):");
    for i in 0..horizontal_graph.node_count {
        if let Some(neighbors) = horizontal_graph.neighbors(i) {
            println!("  Node {}: {:?}", i, neighbors);
        }
    }

    // Export to adjacency matrix
    println!("\nAdjacency matrix (natural):");
    let matrix = natural_graph.to_adjacency_matrix();
    for (_i, row) in matrix.iter().enumerate() {
        print!("  [");
        for (j, &val) in row.iter().enumerate() {
            print!("{}", if val { 1 } else { 0 });
            if j < row.len() - 1 {
                print!(" ");
            }
        }
        println!("]");
    }

    Ok(())
}

