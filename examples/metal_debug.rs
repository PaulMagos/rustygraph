//! Debug Metal GPU implementation

use rustygraph::*;
use rustygraph::performance::{GpuVisibilityGraph, GpuConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Metal GPU Debug Test\n");

    // Medium test case to find divergence
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let series = TimeSeries::from_raw(data.clone())?;

    println!("Test data: {} points\n", data.len());

    // CPU result
    println!("CPU Computation:");
    let cpu_graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;
    let cpu_edges = cpu_graph.edges();
    println!("  Edges: {}", cpu_edges.len());

    let mut cpu_edge_list: Vec<_> = cpu_edges.iter()
        .map(|((a, b), _)| (*a, *b))
        .collect();
    cpu_edge_list.sort();

    // Only print first 10 edges
    for (i, (a, b)) in cpu_edge_list.iter().take(10).enumerate() {
        println!("  {:2}: ({}, {})", i, a, b);
    }
    if cpu_edge_list.len() > 10 {
        println!("  ... and {} more", cpu_edge_list.len() - 10);
    }

    // GPU result
    println!("\nGPU Computation:");
    let gpu_config = GpuConfig::for_apple_silicon().with_min_nodes(1);
    let gpu_builder = GpuVisibilityGraph::with_config(gpu_config);
    let gpu_graph = gpu_builder.build_natural(&series)?;
    let gpu_edges = gpu_graph.edges();
    println!("  Edges: {}", gpu_edges.len());

    let mut gpu_edge_list: Vec<_> = gpu_edges.iter()
        .map(|((a, b), _)| (*a, *b))
        .collect();
    gpu_edge_list.sort();

    // Only print first 10 edges
    for (i, (a, b)) in gpu_edge_list.iter().take(10).enumerate() {
        println!("  {:2}: ({}, {})", i, a, b);
    }
    if gpu_edge_list.len() > 10 {
        println!("  ... and {} more", gpu_edge_list.len() - 10);
    }

    // Compare
    println!("\nComparison:");
    let cpu_set: std::collections::HashSet<_> = cpu_edge_list.iter().collect();
    let gpu_set: std::collections::HashSet<_> = gpu_edge_list.iter().collect();

    let missing: Vec<_> = cpu_set.difference(&gpu_set).collect();
    let extra: Vec<_> = gpu_set.difference(&cpu_set).collect();

    if missing.is_empty() && extra.is_empty() {
        println!("✅ Perfect match!");
    } else {
        if !missing.is_empty() {
            println!("Missing in GPU: {} edges", missing.len());
            for edge in missing.iter().take(5) {
                println!("  ({}, {})", edge.0, edge.1);
            }
            if missing.len() > 5 {
                println!("  ... and {} more", missing.len() - 5);
            }
        }
        if !extra.is_empty() {
            println!("Extra in GPU: {} edges", extra.len());
            for edge in extra.iter().take(5) {
                println!("  ({}, {})", edge.0, edge.1);
            }
            if extra.len() > 5 {
                println!("  ... and {} more", extra.len() - 5);
            }
        }
    }

    Ok(())
}

