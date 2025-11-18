//! Weighted visibility graph example.
//!
//! This example demonstrates how to create weighted visibility graphs
//! with custom edge weight functions.

use rustygraph::algorithms::{visibility_weighted, VisibilityType};
use rustygraph::TimeSeries;

fn main() {
    println!("=== RustyGraph Weighted Graphs Example ===\n");

    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();

    // Example 1: Natural visibility with value difference weights
    println!("1. Natural visibility - Value difference weights:");
    let edges = visibility_weighted(&series, VisibilityType::Natural, |_, _, vi: f64, vj: f64| {
        (vj - vi).abs()
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.2}", src, dst, weight);
    }

    // Example 2: Natural visibility with temporal distance weights
    println!("\n2. Natural visibility - Temporal distance weights:");
    let edges = visibility_weighted(&series, VisibilityType::Natural, |i, j, _, _| {
        (j - i) as f64
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.0}", src, dst, weight);
    }

    // Example 3: Horizontal visibility with geometric mean weights
    println!("\n3. Horizontal visibility - Geometric mean weights:");
    let edges = visibility_weighted(&series, VisibilityType::Horizontal, |_, _, vi: f64, vj: f64| {
        (vi * vj).sqrt()
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.2}", src, dst, weight);
    }

    // Example 4: Custom complex weight function
    println!("\n4. Custom weight function (normalized difference):");
    let max_val = series.values.iter().filter_map(|&v| v).fold(f64::NEG_INFINITY, |a, b| a.max(b.into()));
    let min_val = series.values.iter().filter_map(|&v| v).fold(f64::INFINITY, |a, b| a.min(b.into()));
    let range = max_val - min_val;

    let edges = visibility_weighted(&series, VisibilityType::Natural, |_, _, vi: f64, vj: f64| {
        if range > 0.0 {
            (vj - vi).abs() / range
        } else {
            0.0
        }
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.3}", src, dst, weight);
    }

    // Example 5: Constant weights (equivalent to unweighted)
    println!("\n5. Constant weights (unweighted equivalent):");
    let edges = visibility_weighted(&series, VisibilityType::Horizontal, |_, _, _, _| 1.0);

    println!("  All edges have weight 1.0");
    println!("  Total edges: {}", edges.len());
}

