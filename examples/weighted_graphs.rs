//! Weighted visibility graph example.
//!
//! This example demonstrates how to create weighted visibility graphs
//! with custom edge weight functions.

use rustygraph::algorithms::{natural_visibility, horizontal_visibility, visibility_weighted};

fn main() {
    println!("=== RustyGraph Weighted Graphs Example ===\n");

    let series: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 1.0];

    // Example 1: Natural visibility with value difference weights
    println!("1. Natural visibility - Value difference weights:");
    let edges = visibility_weighted(&series, natural_visibility, |_, _, vi, vj| {
        (vj - vi).abs()
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.2}", src, dst, weight);
    }

    // Example 2: Natural visibility with temporal distance weights
    println!("\n2. Natural visibility - Temporal distance weights:");
    let edges = visibility_weighted(&series, natural_visibility, |i, j, _, _| {
        (j - i) as f64
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.0}", src, dst, weight);
    }

    // Example 3: Horizontal visibility with geometric mean weights
    println!("\n3. Horizontal visibility - Geometric mean weights:");
    let edges = visibility_weighted(&series, horizontal_visibility, |_, _, vi, vj| {
        (vi * vj).sqrt()
    });

    for (src, dst, weight) in &edges {
        println!("  {} -> {} : weight = {:.2}", src, dst, weight);
    }

    // Example 4: Custom complex weight function
    println!("\n4. Custom weight function (normalized difference):");
    let max_val = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let range = max_val - min_val;

    let edges = visibility_weighted(&series, natural_visibility(), |_, _, vi, vj| {
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
    let edges = visibility_weighted(&series, horizontal_visibility, |_, _, _, _| 1.0);

    println!("  All edges have weight 1.0");
    println!("  Total edges: {}", edges.len());
}

