//! Advanced analytics example.
//!
//! This example demonstrates:
//! - Betweenness centrality computation
//! - DOT format export for GraphViz
//! - Batch processing multiple time series
//! - Graph comparison metrics

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Advanced Analytics Example ===\n");

    // 1. Betweenness Centrality
    println!("1. BETWEENNESS CENTRALITY");
    println!("─────────────────────────");

    let series = TimeSeries::from_raw(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0])?;
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;

    println!("Computing betweenness centrality for all nodes:");
    let centralities = graph.betweenness_centrality_all();
    for (i, bc) in centralities.iter().enumerate() {
        println!("  Node {}: {:.4}", i, bc);
    }

    let max_bc = centralities.iter().fold(0.0f64, |a, &b| a.max(b));
    let most_central = centralities.iter().position(|&x| x == max_bc).unwrap();
    println!("\nMost central node: {} (BC = {:.4})", most_central, max_bc);

    // 2. GraphViz DOT Export
    println!("\n2. GRAPHVIZ DOT EXPORT");
    println!("──────────────────────");

    let dot = graph.to_dot();
    println!("DOT format output:");
    println!("{}", dot);

    println!("Save to file and visualize:");
    println!("  echo '{}' > graph.dot", dot.lines().next().unwrap());
    println!("  dot -Tpng graph.dot -o graph.png");

    // Custom labels
    let dot_custom = graph.to_dot_with_labels(|i| {
        format!("t{}", i)
    });
    println!("\nWith custom labels (first few lines):");
    for line in dot_custom.lines().take(8) {
        println!("{}", line);
    }

    // 3. Batch Processing
    println!("\n3. BATCH PROCESSING");
    println!("───────────────────");

    let series1 = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 3.0])?;
    let series2 = TimeSeries::from_raw(vec![2.0, 1.0, 3.0, 2.0, 4.0])?;
    let series3 = TimeSeries::from_raw(vec![1.0, 2.0, 1.0, 3.0, 2.0])?;

    let batch_results = BatchProcessor::new()
        .add_series(&series1, "Stock A")
        .add_series(&series2, "Stock B")
        .add_series(&series3, "Stock C")
        .process_natural()?;

    batch_results.print_summary();

    // 4. Graph Comparison
    println!("\n4. GRAPH COMPARISON");
    println!("───────────────────");

    let g1 = VisibilityGraph::from_series(&series1).natural_visibility()?;
    let g2 = VisibilityGraph::from_series(&series2).natural_visibility()?;
    let g3 = VisibilityGraph::from_series(&series3).natural_visibility()?;

    println!("Comparing graphs:");

    let sim_12 = compare_graphs(&g1, &g2);
    println!("\nStock A vs Stock B:");
    println!("  Edge overlap: {:.2}%", sim_12.get("edge_overlap").unwrap_or(&0.0) * 100.0);
    if let Some(corr) = sim_12.get("degree_correlation") {
        println!("  Degree correlation: {:.4}", corr);
    }
    println!("  Clustering difference: {:.4}", sim_12.get("clustering_diff").unwrap_or(&0.0));
    println!("  Density difference: {:.4}", sim_12.get("density_diff").unwrap_or(&0.0));

    let sim_13 = compare_graphs(&g1, &g3);
    println!("\nStock A vs Stock C:");
    println!("  Edge overlap: {:.2}%", sim_13.get("edge_overlap").unwrap_or(&0.0) * 100.0);
    if let Some(corr) = sim_13.get("degree_correlation") {
        println!("  Degree correlation: {:.4}", corr);
    }
    println!("  Clustering difference: {:.4}", sim_13.get("clustering_diff").unwrap_or(&0.0));
    println!("  Density difference: {:.4}", sim_13.get("density_diff").unwrap_or(&0.0));

    let sim_23 = compare_graphs(&g2, &g3);
    println!("\nStock B vs Stock C:");
    println!("  Edge overlap: {:.2}%", sim_23.get("edge_overlap").unwrap_or(&0.0) * 100.0);
    if let Some(corr) = sim_23.get("degree_correlation") {
        println!("  Degree correlation: {:.4}", corr);
    }
    println!("  Clustering difference: {:.4}", sim_23.get("clustering_diff").unwrap_or(&0.0));
    println!("  Density difference: {:.4}", sim_23.get("density_diff").unwrap_or(&0.0));

    // 5. Centrality Analysis
    println!("\n5. CENTRALITY ANALYSIS");
    println!("──────────────────────");

    let large_series = TimeSeries::from_raw(vec![
        1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 3.0, 6.0
    ])?;
    let large_graph = VisibilityGraph::from_series(&large_series)
        .natural_visibility()?;

    let bc_all = large_graph.betweenness_centrality_all();
    let degrees = large_graph.degree_sequence();

    println!("Node analysis:");
    println!("  Node | Degree | Betweenness | Role");
    println!("  -----|--------|-------------|-----");
    for i in 0..large_graph.node_count {
        let role = if bc_all[i] > 0.1 {
            "Hub"
        } else if degrees[i] > 3 {
            "Connector"
        } else if degrees[i] == 1 {
            "Peripheral"
        } else {
            "Regular"
        };
        println!("  {:4} | {:6} | {:11.4} | {}", i, degrees[i], bc_all[i], role);
    }

    println!("\n6. VISUALIZATION WORKFLOW");
    println!("─────────────────────────");

    println!("To visualize graphs:");
    println!("  1. Export to DOT: graph.to_dot()");
    println!("  2. Save to file: std::fs::write(\"graph.dot\", dot)");
    println!("  3. Render with GraphViz:");
    println!("     - PNG: dot -Tpng graph.dot -o graph.png");
    println!("     - SVG: dot -Tsvg graph.dot -o graph.svg");
    println!("     - PDF: dot -Tpdf graph.dot -o graph.pdf");
    println!("  4. Or use online: https://dreampuf.github.io/GraphvizOnline/");

    Ok(())
}

