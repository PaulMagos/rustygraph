//! Advanced features example demonstrating export, metrics, and directed graphs.
//!
//! This example shows:
//! - Directed vs undirected graphs
//! - Graph metrics (clustering coefficient, path length, diameter)
//! - Export formats (JSON, CSV edge list, adjacency matrix, features CSV)

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Advanced Features Example ===\n");

    // Create a time series
    let series = TimeSeries::from_raw(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;

    // 1. Directed Graph
    println!("1. DIRECTED GRAPH");
    println!("─────────────────");
    let directed_graph = VisibilityGraph::from_series(&series)
        .with_direction(GraphDirection::Directed)
        .natural_visibility()?;

    println!("Directed graph: {} nodes, {} edges",
        directed_graph.node_count,
        directed_graph.edges().len()
    );

    // 2. Undirected Graph (default)
    println!("\n2. UNDIRECTED GRAPH");
    println!("───────────────────");
    let undirected_graph = VisibilityGraph::from_series(&series)
        .with_direction(GraphDirection::Undirected)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::LocalSlope)
        )
        .natural_visibility()?;

    println!("Undirected graph: {} nodes, {} edges",
        undirected_graph.node_count,
        undirected_graph.edges().len()
    );

    // 3. Graph Metrics
    println!("\n3. GRAPH METRICS");
    println!("────────────────");

    // Clustering coefficient
    let avg_clustering = undirected_graph.average_clustering_coefficient();
    println!("Average clustering coefficient: {:.4}", avg_clustering);

    for i in 0..undirected_graph.node_count {
        if let Some(cc) = undirected_graph.clustering_coefficient(i) {
            println!("  Node {}: {:.4}", i, cc);
        }
    }

    // Path metrics
    let avg_path = undirected_graph.average_path_length();
    let diameter = undirected_graph.diameter();
    println!("\nAverage path length: {:.2}", avg_path);
    println!("Graph diameter: {}", diameter);

    // Connectivity
    if undirected_graph.is_connected() {
        println!("Graph is connected ✓");
    } else {
        println!("Graph is not connected");
    }

    // Density
    let density = undirected_graph.density();
    println!("Graph density: {:.4}", density);

    // Degree distribution
    println!("\nDegree distribution:");
    let dist = undirected_graph.degree_distribution();
    let mut degrees: Vec<_> = dist.keys().collect();
    degrees.sort();
    for degree in degrees {
        println!("  Degree {}: {} nodes", degree, dist[degree]);
    }

    // 4. Export Formats
    println!("\n4. EXPORT FORMATS");
    println!("─────────────────");

    // JSON export
    println!("\n▸ JSON Export:");
    let json = undirected_graph.to_json(ExportOptions {
        include_weights: true,
        include_features: true,
        pretty: true,
    });
    println!("{}", json);

    // CSV edge list
    println!("\n▸ CSV Edge List:");
    let edge_csv = undirected_graph.to_edge_list_csv(true);
    println!("{}", edge_csv);

    // Adjacency matrix CSV
    println!("▸ Adjacency Matrix CSV:");
    let adj_csv = undirected_graph.to_adjacency_matrix_csv();
    let lines: Vec<&str> = adj_csv.lines().take(5).collect();
    for line in lines {
        println!("{}", line);
    }
    if undirected_graph.node_count > 4 {
        println!("... ({} more rows)", undirected_graph.node_count - 4);
    }

    // Features CSV
    println!("\n▸ Features CSV:");
    let features_csv = undirected_graph.features_to_csv();
    println!("{}", features_csv);

    // 5. Shortest Paths
    println!("\n5. SHORTEST PATHS");
    println!("─────────────────");
    for i in 0..3.min(undirected_graph.node_count) {
        for j in (i+1)..undirected_graph.node_count {
            if let Some(dist) = undirected_graph.shortest_path_length(i, j) {
                println!("Distance from {} to {}: {}", i, j, dist);
            }
        }
    }

    Ok(())
}

