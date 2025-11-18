//! Community detection and GraphML export example.
//!
//! This example demonstrates:
//! - Community detection in visibility graphs
//! - Connected components analysis
//! - GraphML export for use with graph analysis tools

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Community Detection Example ===\n");

    // Create a larger time series to show community structure
    let series = TimeSeries::from_raw(vec![
        1.0, 5.0, 2.0, 6.0, 1.5, 5.5, 2.5,  // Group 1
        8.0, 12.0, 9.0, 13.0, 8.5, 12.5, 9.5,  // Group 2
        4.0, 7.0, 4.5, 7.5, 4.2  // Group 3
    ])?;

    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;

    // 1. Community Detection
    println!("1. COMMUNITY DETECTION");
    println!("──────────────────────");

    let communities = graph.detect_communities();

    println!("Found {} communities", communities.num_communities);
    println!("Modularity score: {:.4}", communities.modularity);
    println!();

    // Show community sizes
    let sizes = communities.community_sizes();
    println!("Community sizes:");
    for (i, size) in sizes.iter().enumerate() {
        println!("  Community {}: {} nodes", i, size);
    }
    println!();

    // Show nodes in each community
    println!("Community assignments:");
    for comm_id in 0..communities.num_communities {
        let nodes = communities.get_community_nodes(comm_id);
        println!("  Community {}: nodes {:?}", comm_id, nodes);
    }

    // 2. Connected Components
    println!("\n2. CONNECTED COMPONENTS");
    println!("───────────────────────");

    let components = graph.connected_components();
    let num_components = components.iter().max().map(|&m| m + 1).unwrap_or(0);

    println!("Number of connected components: {}", num_components);

    if num_components > 1 {
        println!("Graph is disconnected!");
        for comp_id in 0..num_components {
            let nodes_in_comp: Vec<_> = components.iter()
                .enumerate()
                .filter(|(_, &c)| c == comp_id)
                .map(|(n, _)| n)
                .collect();
            println!("  Component {}: nodes {:?}", comp_id, nodes_in_comp);
        }
    } else {
        println!("Graph is fully connected ✓");
    }

    // 3. GraphML Export
    println!("\n3. GRAPHML EXPORT");
    println!("─────────────────");

    let graphml = graph.to_graphml();

    println!("GraphML export (first 20 lines):");
    for (i, line) in graphml.lines().enumerate() {
        if i >= 20 {
            println!("... ({} more lines)", graphml.lines().count() - 20);
            break;
        }
        println!("{}", line);
    }

    println!("\nTo visualize:");
    println!("  1. Save to file: std::fs::write(\"graph.graphml\", graphml)?;");
    println!("  2. Open in Gephi, yEd, or Cytoscape");
    println!("  3. Apply layout algorithms (Force Atlas 2, etc.)");
    println!("  4. Analyze communities visually");

    // 4. Compare with Graph Metrics
    println!("\n4. COMMUNITY VS METRICS");
    println!("───────────────────────");

    let stats = graph.compute_statistics();
    println!("Graph metrics:");
    println!("  Clustering coefficient: {:.4}", stats.average_clustering);
    println!("  Density: {:.4}", stats.density);
    println!("  Diameter: {}", stats.diameter);

    println!("\nCommunity structure:");
    println!("  Number of communities: {}", communities.num_communities);
    println!("  Modularity: {:.4}", communities.modularity);

    if communities.modularity > 0.3 {
        println!("  → Strong community structure detected!");
    } else if communities.modularity > 0.1 {
        println!("  → Moderate community structure");
    } else {
        println!("  → Weak community structure");
    }

    // 5. Export Everything
    println!("\n5. EXPORT CAPABILITIES");
    println!("──────────────────────");

    println!("Available export formats:");
    println!("  ✓ JSON - graph.to_json()");
    println!("  ✓ CSV Edge List - graph.to_edge_list_csv()");
    println!("  ✓ CSV Adjacency Matrix - graph.to_adjacency_matrix_csv()");
    println!("  ✓ CSV Features - graph.features_to_csv()");
    println!("  ✓ GraphViz DOT - graph.to_dot()");
    println!("  ✓ GraphML - graph.to_graphml() ⭐ NEW");

    // 6. Workflow Example
    println!("\n6. ANALYSIS WORKFLOW");
    println!("────────────────────");

    println!("Complete analysis workflow:");
    println!("  1. Create graph from time series");
    println!("  2. Detect communities");
    println!("  3. Export to GraphML");
    println!("  4. Open in Gephi/yEd");
    println!("  5. Apply layout algorithms");
    println!("  6. Color nodes by community");
    println!("  7. Identify patterns visually");
    println!("  8. Export publication-quality figures");

    Ok(())
}

