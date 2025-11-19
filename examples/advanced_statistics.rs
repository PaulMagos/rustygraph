use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  RustyGraph Advanced Statistics Showcase          ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    // Create a sample time series
    let size = 50;
    let data: Vec<f64> = (0..size)
        .map(|i| {
            let t = i as f64 * 0.2;
            (t).sin() + 0.3 * (2.0 * t).cos()
        })
        .collect();
    let series = TimeSeries::from_raw(data)?;

    println!("📊 Building visibility graph from {} data points...\n", size);
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;

    // ========================================================================
    // 1. Comprehensive Statistics Summary
    // ========================================================================
    println!("1️⃣  COMPREHENSIVE STATISTICS");
    println!("═══════════════════════════════════════════════════\n");

    let stats = graph.compute_statistics();
    println!("{}", stats);

    // ========================================================================
    // 2. Degree Distribution Analysis
    // ========================================================================
    println!("\n2️⃣  DEGREE DISTRIBUTION ANALYSIS");
    println!("═══════════════════════════════════════════════════\n");

    let degree_dist = graph.degree_distribution_histogram();
    println!("Degree Distribution (degree → count):");
    for (degree, count) in degree_dist.iter().enumerate() {
        if *count > 0 {
            println!("  Degree {}: {} nodes", degree, count);
        }
    }

    let degree_entropy = graph.degree_entropy();
    println!("\nDegree Entropy: {:.4} bits", degree_entropy);
    println!("(Higher entropy = more diverse degree distribution)");

    // ========================================================================
    // 3. Centrality Measures
    // ========================================================================
    println!("\n3️⃣  CENTRALITY ANALYSIS");
    println!("═══════════════════════════════════════════════════\n");

    let degree_cent = graph.degree_centrality();
    let betweenness_cent = graph.betweenness_centrality_batch();
    let closeness_cent = graph.closeness_centrality();

    // Find top 5 nodes by each centrality measure
    println!("📈 Top 5 Nodes by Degree Centrality:");
    let mut deg_ranked: Vec<_> = degree_cent.iter().enumerate().collect();
    deg_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (rank, (node, centrality)) in deg_ranked.iter().take(5).enumerate() {
        println!("  {}. Node {}: {:.4}", rank + 1, node, centrality);
    }

    println!("\n📊 Top 5 Nodes by Betweenness Centrality:");
    let mut bet_ranked: Vec<_> = betweenness_cent.iter().enumerate().collect();
    bet_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (rank, (node, centrality)) in bet_ranked.iter().take(5).enumerate() {
        println!("  {}. Node {}: {:.4}", rank + 1, node, centrality);
    }

    println!("\n📍 Top 5 Nodes by Closeness Centrality:");
    let mut close_ranked: Vec<_> = closeness_cent.iter().enumerate().collect();
    close_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (rank, (node, centrality)) in close_ranked.iter().take(5).enumerate() {
        println!("  {}. Node {}: {:.4}", rank + 1, node, centrality);
    }

    // ========================================================================
    // 4. Network Structure Analysis
    // ========================================================================
    println!("\n4️⃣  NETWORK STRUCTURE");
    println!("═══════════════════════════════════════════════════\n");

    println!("Connectivity:");
    println!("  Connected: {}", stats.is_connected);
    println!("  Components: {}", stats.num_components);
    println!("  Largest component size: {}", stats.largest_component_size);

    println!("\nClustering:");
    println!("  Average clustering: {:.4}", stats.average_clustering);
    println!("  Global clustering: {:.4}", stats.global_clustering);

    println!("\nDistance Metrics:");
    println!("  Average path length: {:.2}", stats.average_path_length);
    println!("  Diameter: {}", stats.diameter);
    println!("  Radius: {}", stats.radius);

    println!("\nNetwork Properties:");
    println!("  Density: {:.4}", stats.density);
    println!("  Assortativity: {:.4}", stats.assortativity);
    if stats.assortativity > 0.0 {
        println!("  → Assortative (nodes connect to similar-degree nodes)");
    } else if stats.assortativity < 0.0 {
        println!("  → Disassortative (nodes connect to different-degree nodes)");
    } else {
        println!("  → Neutral mixing");
    }

    // ========================================================================
    // 5. Statistical Distribution
    // ========================================================================
    println!("\n5️⃣  STATISTICAL DISTRIBUTION");
    println!("═══════════════════════════════════════════════════\n");

    println!("Degree Statistics:");
    println!("  Mean: {:.2}", stats.average_degree);
    println!("  Std Dev: {:.2}", stats.degree_std_dev);
    println!("  Variance: {:.2}", stats.degree_variance);
    println!("  Min: {}", stats.min_degree);
    println!("  Max: {}", stats.max_degree);
    println!("  Range: {}", stats.max_degree - stats.min_degree);

    // ========================================================================
    // 6. Comparison with Different Graph Types
    // ========================================================================
    println!("\n6️⃣  HORIZONTAL VS NATURAL VISIBILITY");
    println!("═══════════════════════════════════════════════════\n");

    let h_graph = VisibilityGraph::from_series(&series)
        .horizontal_visibility()?;
    let h_stats = h_graph.compute_statistics();

    println!("Comparison:");
    println!("                      Natural     Horizontal");
    println!("  Edges:             {:>8}    {:>8}", stats.edge_count, h_stats.edge_count);
    println!("  Avg Degree:        {:>8.2}    {:>8.2}", stats.average_degree, h_stats.average_degree);
    println!("  Clustering:        {:>8.4}    {:>8.4}", stats.average_clustering, h_stats.average_clustering);
    println!("  Avg Path Length:   {:>8.2}    {:>8.2}", stats.average_path_length, h_stats.average_path_length);
    println!("  Diameter:          {:>8}    {:>8}", stats.diameter, h_stats.diameter);
    println!("  Assortativity:     {:>8.4}    {:>8.4}", stats.assortativity, h_stats.assortativity);

    // ========================================================================
    // 7. Key Insights
    // ========================================================================
    println!("\n7️⃣  KEY INSIGHTS");
    println!("═══════════════════════════════════════════════════\n");

    println!("Network Characteristics:");

    if stats.density > 0.5 {
        println!("  ✓ Dense network (many connections)");
    } else if stats.density > 0.2 {
        println!("  ✓ Moderately connected network");
    } else {
        println!("  ✓ Sparse network (few connections)");
    }

    if stats.average_clustering > 0.5 {
        println!("  ✓ High clustering (nodes form tight groups)");
    } else if stats.average_clustering > 0.2 {
        println!("  ✓ Moderate clustering");
    } else {
        println!("  ✓ Low clustering (dispersed structure)");
    }

    if stats.is_connected {
        println!("  ✓ Fully connected (all nodes reachable)");
    } else {
        println!("  ✓ Disconnected ({} components)", stats.num_components);
    }

    let cv = stats.degree_std_dev / stats.average_degree;
    println!("  ✓ Degree variation: {:.2}x (coefficient of variation)", cv);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║  ✨ Analysis Complete!                              ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!("\n🎯 Advanced statistics features:");
    println!("   ✓ Comprehensive graph metrics");
    println!("   ✓ Multiple centrality measures");
    println!("   ✓ Degree distribution analysis");
    println!("   ✓ Clustering coefficients");
    println!("   ✓ Network topology analysis");
    println!("   ✓ Assortativity measurement");
    println!("   ✓ Component analysis");
    println!("\n📚 All metrics computed efficiently with optimized algorithms!");

    Ok(())
}

