//! Performance and I/O features example.
//!
//! This example demonstrates:
//! - CSV import for time series
//! - Graph statistics summary
//! - Parallel feature computation (with 'parallel' feature)

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Performance & I/O Example ===\n");

    // 1. CSV Import
    println!("1. CSV IMPORT");
    println!("─────────────");

    let csv_data = r#"timestamp,value
0.0,1.5
1.0,3.2
2.0,2.1
3.0,4.8
4.0,3.5
5.0,5.2
6.0,4.1
7.0,6.3
8.0,5.8
9.0,7.1"#;

    let series = TimeSeries::<f64>::from_csv_string(
        csv_data,
        CsvImportOptions {
            has_header: true,
            timestamp_column: Some(0),
            value_column: 1,
            delimiter: ',',
            missing_value: String::new(),
        },
    )?;

    println!("✓ Imported {} data points from CSV", series.len());
    println!("  First 3 values: {:?}", &series.values[0..3]);

    // 2. Build Graph with Features
    println!("\n2. GRAPH CONSTRUCTION");
    println!("─────────────────────");

    let graph = VisibilityGraph::from_series(&series)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::DeltaBackward)
                .add_builtin(BuiltinFeature::LocalSlope)
                .add_builtin(BuiltinFeature::IsLocalMax)
                .add_builtin(BuiltinFeature::IsLocalMin)
        )
        .with_direction(GraphDirection::Undirected)
        .natural_visibility()?;

    println!("✓ Graph constructed with {} nodes and {} edges",
        graph.node_count,
        graph.edges().len()
    );

    // 3. Graph Statistics Summary
    println!("\n3. GRAPH STATISTICS");
    println!("───────────────────");

    let stats = graph.compute_statistics();
    println!("{}", stats);

    // 4. Detailed Metrics
    println!("4. DETAILED METRICS");
    println!("───────────────────");

    // Degree distribution
    println!("\nDegree Distribution:");
    let dist = graph.degree_distribution();
    let mut degrees: Vec<_> = dist.keys().collect();
    degrees.sort();
    for degree in degrees {
        let count = dist[degree];
        let bar = "█".repeat(count);
        println!("  Degree {}: {} {}", degree, count, bar);
    }

    // Clustering coefficients
    println!("\nClustering Coefficients (first 5 nodes):");
    for i in 0..5.min(graph.node_count) {
        if let Some(cc) = graph.clustering_coefficient(i) {
            println!("  Node {}: {:.4}", i, cc);
        }
    }

    // Shortest paths
    println!("\nShortest Path Examples:");
    for i in 0..3.min(graph.node_count) {
        for j in (i+1)..5.min(graph.node_count) {
            if let Some(dist) = graph.shortest_path_length(i, j) {
                println!("  {} → {}: {} hops", i, j, dist);
            }
        }
    }

    // 5. Feature Summary
    println!("\n5. FEATURE SUMMARY");
    println!("──────────────────");

    if !graph.node_features.is_empty() {
        println!("Features computed for {} nodes:", graph.node_count);

        // Show first node's features
        if let Some(features) = graph.node_features(0) {
            println!("\nNode 0 features:");
            let mut feature_names: Vec<_> = features.keys().collect();
            feature_names.sort();
            for name in feature_names {
                println!("  {}: {:.4}", name, features[name]);
            }
        }

        // Count peaks and valleys
        let peak_count = graph.node_features.iter()
            .filter(|f| f.get("is_local_max").map(|&v| v > 0.5).unwrap_or(false))
            .count();
        let valley_count = graph.node_features.iter()
            .filter(|f| f.get("is_local_min").map(|&v| v > 0.5).unwrap_or(false))
            .count();

        println!("\nLocal extrema:");
        println!("  Peaks: {}", peak_count);
        println!("  Valleys: {}", valley_count);
    }

    // 6. Export Examples
    println!("\n6. EXPORT EXAMPLES");
    println!("──────────────────");

    println!("\n▸ Edge List (first 5 edges):");
    let edge_csv = graph.to_edge_list_csv(true);
    let lines: Vec<&str> = edge_csv.lines().take(6).collect();
    for line in lines {
        println!("{}", line);
    }

    println!("\n▸ Features CSV (first 5 rows):");
    let features_csv = graph.features_to_csv();
    let lines: Vec<&str> = features_csv.lines().take(6).collect();
    for line in lines {
        println!("{}", line);
    }

    // 7. Performance Info
    println!("\n7. PERFORMANCE");
    println!("──────────────");

    #[cfg(feature = "parallel")]
    println!("✓ Parallel feature computation: ENABLED");
    #[cfg(not(feature = "parallel"))]
    println!("○ Parallel feature computation: disabled");

    #[cfg(feature = "csv-import")]
    println!("✓ CSV import: ENABLED");
    #[cfg(not(feature = "csv-import"))]
    println!("○ CSV import: disabled");

    println!("\nTip: Enable parallel processing with:");
    println!("  cargo build --features parallel");

    Ok(())
}

