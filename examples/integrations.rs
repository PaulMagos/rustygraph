//! Integration features example.
//!
//! Demonstrates petgraph, ndarray, and Python bindings integration.

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Integrations Example ===\n");

    // Create sample graph
    let data = datasets::sine_wave(20, 2.0, 1.0);
    let series = TimeSeries::from_raw(data)?;
    let graph = VisibilityGraph::from_series(&series).natural_visibility()?;

    println!("Graph: {} nodes, {} edges, density: {:.4}\n",
        graph.node_count, graph.edges().len(), graph.density());

    // 1. Petgraph Integration
    println!("1. PETGRAPH INTEGRATION");
    #[cfg(feature = "petgraph-integration")]
    {
        let pg = graph.to_petgraph();
        println!("✓ Converted to petgraph: {} nodes, {} edges", pg.node_count(), pg.edge_count());

        let distances = graph.dijkstra_shortest_paths(0);
        println!("  Shortest paths from node 0 (first 3): {:?}",
            distances.iter().take(3).collect::<Vec<_>>());

        let mst = graph.minimum_spanning_tree();
        println!("  Minimum spanning tree: {} edges", mst.len());
    }
    #[cfg(not(feature = "petgraph-integration"))]
    println!("✗ Not available. Enable with: --features petgraph-integration");

    // 2. ndarray Support
    println!("\n2. NDARRAY SUPPORT");
    #[cfg(feature = "ndarray-support")]
    {
        let adj = graph.to_ndarray_adjacency();
        let lap = graph.to_ndarray_laplacian();
        let eigenvalue = graph.dominant_eigenvalue(100);

        println!("✓ Adjacency matrix: {:?}, {} non-zero",
            adj.shape(), adj.iter().filter(|&&x| x > 0.0).count());
        println!("  Laplacian: {:?}, dominant eigenvalue: {:.4}",
            lap.shape(), eigenvalue);
    }
    #[cfg(not(feature = "ndarray-support"))]
    println!("✗ Not available. Enable with: --features ndarray-support");

    // 3. Python Bindings
    println!("\n3. PYTHON BINDINGS");
    #[cfg(feature = "python-bindings")]
    {
        println!("✓ Available. Build with: maturin develop --features python-bindings");
        println!("  Usage: import rustygraph; graph = rustygraph.TimeSeries(data).natural_visibility()");
    }
    #[cfg(not(feature = "python-bindings"))]
    println!("✗ Not available. Build with: maturin develop --features python-bindings");

    // 4. Machine Learning Data Preparation
    println!("\n4. MACHINE LEARNING DATA PREPARATION");
    {
        // Create windowed time series for ML
        let windows = WindowedTimeSeries::from_series(&series, 5, 1)?;
        println!("✓ Created {} windows of size {} from time series", windows.len(), windows.window_size);

        // Split data for training
        let split = split_windowed_data(windows, SplitStrategy::TimeBased {
            train_frac: 0.7,
            val_frac: 0.2,
        })?;
        println!("  Data split: {} train, {} val, {} test windows",
            split.train.len(),
            split.val.as_ref().map(|v| v.len()).unwrap_or(0),
            split.test.len());
    }

    // 5. Burn Integration (Rust ML)
    println!("\n5. BURN INTEGRATION (Rust ML)");
    #[cfg(feature = "burn-integration")]
    {
        println!("✓ Available. Build with: --features burn-integration");
        println!("  Provides Burn-compatible datasets and data loaders for time series ML");
    }
    #[cfg(not(feature = "burn-integration"))]
    println!("✗ Not available. Enable with: --features burn-integration");

    // 6. Summary
    println!("\n6. SUMMARY");

    let mut available: Vec<&str> = Vec::new();
    #[cfg(feature = "petgraph-integration")]
    available.push("petgraph");
    #[cfg(feature = "ndarray-support")]
    available.push("ndarray");
    #[cfg(feature = "python-bindings")]
    available.push("Python");
    #[cfg(feature = "burn-integration")]
    available.push("Burn");

    available.push("ML DataLoader"); // Always available now

    println!("✓ Available integrations: {}",
        if available.is_empty() {
            "none (enable with features)".to_string()
        } else {
            available.join(", ")
        });

    Ok(())
}
