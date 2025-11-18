//! Advanced export formats example.
//!
//! This example demonstrates:
//! - NPY export for NumPy/Python integration
//! - Parquet export for big data analytics
//! - HDF5 export for scientific computing
//! - Usage patterns for each format

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Advanced Export Formats Example ===\n");

    // Create a sample graph
    let data = datasets::multi_frequency(50, &[1.0, 3.0, 5.0]);
    let series = TimeSeries::from_raw(data)?;

    let graph = VisibilityGraph::from_series(&series)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::LocalSlope)
        )
        .natural_visibility()?;

    println!("Created visibility graph:");
    println!("  Nodes: {}", graph.node_count);
    println!("  Edges: {}", graph.edges().len());
    println!("  Density: {:.4}", graph.density());

    // 1. NPY Export (NumPy arrays)
    println!("\n1. NPY EXPORT (NumPy Integration)");
    println!("──────────────────────────────────");

    #[cfg(feature = "npy-export")]
    {
        std::fs::create_dir_all("output")?;
        let output_path = "output/graph";
        graph.to_npy(output_path)?;

        println!("✓ Exported to NPY format:");
        println!("  - graph_edges.npy       (edge list as NumPy array)");
        println!("  - graph_adjacency.npy   (adjacency matrix)");
        println!("  - graph_degrees.npy     (degree sequence)");
        println!("\nPython usage:");
        println!("  import numpy as np");
        println!("  edges = np.load('graph_edges.npy')");
        println!("  adj = np.load('graph_adjacency.npy')");
        println!("  degrees = np.load('graph_degrees.npy')");
    }

    #[cfg(not(feature = "npy-export"))]
    {
        println!("✗ NPY export not available");
        println!("  Run with: cargo run --example export_formats --features npy-export");
    }

    // 2. Parquet Export (Big Data)
    println!("\n2. PARQUET EXPORT (Big Data Analytics)");
    println!("───────────────────────────────────────");

    #[cfg(feature = "parquet-export")]
    {
        std::fs::create_dir_all("output")?;
        let output_path = "output/graph.parquet";
        graph.to_parquet(output_path)?;

        println!("✓ Exported to Parquet format:");
        println!("  - graph.parquet (columnar format, SNAPPY compressed)");
        println!("\nUse cases:");
        println!("  - Apache Spark: spark.read.parquet('graph.parquet')");
        println!("  - Pandas: pd.read_parquet('graph.parquet')");
        println!("  - DuckDB: SELECT * FROM 'graph.parquet'");
        println!("  - Polars: pl.read_parquet('graph.parquet')");
        println!("\nColumns: source, target, weight");
    }

    #[cfg(not(feature = "parquet-export"))]
    {
        println!("✗ Parquet export not available");
        println!("  Run with: cargo run --example export_formats --features parquet-export");
    }

    // 3. HDF5 Export (Scientific Computing)
    println!("\n3. HDF5 EXPORT (Scientific Computing)");
    println!("──────────────────────────────────────");

    #[cfg(feature = "hdf5-export")]
    {
        std::fs::create_dir_all("output")?;
        let output_path = "output/graph.h5";
        graph.to_hdf5(output_path)?;

        println!("✓ Exported to HDF5 format:");
        println!("  - graph.h5 (hierarchical data format)");
        println!("\nStructure:");
        println!("  /edges/");
        println!("    sources  - Source node indices");
        println!("    targets  - Target node indices");
        println!("    weights  - Edge weights");
        println!("  /graph/");
        println!("    adjacency - Adjacency matrix");
        println!("    degrees   - Degree sequence");
        println!("  /metadata/");
        println!("    node_count - Total nodes");
        println!("    edge_count - Total edges");
        println!("\nPython usage:");
        println!("  import h5py");
        println!("  with h5py.File('graph.h5', 'r') as f:");
        println!("      edges = f['edges/sources'][:]");
        println!("      adj = f['graph/adjacency'][:]");
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        println!("✗ HDF5 export not available");
        println!("  Run with: cargo run --example export_formats --features hdf5-export");
    }

    // 4. Format Comparison
    println!("\n4. FORMAT COMPARISON");
    println!("────────────────────");

    println!("│ Format  │ Best For                  │ Tools                      │");
    println!("├─────────┼───────────────────────────┼────────────────────────────┤");
    println!("│ NPY     │ NumPy/Python integration  │ NumPy, SciPy, scikit-learn │");
    println!("│ Parquet │ Big data analytics        │ Spark, Pandas, DuckDB      │");
    println!("│ HDF5    │ Scientific computing      │ MATLAB, Python h5py        │");
    println!("│ JSON    │ Web APIs, visualization   │ D3.js, JavaScript          │");
    println!("│ CSV     │ Excel, simple tools       │ Excel, R, any tool         │");
    println!("│ GraphML │ Graph visualization       │ Gephi, Cytoscape, yEd      │");
    println!("│ DOT     │ GraphViz rendering        │ GraphViz, dot command      │");

    // 5. Use Case Examples
    println!("\n5. USE CASE EXAMPLES");
    println!("────────────────────");

    println!("\nMachine Learning Pipeline:");
    println!("  1. Rust: Create visibility graph");
    println!("  2. Export to NPY");
    println!("  3. Python: Load with NumPy");
    println!("  4. Train ML model with scikit-learn");

    println!("\nBig Data Analysis:");
    println!("  1. Rust: Process time series");
    println!("  2. Export to Parquet");
    println!("  3. Apache Spark: Distributed analysis");
    println!("  4. Visualize with dashboards");

    println!("\nScientific Research:");
    println!("  1. Rust: Compute graphs");
    println!("  2. Export to HDF5");
    println!("  3. MATLAB/Python: Further analysis");
    println!("  4. Publish results");

    // 6. Performance Characteristics
    println!("\n6. PERFORMANCE CHARACTERISTICS");
    println!("───────────────────────────────");

    println!("\nFile Sizes (approximate, for 1000 nodes, 5000 edges):");
    println!("  NPY:     ~120 KB (3 files, uncompressed)");
    println!("  Parquet: ~40 KB (compressed with SNAPPY)");
    println!("  HDF5:    ~100 KB (hierarchical, compressed)");
    println!("  JSON:    ~250 KB (text-based)");
    println!("  CSV:     ~150 KB (text-based)");

    println!("\nRead/Write Speed (relative):");
    println!("  NPY:     ★★★★★ (fastest binary)");
    println!("  Parquet: ★★★★☆ (fast columnar)");
    println!("  HDF5:    ★★★★☆ (fast hierarchical)");
    println!("  JSON:    ★★★☆☆ (moderate text)");
    println!("  CSV:     ★★★☆☆ (moderate text)");

    // 7. Batch Export Example
    println!("\n7. BATCH EXPORT TO ALL FORMATS");
    println!("───────────────────────────────");

    println!("Exporting to all available formats...");

    // Standard exports (always available)
    std::fs::create_dir_all("output")?;
    std::fs::write("output/graph.json", graph.to_json(ExportOptions::default()))?;
    std::fs::write("output/graph_edges.csv", graph.to_edge_list_csv(true))?;
    std::fs::write("output/graph.dot", graph.to_dot())?;
    std::fs::write("output/graph.graphml", graph.to_graphml())?;

    println!("✓ Standard formats:");
    println!("  - graph.json");
    println!("  - graph_edges.csv");
    println!("  - graph.dot");
    println!("  - graph.graphml");

    #[cfg(feature = "npy-export")]
    {
        graph.to_npy("output/graph")?;
        println!("✓ NPY format");
    }

    #[cfg(feature = "parquet-export")]
    {
        graph.to_parquet("output/graph.parquet")?;
        println!("✓ Parquet format");
    }

    #[cfg(feature = "hdf5-export")]
    {
        graph.to_hdf5("output/graph.h5")?;
        println!("✓ HDF5 format");
    }

    println!("\nAll exports complete! Check the 'output/' directory.");

    // 8. Summary
    println!("\n8. SUMMARY");
    println!("──────────");

    let mut formats_available = vec!["JSON", "CSV", "DOT", "GraphML"];

    #[cfg(feature = "npy-export")]
    formats_available.push("NPY");

    #[cfg(feature = "parquet-export")]
    formats_available.push("Parquet");

    #[cfg(feature = "hdf5-export")]
    formats_available.push("HDF5");

    println!("✓ {} export formats available", formats_available.len());
    println!("✓ Formats: {}", formats_available.join(", "));
    println!("✓ Compatible with Python, Spark, MATLAB, R, and more");
    println!("✓ Optimized for different use cases");
    println!("✓ Production-ready for data science pipelines");

    Ok(())
}

