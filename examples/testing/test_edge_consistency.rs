use rustygraph::*;

fn main() {
    println!("Testing Sequential vs Optimized with SAME data");
    println!("{}", "=".repeat(60));

    // Simple test data
    let data = vec![1.0, 2.0, 1.5, 3.0, 2.5];
    println!("\nTest data: {:?}", data);

    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    println!("\nResults:");
    println!("  Nodes: {}", graph.node_count);
    println!("  Edges: {}", graph.edges().len());
    println!("\nEdge list:");
    let mut edges: Vec<_> = graph.edges().iter().collect();
    edges.sort_by_key(|(k, _)| *k);
    for ((src, dst), weight) in edges {
        println!("  {} -> {} (weight: {})", src, dst, weight);
    }

    #[cfg(feature = "simd")]
    println!("\n✅ SIMD enabled");
    #[cfg(not(feature = "simd"))]
    println!("\n❌ SIMD disabled");

    #[cfg(feature = "parallel")]
    println!("✅ Parallel enabled");
    #[cfg(not(feature = "parallel"))]
    println!("❌ Parallel disabled");
}

