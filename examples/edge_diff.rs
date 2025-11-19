use rustygraph::*;
use std::collections::HashSet;

fn main() {
    println!("Finding the difference in edges...");
    println!();

    let size = 50;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();

    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    let edges: HashSet<_> = graph.edges().keys().copied().collect();

    println!("This version found {} edges", edges.len());

    #[cfg(feature = "simd")]
    {
        println!("Compiled WITH SIMD");
        println!();
        println!("Edges that might be different from sequential:");
        // Show edges with distance > 8 (where SIMD kicks in)
        let simd_edges: Vec<_> = edges.iter()
            .filter(|(src, dst)| *dst - *src > 8)
            .take(10)
            .collect();
        println!("Sample long-distance edges (>8 apart): {:?}", simd_edges);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("Compiled WITHOUT SIMD");
        println!();
        // Show some long-distance edges
        let long_edges: Vec<_> = edges.iter()
            .filter(|(src, dst)| *dst - *src > 8)
            .take(10)
            .collect();
        println!("Sample long-distance edges (>8 apart): {:?}", long_edges);
    }

    println!();
    println!("Total edges by distance:");
    let mut dist_counts = std::collections::HashMap::new();
    for (src, dst) in edges.iter() {
        let dist = dst - src;
        *dist_counts.entry(dist).or_insert(0) += 1;
    }

    let mut distances: Vec<_> = dist_counts.keys().copied().collect();
    distances.sort();

    for dist in distances {
        let count = dist_counts[&dist];
        println!("  Distance {}: {} edges", dist, count);
    }
}

