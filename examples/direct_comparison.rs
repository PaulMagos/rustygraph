use rustygraph::*;
use std::collections::HashSet;

fn main() {
    println!("Direct Comparison: Same Data, Different Compilation");
    println!("{}", "=".repeat(60));
    println!();

    let sizes = vec![50, 100, 200];

    for size in sizes {
        println!("Testing size: {}", size);

        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();

        let series = TimeSeries::from_raw(data).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let edges: HashSet<_> = graph.edges().keys().copied().collect();

        println!("  Total edges: {}", edges.len());

        // Check for specific patterns that might indicate bugs
        let mut adjacent_missing = 0;
        for i in 0..(size - 1) {
            if !edges.contains(&(i, i + 1)) {
                adjacent_missing += 1;
            }
        }

        if adjacent_missing > 0 {
            println!("  ❌ Missing {} adjacent edges!", adjacent_missing);
        } else {
            println!("  ✅ All adjacent edges present");
        }

        // Sample some non-adjacent edges
        let non_adjacent: Vec<_> = edges.iter()
            .filter(|(src, dst)| *dst > *src + 1)
            .take(5)
            .collect();

        if !non_adjacent.is_empty() {
            println!("  Sample non-adjacent edges: {:?}", non_adjacent);
        }

        println!();
    }

    #[cfg(feature = "simd")]
    println!("This binary was compiled WITH SIMD");
    #[cfg(not(feature = "simd"))]
    println!("This binary was compiled WITHOUT SIMD");

    #[cfg(feature = "parallel")]
    println!("This binary was compiled WITH parallel");
    #[cfg(not(feature = "parallel"))]
    println!("This binary was compiled WITHOUT parallel");
}

