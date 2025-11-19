// Rust Performance Benchmark: Sequential vs Parallel+SIMD
//
// This benchmark creates two versions:
// 1. Sequential (no-default-features)
// 2. Optimized (with parallel and simd)

use rustygraph::*;
use std::time::Instant;

fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (i as f64 * 0.2).sin())
        .collect()
}

fn benchmark_visibility(data: &[f64], name: &str) -> (u128, usize) {
    let series = TimeSeries::from_raw(data.to_vec()).unwrap();

    let start = Instant::now();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();
    let elapsed = start.elapsed().as_micros();

    println!("{}: {}μs ({} edges)", name, elapsed, graph.edges().len());

    (elapsed, graph.edges().len())
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        RustyGraph Rust Performance Benchmark              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Check which features are enabled
    println!("Configuration:");
    #[cfg(feature = "parallel")]
    println!("  ✅ Parallel processing ENABLED");
    #[cfg(not(feature = "parallel"))]
    println!("  ❌ Parallel processing DISABLED");

    #[cfg(feature = "simd")]
    println!("  ✅ SIMD optimizations ENABLED");
    #[cfg(not(feature = "simd"))]
    println!("  ❌ SIMD optimizations DISABLED");

    println!();
    println!("Running benchmarks...");
    println!();

    let sizes = vec![50, 100, 200, 500, 1000, 2000];

    for size in sizes {
        println!("────────────────────────────────────────────────────────────");
        println!("Test size: {} nodes", size);
        println!("────────────────────────────────────────────────────────────");

        let data = generate_test_data(size);

        // Warmup
        let _ = benchmark_visibility(&data, "Warmup");

        // Actual benchmark (run 3 times, take average)
        let mut times = Vec::new();
        for i in 1..=3 {
            let (time, _) = benchmark_visibility(&data, &format!("Run {}", i));
            times.push(time);
        }

        let avg_time = times.iter().sum::<u128>() / times.len() as u128;
        println!("Average: {}μs ({:.2}ms)", avg_time, avg_time as f64 / 1000.0);
        println!();
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Complete                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("To compare sequential vs optimized:");
    println!("  1. Sequential: cargo run --example rust_benchmark --no-default-features");
    println!("  2. Optimized:  cargo run --example rust_benchmark --features parallel,simd --release");
}

