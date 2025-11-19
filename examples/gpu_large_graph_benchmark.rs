//! GPU vs CPU Benchmark for Very Large Graphs
//!
//! This benchmark tests where GPU acceleration actually becomes beneficial
//! by testing graph sizes from 10,000 to 50,000 nodes.
//!
//! Run with: cargo run --example gpu_large_graph_benchmark --features metal --release

use rustygraph::TimeSeries;
use std::time::Instant;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
use rustygraph::performance::{GpuVisibilityGraph, GpuConfig, GpuCapabilities};

/// Generate test data for large graphs
fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            // Mix of patterns to create realistic visibility graph
            (x.sin() * 50.0) + (x * 0.5).cos() * 30.0 + (x * 0.1).sin() * 20.0 + 100.0
        })
        .collect()
}

fn main() {
    println!("🚀 GPU vs CPU Large Graph Benchmark\n");
    println!("{}", "=".repeat(80));

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
    {
        // Detect GPU capabilities
        let caps = GpuCapabilities::detect();
        println!("\n📊 Hardware Detection:");
        caps.print_info();

        if !caps.has_metal() {
            println!("\n⚠️  Metal GPU not available. This benchmark requires Apple Silicon.");
            return;
        }

        println!("\n{}", "=".repeat(80));
        println!("\n📈 Large Graph Performance Test (Natural Visibility)");
        println!("{}", "=".repeat(80));
        println!("{:<10} {:>15} {:>15} {:>12} {:>12}",
                 "Size", "GPU Time", "CPU Time", "Speedup", "Winner");
        println!("{}", "-".repeat(80));

        // Test very large graphs
        let test_sizes = vec![10_000, 15_000, 20_000, 30_000, 40_000, 50_000];

        for &size in &test_sizes {
            println!("\n🔄 Testing size: {} nodes...", size);

            // Generate data
            let data_f64 = generate_test_data(size);
            let series = TimeSeries::from_raw(data_f64.clone()).unwrap();

            // Warm up GPU (first run includes shader compilation)
            if size == test_sizes[0] {
                println!("   ⏳ Warming up GPU (compiling shaders)...");
                let config = GpuConfig::for_apple_silicon().with_min_nodes(0);
                let gpu = GpuVisibilityGraph::with_config(config);
                let _ = gpu.build_natural(&series);
                println!("   ✅ GPU warmed up");
            }

            // Test GPU
            print!("   🎮 GPU: Running... ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let config = GpuConfig::for_apple_silicon().with_min_nodes(0);
            let gpu = GpuVisibilityGraph::with_config(config);

            let start = Instant::now();
            let graph_gpu = gpu.build_natural(&series).unwrap();
            let gpu_time = start.elapsed();
            let gpu_edges = graph_gpu.edges().len();

            println!("{:?} ({} edges)", gpu_time, gpu_edges);

            // Test CPU (standard implementation)
            print!("   💻 CPU: Running... ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let start = Instant::now();
            let graph_cpu = rustygraph::VisibilityGraph::from_series(&series)
                .natural_visibility()
                .unwrap();
            let cpu_time = start.elapsed();
            let cpu_edges = graph_cpu.edges().len();

            println!("{:?} ({} edges)", cpu_time, cpu_edges);

            // Calculate speedup
            let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
            let winner = if speedup > 1.1 {
                "🏆 GPU"
            } else if speedup < 0.9 {
                "🏆 CPU"
            } else {
                "🤝 Tie"
            };

            // Check correctness
            let edge_diff = (gpu_edges as i32 - cpu_edges as i32).abs();
            let diff_pct = edge_diff as f64 / cpu_edges as f64 * 100.0;

            println!("   📊 Edge difference: {} ({:.2}%)", edge_diff, diff_pct);

            // Print summary row
            println!("{:<10} {:>15.2?} {:>15.2?} {:>11.2}x {:>12}",
                     size, gpu_time, cpu_time, speedup, winner);

            // Memory check
            if size >= 30_000 {
                println!("   💾 Memory: Large graph - {} nodes × {} nodes potential edges",
                         size, size);
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("\n📊 Analysis:");
        println!("{}", "-".repeat(80));

        println!("\n🔍 Key Observations:");
        println!("  • GPU overhead is significant for smaller graphs");
        println!("  • Break-even point is where GPU speedup > 1.0x");
        println!("  • GPU advantage grows with graph size (more parallel work)");
        println!("  • CPU optimizations (SIMD/parallel) are already excellent");

        println!("\n💡 Recommendations:");
        println!("  • Use CPU for graphs < break-even point");
        println!("  • Use GPU for graphs > break-even point");
        println!("  • Auto-selection (default) makes the right choice");

        println!("\n🎯 GPU Sweet Spot:");
        println!("  • Best performance: Graphs > 20,000 nodes");
        println!("  • Batch processing: Multiple large graphs");
        println!("  • Unified memory: Lower transfer overhead on Apple Silicon");

        println!("\n{}", "=".repeat(80));
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "metal")))]
    {
        println!("⚠️  This benchmark requires Apple Silicon and the 'metal' feature.");
        println!("   Build with: cargo run --example gpu_large_graph_benchmark --features metal --release");
    }
}

