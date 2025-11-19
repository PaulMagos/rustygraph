//! GPU vs CPU Performance Comparison with Fair Precision Matching
//!
//! This example demonstrates a fair comparison between GPU and CPU implementations
//! by using the same float32 precision on both, eliminating precision differences
//! from the performance analysis.
//!
//! Compares:
//! 1. GPU (f32) - Metal GPU using float32
//! 2. CPU (f32) - CPU implementation forced to use float32
//! 3. CPU (f64) - Standard CPU implementation with float64
//!
//! Run with: cargo run --example gpu_precision_comparison --features metal

use rustygraph::TimeSeries;
use std::time::Instant;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
use rustygraph::performance::{GpuVisibilityGraph, GpuConfig, GpuCapabilities};

/// CPU implementation using f32 precision (matching GPU)
fn build_natural_cpu_f32(data: &[f32]) -> Vec<(usize, usize)> {
    let n = data.len();
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if is_visible_natural_f32(data, i, j) {
                edges.push((i, j));
            }
        }
    }

    edges
}

/// Natural visibility check using f32 precision (matching GPU)
fn is_visible_natural_f32(data: &[f32], i: usize, j: usize) -> bool {
    let vi = data[i];
    let vj = data[j];

    for k in (i + 1)..j {
        let vk = data[k];
        let line_height = vi + (vj - vi) * ((k - i) as f32 / (j - i) as f32);
        if vk >= line_height {
            return false;
        }
    }

    true
}

/// CPU implementation using f64 precision (standard)
fn build_natural_cpu_f64(data: &[f64]) -> Vec<(usize, usize)> {
    let n = data.len();
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if is_visible_natural_f64(data, i, j) {
                edges.push((i, j));
            }
        }
    }

    edges
}

/// Natural visibility check using f64 precision (standard)
fn is_visible_natural_f64(data: &[f64], i: usize, j: usize) -> bool {
    let vi = data[i];
    let vj = data[j];

    for k in (i + 1)..j {
        let vk = data[k];
        let line_height = vi + (vj - vi) * ((k - i) as f64 / (j - i) as f64);
        if vk >= line_height {
            return false;
        }
    }

    true
}

fn main() {
    println!("🔬 GPU vs CPU Precision-Matched Performance Comparison\n");
    println!("{}", "=".repeat(70));

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
    {
        // Detect GPU capabilities
        let caps = GpuCapabilities::detect();
        println!("\n📊 Hardware Detection:");
        caps.print_info();
        println!();

        if !caps.has_metal() {
            println!("⚠️  Metal GPU not available. This comparison requires Apple Silicon.");
            return;
        }

        // Test different graph sizes
        let test_sizes = vec![10, 50, 100, 200, 500, 1000, 2000, 5000];

        println!("\n📈 Performance Comparison (Natural Visibility)");
        println!("{}", "=".repeat(70));
        println!("{:<10} {:>12} {:>12} {:>12} {:>12}",
                 "Size", "GPU (f32)", "CPU (f32)", "CPU (f64)", "Match");
        println!("{}", "-".repeat(70));

        for &size in &test_sizes {
            // Generate test data
            let data_f64: Vec<f64> = (0..size)
                .map(|i| (i as f64 * 0.1).sin() * 100.0 + 100.0)
                .collect();
            let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();

            // Warm up
            let _ = build_natural_cpu_f32(&data_f32);

            // Test GPU (f32)
            let gpu_time = if size >= 100 {  // GPU has overhead for small sizes
                let config = GpuConfig::for_apple_silicon().with_min_nodes(0);
                let gpu = GpuVisibilityGraph::with_config(config);
                let series = TimeSeries::from_raw(data_f64.clone()).unwrap();

                let start = Instant::now();
                let graph = gpu.build_natural(&series).unwrap();
                let elapsed = start.elapsed();

                Some((elapsed, graph.edges().len()))
            } else {
                None
            };

            // Test CPU (f32)
            let start = Instant::now();
            let edges_f32 = build_natural_cpu_f32(&data_f32);
            let cpu_f32_time = start.elapsed();
            let cpu_f32_edges = edges_f32.len();

            // Test CPU (f64)
            let start = Instant::now();
            let edges_f64 = build_natural_cpu_f64(&data_f64);
            let cpu_f64_time = start.elapsed();
            let cpu_f64_edges = edges_f64.len();

            // Compare results
            let match_status = if let Some((_, gpu_edges)) = gpu_time {
                let f32_match = (gpu_edges as i32 - cpu_f32_edges as i32).abs();
                let _f64_diff = (gpu_edges as i32 - cpu_f64_edges as i32).abs();

                if f32_match == 0 {
                    "✅ Perfect"
                } else if f32_match < size / 20 {
                    "✓ Close"
                } else {
                    "⚠️  Diff"
                }
            } else {
                "N/A"
            };

            // Print results
            if let Some((gpu_elapsed, _)) = gpu_time {
                println!("{:<10} {:>11.2?} {:>11.2?} {:>11.2?} {:>12}",
                         size, gpu_elapsed, cpu_f32_time, cpu_f64_time, match_status);
            } else {
                println!("{:<10} {:>12} {:>11.2?} {:>11.2?} {:>12}",
                         size, "N/A", cpu_f32_time, cpu_f64_time, "N/A");
            }
        }

        println!("\n{}", "=".repeat(70));

        // Detailed comparison for one size
        println!("\n🔍 Detailed Analysis (size=500):");
        println!("{}", "=".repeat(70));

        let size = 500;
        let data_f64: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1).sin() * 100.0 + 100.0)
            .collect();
        let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();

        // GPU (f32)
        let config = GpuConfig::for_apple_silicon().with_min_nodes(0);
        let gpu = GpuVisibilityGraph::with_config(config);
        let series = TimeSeries::from_raw(data_f64.clone()).unwrap();
        let start = Instant::now();
        let graph_gpu = gpu.build_natural(&series).unwrap();
        let gpu_time = start.elapsed();

        // CPU (f32)
        let start = Instant::now();
        let edges_f32 = build_natural_cpu_f32(&data_f32);
        let cpu_f32_time = start.elapsed();

        // CPU (f64)
        let start = Instant::now();
        let edges_f64 = build_natural_cpu_f64(&data_f64);
        let cpu_f64_time = start.elapsed();

        let gpu_edge_count = graph_gpu.edges().len();

        println!("\nEdge Counts:");
        println!("  GPU (f32):  {} edges", gpu_edge_count);
        println!("  CPU (f32):  {} edges", edges_f32.len());
        println!("  CPU (f64):  {} edges", edges_f64.len());

        println!("\nPrecision Differences:");
        let gpu_vs_f32 = (gpu_edge_count as i32 - edges_f32.len() as i32).abs();
        let gpu_vs_f64 = (gpu_edge_count as i32 - edges_f64.len() as i32).abs();
        let f32_vs_f64 = (edges_f32.len() as i32 - edges_f64.len() as i32).abs();

        println!("  GPU vs CPU(f32): {} edges difference ({:.2}%)",
                 gpu_vs_f32,
                 gpu_vs_f32 as f64 / edges_f32.len() as f64 * 100.0);
        println!("  GPU vs CPU(f64): {} edges difference ({:.2}%)",
                 gpu_vs_f64,
                 gpu_vs_f64 as f64 / edges_f64.len() as f64 * 100.0);
        println!("  CPU(f32) vs CPU(f64): {} edges difference ({:.2}%)",
                 f32_vs_f64,
                 f32_vs_f64 as f64 / edges_f64.len() as f64 * 100.0);

        println!("\nPerformance (Apples-to-Apples f32 comparison):");
        let speedup = cpu_f32_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("  GPU (f32):  {:>8.2?}", gpu_time);
        println!("  CPU (f32):  {:>8.2?}  (baseline)", cpu_f32_time);
        println!("  CPU (f64):  {:>8.2?}", cpu_f64_time);
        println!("\n  GPU Speedup vs CPU(f32): {:.2}x {}",
                 speedup,
                 if speedup > 1.0 { "🚀 GPU FASTER" }
                 else { "⚠️  CPU FASTER" });

        println!("\n💡 Key Findings:");
        println!("  • GPU uses float32 due to Metal limitations");
        println!("  • CPU(f32) provides fair performance comparison");
        println!("  • CPU(f64) shows precision impact on edge detection");
        println!("  • Small precision differences are expected and acceptable");

        println!("\n✅ Conclusion:");
        if speedup > 1.2 {
            println!("  GPU provides {:.1}x speedup over CPU when using same precision!", speedup);
        } else if speedup > 0.8 {
            println!("  GPU and CPU have comparable performance at this size.");
            println!("  GPU benefits emerge with larger graphs (>5000 nodes).");
        } else {
            println!("  CPU is faster at this size due to GPU overhead.");
            println!("  GPU excels with larger graphs where parallelism helps.");
        }
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "metal")))]
    {
        println!("⚠️  This example requires Apple Silicon and the 'metal' feature.");
        println!("   Build with: cargo run --example gpu_precision_comparison --features metal");
    }
}

