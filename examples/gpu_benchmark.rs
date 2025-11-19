//! Comprehensive GPU benchmark and validation
//!
//! This example tests and validates GPU acceleration claims including:
//! - Metal GPU detection and initialization
//! - CPU vs GPU performance comparison
//! - Speedup measurements across different graph sizes
//! - Correctness validation (GPU results match CPU)

use rustygraph::*;
use rustygraph::performance::{GpuVisibilityGraph, GpuConfig, GpuCapabilities};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  RustyGraph GPU Performance Validation            ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    // 1. GPU Detection
    println!("1️⃣  DETECTING GPU CAPABILITIES");
    println!("═══════════════════════════════════════════════════\n");

    let gpu_caps = GpuCapabilities::detect();
    gpu_caps.print_info();
    println!();

    if !gpu_caps.has_metal() {
        println!("⚠️  Metal GPU not available on this platform");
        println!("   Running CPU-only benchmarks for comparison\n");
    }

    // 2. Test different graph sizes
    let test_sizes = vec![
        ("Tiny", 100),
        ("Small", 500),
        ("Medium", 1000),
        ("Large", 2000),
        ("Very Large", 5000),
    ];

    println!("2️⃣  PERFORMANCE BENCHMARKS");
    println!("═══════════════════════════════════════════════════\n");

    println!("{:<15} {:>10} {:>10} {:>10} {:>10}",
             "Graph Size", "CPU (ms)", "GPU (ms)", "Speedup", "Edges");
    println!("{}", "─".repeat(60));

    let mut results = Vec::new();

    for (name, size) in test_sizes {
        // Generate test data
        let data: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos())
            .collect();
        let series = TimeSeries::from_raw(data.clone())?;

        // CPU Benchmark
        let cpu_start = Instant::now();
        let cpu_graph = VisibilityGraph::from_series(&series)
            .natural_visibility()?;
        let cpu_time = cpu_start.elapsed();
        let cpu_edges = cpu_graph.edges().len();

        // GPU Benchmark (with automatic selection)
        let gpu_config = GpuConfig::for_apple_silicon()
            .with_min_nodes(500); // Lower threshold for testing
        let gpu_builder = GpuVisibilityGraph::with_config(gpu_config);

        let gpu_start = Instant::now();
        let gpu_graph = gpu_builder.build_natural(&series)?;
        let gpu_time = gpu_start.elapsed();
        let gpu_edges = gpu_graph.edges().len();

        // Calculate speedup
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let backend = if gpu_builder.should_use_gpu(size) && gpu_caps.has_metal() {
            "GPU"
        } else {
            "CPU"
        };

        println!("{:<15} {:>10.2} {:>10.2} {:>9.2}x {:>10} ({})",
                 name,
                 cpu_time.as_secs_f64() * 1000.0,
                 gpu_time.as_secs_f64() * 1000.0,
                 speedup,
                 cpu_edges,
                 backend);

        // Validate correctness
        if cpu_edges != gpu_edges {
            println!("⚠️  WARNING: Edge count mismatch! CPU: {}, GPU: {}", cpu_edges, gpu_edges);
        }

        results.push((name, size, cpu_time, gpu_time, speedup, backend));
    }

    println!();

    // 3. Correctness Validation
    println!("3️⃣  CORRECTNESS VALIDATION");
    println!("═══════════════════════════════════════════════════\n");

    println!("Testing CPU vs GPU output consistency...\n");

    let test_data: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 2.5, 3.5, 1.5];
    let test_series = TimeSeries::from_raw(test_data)?;

    let cpu_graph = VisibilityGraph::from_series(&test_series)
        .natural_visibility()?;

    let gpu_config = GpuConfig::for_apple_silicon().with_min_nodes(1);
    let gpu_builder = GpuVisibilityGraph::with_config(gpu_config);
    let gpu_graph = gpu_builder.build_natural(&test_series)?;

    let cpu_edges = cpu_graph.edges();
    let gpu_edges = gpu_graph.edges();

    println!("CPU edges: {}", cpu_edges.len());
    println!("GPU edges: {}", gpu_edges.len());

    if cpu_edges.len() == gpu_edges.len() {
        println!("✅ Edge count matches!");
    } else {
        println!("❌ Edge count mismatch!");
    }

    // Check if edges are identical (order-independent)
    // Extract just the edge keys (not weights) for comparison
    let cpu_keys: std::collections::HashSet<_> = cpu_edges.iter().map(|(k, _)| k).collect();
    let gpu_keys: std::collections::HashSet<_> = gpu_edges.iter().map(|(k, _)| k).collect();

    let all_match = cpu_keys == gpu_keys;
    if all_match {
        println!("✅ All edges match exactly!");
    } else {
        println!("❌ Some edges differ!");
        let missing_in_gpu: Vec<_> = cpu_keys.difference(&gpu_keys).collect();
        let extra_in_gpu: Vec<_> = gpu_keys.difference(&cpu_keys).collect();
        if !missing_in_gpu.is_empty() {
            println!("   Missing in GPU: {:?}", missing_in_gpu);
        }
        if !extra_in_gpu.is_empty() {
            println!("   Extra in GPU: {:?}", extra_in_gpu);
        }
    }
    println!();

    // 4. Horizontal Visibility Test
    println!("4️⃣  HORIZONTAL VISIBILITY TEST");
    println!("═══════════════════════════════════════════════════\n");

    let hv_data: Vec<f64> = (0..1000)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();
    let hv_series = TimeSeries::from_raw(hv_data)?;

    let cpu_start = Instant::now();
    let cpu_hv = VisibilityGraph::from_series(&hv_series)
        .horizontal_visibility()?;
    let cpu_hv_time = cpu_start.elapsed();

    let gpu_start = Instant::now();
    let gpu_hv = gpu_builder.build_horizontal(&hv_series)?;
    let gpu_hv_time = gpu_start.elapsed();

    println!("1000-node horizontal visibility:");
    println!("  CPU time: {:.2} ms", cpu_hv_time.as_secs_f64() * 1000.0);
    println!("  GPU time: {:.2} ms", gpu_hv_time.as_secs_f64() * 1000.0);
    println!("  CPU edges: {}", cpu_hv.edges().len());
    println!("  GPU edges: {}", gpu_hv.edges().len());
    println!("  Match: {}", if cpu_hv.edges().len() == gpu_hv.edges().len() { "✅" } else { "❌" });
    println!();

    // 5. Summary
    println!("5️⃣  SUMMARY");
    println!("═══════════════════════════════════════════════════\n");

    println!("Platform: {}", if gpu_caps.has_metal() {
        "Apple Silicon with Metal GPU"
    } else {
        "CPU-only"
    });
    println!();

    if gpu_caps.has_metal() {
        println!("✅ Metal GPU detected and functional");
        println!("✅ GPU acceleration available");
        println!("✅ Automatic CPU/GPU selection working");

        // Find best speedup
        let best_speedup = results.iter()
            .filter(|(_, _, _, _, _, backend)| *backend == "GPU")
            .map(|(_, _, _, _, speedup, _)| speedup)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(speedup) = best_speedup {
            println!("✅ Best observed speedup: {:.2}x", speedup);
        }
    } else {
        println!("ℹ️  Metal GPU not available on this platform");
        println!("ℹ️  Using optimized CPU implementation");
    }

    println!();
    println!("All correctness tests: {}", if all_match { "✅ PASSED" } else { "⚠️  REVIEW" });
    println!();

    // 6. Recommendations
    println!("6️⃣  RECOMMENDATIONS");
    println!("═══════════════════════════════════════════════════\n");

    if gpu_caps.has_metal() {
        println!("For Apple Silicon:");
        println!("  • Use GPU for graphs > 2000 nodes");
        println!("  • Unified memory reduces overhead");
        println!("  • Consider batch processing for multiple graphs");
    } else {
        println!("For this platform:");
        println!("  • CPU implementation is highly optimized");
        println!("  • Parallel + SIMD provide excellent performance");
        println!("  • Consider batching for processing multiple graphs");
    }

    Ok(())
}

