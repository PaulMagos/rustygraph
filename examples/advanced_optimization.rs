//! Advanced features example.
//!
//! This example demonstrates:
//! - Lazy evaluation for performance
//! - Wavelet-based multi-scale analysis
//! - Advanced complexity metrics
//! - Frequency domain features (with advanced-features flag)

use rustygraph::*;
use rustygraph::advanced::{WaveletFeatures, AdvancedFeatures};
use rustygraph::lazy::LazyFeatureBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph Advanced Features Example ===\n");

    // 1. Lazy Evaluation
    println!("1. LAZY EVALUATION");
    println!("──────────────────");

    let series = TimeSeries::from_raw(datasets::sine_wave(100, 2.0, 1.0))?;
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?
        .with_lazy_metrics(); // Enable lazy evaluation

    println!("Graph created with lazy metrics");
    println!("First access computes and caches...");
    let start = std::time::Instant::now();
    let cc1 = graph.average_clustering_coefficient();
    let time1 = start.elapsed();
    println!("  Clustering: {:.4} (computed in {:?})", cc1, time1);

    println!("Second access uses cache...");
    let start = std::time::Instant::now();
    let cc2 = graph.average_clustering_coefficient();
    let time2 = start.elapsed();
    println!("  Clustering: {:.4} (cached in {:?})", cc2, time2);
    println!("  Speedup: {:.2}x", time1.as_nanos() as f64 / time2.as_nanos() as f64);

    // 2. Wavelet Analysis
    println!("\n2. WAVELET MULTI-SCALE ANALYSIS");
    println!("────────────────────────────────");

    let data = datasets::multi_frequency(128, &[1.0, 4.0, 8.0]);
    println!("Multi-frequency signal: [1Hz, 4Hz, 8Hz]");

    // Haar wavelet decomposition
    let (approx, detail) = WaveletFeatures::haar_transform(&data);
    println!("\nHaar transform:");
    println!("  Approximation coefficients: {}", approx.len());
    println!("  Detail coefficients: {}", detail.len());

    // Multi-level decomposition
    let details = WaveletFeatures::multi_level_decomposition(&data, 3);
    println!("\nMulti-level decomposition (3 levels):");
    for (level, coeffs) in details.iter().enumerate() {
        let energy = coeffs.iter().map(|&x| x * x).sum::<f64>();
        println!("  Level {}: {} coeffs, energy = {:.2}", level + 1, coeffs.len(), energy);
    }

    // Energy at each scale
    println!("\nEnergy by scale:");
    for level in 0..3 {
        let energy = WaveletFeatures::wavelet_energy(&data, level);
        println!("  Scale {}: {:.2}", level + 1, energy);
    }

    // 3. Advanced Complexity Metrics
    println!("\n3. COMPLEXITY METRICS");
    println!("─────────────────────");

    let patterns = vec![
        ("Random Walk", datasets::random_walk(100, 42)),
        ("Chaotic (Logistic)", datasets::logistic_map(100, 3.9, 0.5)),
        ("Periodic (Sine)", datasets::sine_wave(100, 2.0, 1.0)),
        ("Trend", datasets::trend_with_noise(100, 0.1, 0.1, 42)),
    ];

    for (name, data) in &patterns {
        // Sample entropy (complexity)
        let max_val = data.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let entropy = AdvancedFeatures::sample_entropy(data, 2, 0.2 * max_val);

        // Hurst exponent (long-range dependence)
        let hurst = AdvancedFeatures::hurst_exponent(data);

        println!("{}", name);
        println!("  Sample Entropy: {:.4} (higher = more complex)", entropy);
        println!("  Hurst Exponent: {:.4} (0.5=random, >0.5=persistent, <0.5=anti-persistent)", hurst);

        // Interpret Hurst
        let hurst_interp = if (hurst - 0.5).abs() < 0.1 {
            "random/Brownian"
        } else if hurst > 0.5 {
            "persistent/trending"
        } else {
            "anti-persistent/mean-reverting"
        };
        println!("  → {}", hurst_interp);
        println!();
    }

    // 4. Combine with Visibility Graphs
    println!("4. ADVANCED FEATURES + VISIBILITY GRAPHS");
    println!("─────────────────────────────────────────");

    for (name, data) in patterns.iter().take(2) {
        let series = TimeSeries::from_raw(data.clone())?;
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()?;

        // Graph metrics
        let density = graph.density();
        let clustering = graph.average_clustering_coefficient();

        // Complexity metrics
        let entropy = AdvancedFeatures::sample_entropy(data, 2, 0.2);
        let hurst = AdvancedFeatures::hurst_exponent(data);

        println!("{}", name);
        println!("  Graph density: {:.4}", density);
        println!("  Clustering: {:.4}", clustering);
        println!("  Entropy: {:.4}", entropy);
        println!("  Hurst: {:.4}", hurst);
        println!("  → Complexity-density correlation: {:.4}",
            (entropy * density).sqrt());
        println!();
    }

    // 5. Lazy Feature Computation
    println!("5. LAZY FEATURE COMPUTATION");
    println!("───────────────────────────");

    let series_data = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
    let lazy_builder = LazyFeatureBuilder::new(series_data);

    println!("Computing features on-demand:");

    // Access features - computed on first access
    let _f1 = lazy_builder.get_feature(2, "delta", |s, i| {
        println!("  Computing delta for node {}...", i);
        if i > 0 && i < s.len() {
            match (s[i], s[i - 1]) {
                (Some(curr), Some(prev)) => Some(curr - prev),
                _ => None,
            }
        } else {
            None
        }
    });

    // Second access uses cache
    let _f2 = lazy_builder.get_feature(2, "delta", |s, i| {
        println!("  Computing delta for node {} (should not see this)", i);
        if i > 0 && i < s.len() {
            match (s[i], s[i - 1]) {
                (Some(curr), Some(prev)) => Some(curr - prev),
                _ => None,
            }
        } else {
            None
        }
    });

    println!("Cache size: {} features", lazy_builder.cache_size());

    #[cfg(feature = "advanced-features")]
    {
        println!("\n6. FREQUENCY DOMAIN FEATURES");
        println!("────────────────────────────");
        println!("(Requires 'advanced-features' flag)");

        use rustygraph::advanced::FrequencyFeatures;

        let data = datasets::multi_frequency(64, &[2.0, 5.0]);

        let dominant = FrequencyFeatures::dominant_frequency(&data, 32, 64);
        println!("Dominant frequency index: {}", dominant);

        let energy_low = FrequencyFeatures::spectral_energy(&data, 32, 64, 1, 10);
        let energy_high = FrequencyFeatures::spectral_energy(&data, 32, 64, 10, 20);

        println!("Low frequency energy: {:.2}", energy_low);
        println!("High frequency energy: {:.2}", energy_high);
    }

    #[cfg(not(feature = "advanced-features"))]
    {
        println!("\n6. FREQUENCY DOMAIN FEATURES");
        println!("────────────────────────────");
        println!("Not available (requires 'advanced-features' cargo flag)");
        println!("Run with: cargo run --example advanced_optimization --features advanced-features");
    }

    println!("\n7. SUMMARY");
    println!("──────────");
    println!("✓ Lazy evaluation provides caching speedup");
    println!("✓ Wavelet analysis enables multi-scale patterns");
    println!("✓ Complexity metrics (entropy, Hurst) characterize dynamics");
    println!("✓ Advanced features complement visibility graph analysis");
    println!("✓ On-demand computation reduces memory usage");

    Ok(())
}

