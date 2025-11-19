use rustygraph::*;

fn reference_natural_visibility(data: &[f64]) -> Vec<(usize, usize)> {
    // Reference implementation: Simple brute force O(n²)
    // This is the GROUND TRUTH - we know this is correct
    let n = data.len();
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            // Check if i can see j
            let mut visible = true;

            for k in (i + 1)..j {
                // Line from i to j: y = yi + (yj - yi) * (t - ti) / (tj - ti)
                let expected_height = data[i] + (data[j] - data[i]) * ((k - i) as f64 / (j - i) as f64);

                if data[k] >= expected_height {
                    visible = false;
                    break;
                }
            }

            if visible {
                edges.push((i, j));
            }
        }
    }

    edges
}

fn main() {
    println!("Ground Truth Comparison");
    println!("{}", "=".repeat(60));
    println!();

    let size = 50;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();

    // Ground truth (brute force)
    println!("Computing ground truth (brute force)...");
    let ground_truth = reference_natural_visibility(&data);
    println!("Ground truth: {} edges", ground_truth.len());
    println!();

    // Library implementation
    println!("Computing with library implementation...");
    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    let library_edges: Vec<_> = graph.edges()
        .keys()
        .map(|(i, j)| (*i, *j))
        .collect();

    println!("Library: {} edges", library_edges.len());
    println!();

    // Compare
    let ground_truth_set: std::collections::HashSet<_> = ground_truth.iter().collect();
    let library_set: std::collections::HashSet<_> = library_edges.iter().collect();

    let missing: Vec<_> = ground_truth_set.difference(&library_set).collect();
    let extra: Vec<_> = library_set.difference(&ground_truth_set).collect();

    println!("Comparison:");
    println!("  Missing in library: {}", missing.len());
    println!("  Extra in library: {}", extra.len());
    println!();

    if missing.is_empty() && extra.is_empty() {
        println!("✅ PERFECT MATCH!");
    } else {
        if !missing.is_empty() {
            println!("❌ Missing edges (should exist but don't):");
            for edge in missing.iter().take(10) {
                println!("  {} -> {}", edge.0, edge.1);
            }
            if missing.len() > 10 {
                println!("  ... and {} more", missing.len() - 10);
            }
            println!();
        }

        if !extra.is_empty() {
            println!("❌ Extra edges (exist but shouldn't):");
            for edge in extra.iter().take(10) {
                println!("  {} -> {}", edge.0, edge.1);
            }
            if extra.len() > 10 {
                println!("  ... and {} more", extra.len() - 10);
            }
            println!();
        }
    }

    #[cfg(feature = "simd")]
    println!("Tested version: WITH SIMD");
    #[cfg(not(feature = "simd"))]
    println!("Tested version: WITHOUT SIMD (sequential)");
}

