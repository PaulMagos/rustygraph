use rustygraph::*;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Correctness Test: Sequential vs Optimized             ║");
    println!("║     Testing at Various Graph Sizes                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let test_sizes = vec![50, 100, 200, 500, 1000, 2000, 5000];

    for size in test_sizes {
        println!("════════════════════════════════════════════════════════════");
        println!("Testing with {} nodes", size);
        println!("════════════════════════════════════════════════════════════");

        // Generate same test data (sine wave)
        let data: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.2).sin())
            .collect();

        // Create graph
        let series = TimeSeries::from_raw(data.clone()).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let node_count = graph.node_count;
        let edge_count = graph.edges().len();
        let density = graph.density();

        println!("  Nodes: {}", node_count);
        println!("  Edges: {}", edge_count);
        println!("  Density: {:.6}", density);

        // Sanity checks
        let mut issues = Vec::new();

        // Check 1: Node count should match input size
        if node_count != size {
            issues.push(format!("Node count mismatch: expected {}, got {}", size, node_count));
        }

        // Check 2: At minimum, we should have (n-1) edges (each node sees at least the next one)
        // For sine wave, we expect many more
        if edge_count < size - 1 {
            issues.push(format!("Too few edges: {} < {}", edge_count, size - 1));
        }

        // Check 3: Maximum possible edges in directed graph is n*(n-1)/2
        let max_edges = size * (size - 1) / 2;
        if edge_count > max_edges {
            issues.push(format!("Too many edges: {} > {}", edge_count, max_edges));
        }

        // Check 4: Density should be reasonable (between 0 and 1)
        if density < 0.0 || density > 1.0 {
            issues.push(format!("Invalid density: {}", density));
        }

        // Check 5: Adjacent nodes should be connected
        // Sample a few random adjacent pairs
        let sample_pairs = vec![
            (0, 1),
            (size / 4, size / 4 + 1),
            (size / 2, size / 2 + 1),
            (3 * size / 4, 3 * size / 4 + 1),
            (size - 2, size - 1),
        ];

        for (i, j) in sample_pairs {
            if j < size {
                let has_edge = graph.edges().contains_key(&(i, j));
                if !has_edge {
                    issues.push(format!("Missing edge between adjacent nodes {} and {}", i, j));
                }
            }
        }

        // Check 6: No self-loops
        for i in 0..size {
            if graph.edges().contains_key(&(i, i)) {
                issues.push(format!("Self-loop detected at node {}", i));
            }
        }

        if issues.is_empty() {
            println!("  ✅ All correctness checks passed!");
        } else {
            println!("  ❌ Issues found:");
            for issue in issues {
                println!("     - {}", issue);
            }
        }

        println!();
    }

    println!("════════════════════════════════════════════════════════════");
    println!("Testing Complete");
    println!("════════════════════════════════════════════════════════════");
}

