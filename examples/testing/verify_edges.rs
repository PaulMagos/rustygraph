use rustygraph::*;

fn main() {
    println!("Testing which version is correct...");
    println!();

    // Use a small size where we can manually verify
    let size = 10;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();

    println!("Test data (size {}):", size);
    for (i, &v) in data.iter().enumerate() {
        println!("  {}: {:.4}", i, v);
    }
    println!();

    // Build graph
    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    let edges = graph.edges();
    println!("Found {} edges", edges.len());
    println!();

    // Get all edges sorted
    let mut edge_list: Vec<_> = edges.iter().map(|(k, v)| (*k, *v)).collect();
    edge_list.sort_by_key(|(k, _)| *k);

    // Manually verify a few edges
    println!("Verifying some edges manually:");
    println!();

    // Check if edge 0->2 exists
    let has_0_2 = edges.contains_key(&(0, 2));
    if has_0_2 {
        println!("Edge 0->2 exists. Checking if it should...");
        // y0 = 0.0, y1 = 0.1987, y2 = 0.3894
        // Line from 0 to 2: y = 0.0 + (0.3894 - 0.0) * (t/2)
        // At t=1: y = 0.1947
        // Actual y1 = 0.1987
        // Is 0.1987 < 0.1947? NO, so node 1 blocks!
        // Edge 0->2 should NOT exist!
        println!("  y0={:.4}, y1={:.4}, y2={:.4}", data[0], data[1], data[2]);
        let expected_at_1 = data[0] + (data[2] - data[0]) * 0.5;
        println!("  Expected height at 1: {:.4}", expected_at_1);
        println!("  Actual height at 1: {:.4}", data[1]);
        println!("  Is actual < expected? {}", data[1] < expected_at_1);
        if data[1] < expected_at_1 {
            println!("  ✅ Edge should exist (1 is below line)");
        } else {
            println!("  ❌ Edge should NOT exist (1 blocks the view)");
        }
    }
    println!();

    // List all edges
    println!("All edges:");
    for ((src, dst), _) in edge_list.iter().take(20) {
        println!("  {} -> {}", src, dst);
    }
    if edge_list.len() > 20 {
        println!("  ... and {} more", edge_list.len() - 20);
    }
}

