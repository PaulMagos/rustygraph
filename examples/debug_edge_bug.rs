use rustygraph::*;

fn main() {
    println!("Debugging edge computation");
    println!("{}", "=".repeat(60));

    // Manually check if 2 can see 3
    let data = vec![1.0, 2.0, 1.5, 3.0, 2.5];
    println!("\nData: {:?}", data);
    println!("\nManual check: Can node 2 (value 1.5) see node 3 (value 3.0)?");
    println!("  Distance: 3 - 2 = 1 (adjacent)");
    println!("  Intermediate points: none");
    println!("  Answer: YES (adjacent nodes always visible)");

    println!("\nNow let's see what the algorithm finds:");

    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    println!("  Total edges: {}", graph.edges().len());

    let has_2_to_3 = graph.edges().contains_key(&(2, 3));
    println!("  Has edge 2->3? {}", if has_2_to_3 { "YES ✅" } else { "NO ❌" });

    if !has_2_to_3 {
        println!("\n🐛 BUG: Algorithm missing edge 2->3!");
        println!("   This is likely due to the envelope optimization incorrectly");
        println!("   removing node 2 from the stack before node 3 is processed.");
    }

    println!("\nAll edges:");
    let mut edges: Vec<_> = graph.edges().iter().collect();
    edges.sort_by_key(|(k, _)| *k);
    for ((src, dst), _) in edges {
        println!("  {} -> {}", src, dst);
    }
}

