use rustygraph::*;

fn main() {
    // Test case: Can node 0 see node 10 with specific intermediate values?
    // This will help us determine which version is correct

    let data = vec![
        0.0,    // 0
        0.199,  // 1
        0.389,  // 2
        0.565,  // 3
        0.717,  // 4
        0.842,  // 5
        0.932,  // 6
        0.985,  // 7
        1.000,  // 8
        0.974,  // 9
        0.909,  // 10
    ];

    println!("Test data:");
    for (i, &v) in data.iter().enumerate() {
        println!("  {}: {:.3}", i, v);
    }
    println!();

    let series = TimeSeries::from_raw(data.clone()).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    // Check specific edge
    let has_0_10 = graph.edges().contains_key(&(0, 10));

    println!("Does edge 0->10 exist? {}", if has_0_10 { "YES" } else { "NO" });
    println!();

    // Manual calculation
    println!("Manual verification:");
    println!("  y0 = {:.3}, y10 = {:.3}", data[0], data[10]);
    println!("  Checking if all intermediate points are below the line...");

    let mut blocked = false;
    for k in 1..10 {
        let expected = data[0] + (data[10] - data[0]) * ((k as f64) / 10.0);
        let actual = data[k];
        let is_below = actual < expected;
        println!("    Point {}: actual={:.3}, expected={:.3}, below={}",
                 k, actual, expected, is_below);
        if !is_below {
            blocked = true;
        }
    }

    if blocked {
        println!("  Result: Edge should NOT exist (blocked)");
    } else {
        println!("  Result: Edge SHOULD exist (visible)");
    }

    println!();
    #[cfg(feature = "simd")]
    println!("Compiled WITH SIMD");
    #[cfg(not(feature = "simd"))]
    println!("Compiled WITHOUT SIMD");
}

