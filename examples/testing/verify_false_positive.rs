use rustygraph::*;

fn main() {
    // Verify edge 26 -> 35 manually
    let size = 50;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();

    let i = 26;
    let j = 35;

    println!("Manual verification of edge {} -> {}", i, j);
    println!("Distance: {}", j - i);
    println!();

    println!("Values:");
    println!("  y[{}] = {:.6}", i, data[i]);
    println!("  y[{}] = {:.6}", j, data[j]);
    println!();

    println!("Checking intermediate points:");
    let mut blocked = false;
    let mut blocker = None;

    for k in (i + 1)..j {
        let expected = data[i] + (data[j] - data[i]) * ((k - i) as f64 / (j - i) as f64);
        let actual = data[k];
        let blocks = actual >= expected;

        println!("  k={}: actual={:.6}, expected={:.6}, blocks={}", k, actual, expected, blocks);

        if blocks && !blocked {
            blocked = true;
            blocker = Some(k);
        }
    }

    println!();
    if blocked {
        println!("❌ Edge {} -> {} should NOT exist (blocked by node {})", i, j, blocker.unwrap());
        println!("   The SIMD implementation incorrectly added this edge!");
    } else {
        println!("✅ Edge {} -> {} should exist (visible)", i, j);
    }
}

