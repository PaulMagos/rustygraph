use rustygraph::*;

fn main() {
    println!("Testing Parallel O(n) Implementation");
    println!("{}", "=".repeat(60));
    println!();

    let sizes = vec![100, 500, 1000, 2000, 5000, 10000];

    for size in sizes {
        println!("Testing size: {} nodes", size);

        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.2).sin()).collect();
        let series = TimeSeries::from_raw(data).unwrap();

        // Sequential (O(n) envelope)
        let start = std::time::Instant::now();
        let edges_seq = algorithms::natural_visibility(&series);
        let time_seq = start.elapsed();

        // Parallel (O(n) envelope per chunk)
        let start = std::time::Instant::now();
        let edges_par = algorithms::natural_visibility_parallel(&series);
        let time_par = start.elapsed();

        // Compare
        let seq_count = edges_seq.len();
        let par_count = edges_par.len();

        println!("  Sequential: {} edges in {:?}", seq_count, time_seq);
        println!("  Parallel:   {} edges in {:?}", par_count, time_par);

        if seq_count == par_count {
            println!("  ✅ MATCH!");
            let speedup = time_seq.as_secs_f64() / time_par.as_secs_f64();
            println!("  Speedup: {:.2}x", speedup);
        } else {
            println!("  ❌ MISMATCH! Difference: {}", (seq_count as i64 - par_count as i64).abs());
        }

        println!();
    }
}

