use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustygraph::*;

fn simd_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_comparison");

    // Test various graph sizes to see SIMD benefit
    for size in [100, 500, 1000, 2000, 5000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.1).sin()).collect();
        let series = TimeSeries::from_raw(data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("natural_visibility", size),
            size,
            |b, _| {
                b.iter(|| {
                    let graph = VisibilityGraph::from_series(&series)
                        .natural_visibility()
                        .unwrap();
                    black_box(graph);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("horizontal_visibility", size),
            size,
            |b, _| {
                b.iter(|| {
                    let graph = VisibilityGraph::from_series(&series)
                        .horizontal_visibility()
                        .unwrap();
                    black_box(graph);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, simd_comparison_benchmark);
criterion_main!(benches);

