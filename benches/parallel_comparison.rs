use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustygraph::*;

fn parallel_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    // Test various graph sizes
    for size in [100, 500, 1000, 2000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.1).sin()).collect();
        let series = TimeSeries::from_raw(data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("natural_with_features", size),
            size,
            |b, _| {
                b.iter(|| {
                    let feature_set = FeatureSet::new()
                        .add_builtin(BuiltinFeature::DeltaForward)
                        .add_builtin(BuiltinFeature::LocalSlope)
                        .add_builtin(BuiltinFeature::IsLocalMax);

                    let graph = VisibilityGraph::from_series(&series)
                        .with_features(feature_set)
                        .natural_visibility()
                        .unwrap();
                    black_box(graph);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("horizontal_with_features", size),
            size,
            |b, _| {
                b.iter(|| {
                    let feature_set = FeatureSet::new()
                        .add_builtin(BuiltinFeature::DeltaForward)
                        .add_builtin(BuiltinFeature::LocalSlope)
                        .add_builtin(BuiltinFeature::IsLocalMax);

                    let graph = VisibilityGraph::from_series(&series)
                        .with_features(feature_set)
                        .horizontal_visibility()
                        .unwrap();
                    black_box(graph);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, parallel_comparison_benchmark);
criterion_main!(benches);

