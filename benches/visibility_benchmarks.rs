use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustygraph::*;

fn natural_visibility_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("natural_visibility");

    for size in [10, 50, 100, 500, 1000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.1).sin()).collect();
        let series = TimeSeries::from_raw(data).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let graph = VisibilityGraph::from_series(&series)
                    .natural_visibility()
                    .unwrap();
                black_box(graph);
            });
        });
    }
    group.finish();
}

fn horizontal_visibility_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("horizontal_visibility");

    for size in [10, 50, 100, 500, 1000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.1).sin()).collect();
        let series = TimeSeries::from_raw(data).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let graph = VisibilityGraph::from_series(&series)
                    .horizontal_visibility()
                    .unwrap();
                black_box(graph);
            });
        });
    }
    group.finish();
}

fn feature_computation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_computation");

    let size = 100;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin()).collect();
    let series = TimeSeries::from_raw(data).unwrap();

    // Benchmark with different numbers of features
    for num_features in [1, 3, 5, 10].iter() {
        let mut feature_set = FeatureSet::new();

        if *num_features >= 1 {
            feature_set = feature_set.add_builtin(BuiltinFeature::DeltaForward);
        }
        if *num_features >= 3 {
            feature_set = feature_set
                .add_builtin(BuiltinFeature::DeltaBackward)
                .add_builtin(BuiltinFeature::LocalSlope);
        }
        if *num_features >= 5 {
            feature_set = feature_set
                .add_builtin(BuiltinFeature::LocalMean)
                .add_builtin(BuiltinFeature::IsLocalMax);
        }
        if *num_features >= 10 {
            feature_set = feature_set
                .add_builtin(BuiltinFeature::IsLocalMin)
                .add_builtin(BuiltinFeature::Acceleration)
                .add_builtin(BuiltinFeature::LocalVariance)
                .add_builtin(BuiltinFeature::DeltaSymmetric)
                .add_builtin(BuiltinFeature::ZScore);
        }

        group.bench_with_input(
            BenchmarkId::new("features", num_features),
            num_features,
            |b, _| {
                b.iter(|| {
                    let graph = VisibilityGraph::from_series(&series)
                        .with_features(feature_set.clone())
                        .natural_visibility()
                        .unwrap();
                    black_box(graph);
                });
            },
        );
    }
    group.finish();
}

fn missing_data_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("missing_data");

    let size = 100;
    let data: Vec<Option<f64>> = (0..size)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some((i as f64 * 0.1).sin())
            }
        })
        .collect();

    let timestamps: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let series = TimeSeries::new(timestamps, data).unwrap();

    group.bench_function("linear_interpolation", |b| {
        b.iter(|| {
            let cleaned = series
                .handle_missing(MissingDataStrategy::LinearInterpolation)
                .unwrap();
            black_box(cleaned);
        });
    });

    group.bench_function("forward_fill", |b| {
        b.iter(|| {
            let cleaned = series
                .handle_missing(MissingDataStrategy::ForwardFill)
                .unwrap();
            black_box(cleaned);
        });
    });

    group.finish();
}

fn graph_metrics_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_metrics");

    let size = 100;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin()).collect();
    let series = TimeSeries::from_raw(data).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()
        .unwrap();

    group.bench_function("clustering_coefficient", |b| {
        b.iter(|| {
            let cc = graph.average_clustering_coefficient();
            black_box(cc);
        });
    });

    group.bench_function("diameter", |b| {
        b.iter(|| {
            let d = graph.diameter();
            black_box(d);
        });
    });

    group.bench_function("betweenness_centrality", |b| {
        b.iter(|| {
            let bc = graph.betweenness_centrality(50);
            black_box(bc);
        });
    });

    group.bench_function("detect_communities", |b| {
        b.iter(|| {
            let communities = graph.detect_communities();
            black_box(communities);
        });
    });

    group.finish();
}

fn export_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("export");

    let size = 100;
    let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin()).collect();
    let series = TimeSeries::from_raw(data).unwrap();
    let graph = VisibilityGraph::from_series(&series)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::LocalSlope)
        )
        .natural_visibility()
        .unwrap();

    group.bench_function("to_json", |b| {
        b.iter(|| {
            let json = graph.to_json(ExportOptions::default());
            black_box(json);
        });
    });

    group.bench_function("to_graphml", |b| {
        b.iter(|| {
            let graphml = graph.to_graphml();
            black_box(graphml);
        });
    });

    group.bench_function("to_dot", |b| {
        b.iter(|| {
            let dot = graph.to_dot();
            black_box(dot);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    natural_visibility_benchmark,
    horizontal_visibility_benchmark,
    feature_computation_benchmark,
    missing_data_benchmark,
    graph_metrics_benchmark,
    export_benchmark
);
criterion_main!(benches);

