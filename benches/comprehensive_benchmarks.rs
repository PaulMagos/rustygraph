//! Comprehensive performance benchmark suite
//!
//! This benchmark validates all performance claims made in the documentation.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rustygraph::*;
use std::time::Duration;

fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let t = i as f64 * 0.1;
            (t).sin() + 0.3 * (2.0 * t).cos()
        })
        .collect()
}

/// Benchmark: Sequential vs Parallel edge computation
fn benchmark_parallel_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_edges");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000, 2000].iter() {
        let data = generate_test_data(*size);
        let series = TimeSeries::from_raw(data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _graph = VisibilityGraph::from_series(&series)
                        .natural_visibility()
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: With and without features
fn benchmark_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("features");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000].iter() {
        let data = generate_test_data(*size);
        let series = TimeSeries::from_raw(data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("without_features", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _graph = VisibilityGraph::from_series(&series)
                        .natural_visibility()
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("with_features", size),
            size,
            |b, _| {
                b.iter(|| {
                    let feature_set = FeatureSet::new()
                        .add_builtin(BuiltinFeature::DeltaForward)
                        .add_builtin(BuiltinFeature::DeltaBackward)
                        .add_builtin(BuiltinFeature::LocalSlope);
                    let _graph = VisibilityGraph::from_series(&series)
                        .with_features(feature_set)
                        .natural_visibility()
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Natural vs Horizontal visibility
fn benchmark_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 500, 1000].iter() {
        let data = generate_test_data(*size);
        let series = TimeSeries::from_raw(data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("natural", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _graph = VisibilityGraph::from_series(&series)
                        .natural_visibility()
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("horizontal", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _graph = VisibilityGraph::from_series(&series)
                        .horizontal_visibility()
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Statistics computation
fn benchmark_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    for size in [100, 500, 1000].iter() {
        let data = generate_test_data(*size);
        let series = TimeSeries::from_raw(data.clone()).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("compute_all", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _stats = graph.compute_statistics();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("degree_centrality", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _cent = graph.degree_centrality();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("betweenness", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _cent = graph.betweenness_centrality_batch();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch processing
fn benchmark_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");
    group.measurement_time(Duration::from_secs(15));

    let sizes = vec![10, 20, 50];

    for num_series in sizes.iter() {
        let mut processor = BatchProcessor::new();
        let mut series_vec = Vec::new();

        for _ in 0..*num_series {
            let data = generate_test_data(100);
            series_vec.push(TimeSeries::from_raw(data).unwrap());
        }

        for (i, series) in series_vec.iter().enumerate() {
            processor = processor.add_series(series, &format!("series_{}", i));
        }

        group.bench_with_input(
            BenchmarkId::new("batch_natural", num_series),
            num_series,
            |b, _| {
                b.iter(|| {
                    let mut proc = BatchProcessor::new();
                    for (i, series) in series_vec.iter().enumerate() {
                        proc = proc.add_series(series, &format!("series_{}", i));
                    }
                    let _results = proc.process_natural().unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_parallel_edges,
    benchmark_features,
    benchmark_algorithms,
    benchmark_statistics,
    benchmark_batch,
);

criterion_main!(benches);

