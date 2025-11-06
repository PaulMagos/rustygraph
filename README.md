# RustyGraph

A high-performance Rust library for visibility graph computation from time series data with extensible node feature computation.

[![Documentation](https://img.shields.io/badge/docs-rustdoc-blue)](https://docs.rs/rustygraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📚 Documentation Index

- **[README.md](README.md)** - You are here! Main overview and quick start
- **[IMPLEMENTATION_SCHEDULE.md](IMPLEMENTATION_SCHEDULE.md)** - ⭐ **START HERE** for step-by-step implementation guide
- **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Visual diagrams and quick reference
- **API Docs** - Run `cargo doc --open` for full rustdoc documentation


## Features

- **Natural Visibility Graphs**: O(n) implementation using monotonic stack optimization
- **Horizontal Visibility Graphs**: Fast O(n) average case algorithm
- **Node Feature Computation**: Extensible system for computing node features (basis expansion/data augmentation)
- **Missing Data Handling**: Configurable strategies for imputation
- **Custom Functions**: Support for user-defined features and imputation strategies
- **Type Generic**: Works with both `f32` and `f64`

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustygraph = "0.1.0"
```

### Basic Usage

```rust
use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a time series
    let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);

    // Build a natural visibility graph
    let graph = VisibilityGraph::from_series(&series)
        .natural_visibility()?;

    // Access the results
    println!("Number of edges: {}", graph.edges().len());
    println!("Degree sequence: {:?}", graph.degree_sequence());
    
    Ok(())
}
```

### Advanced Usage with Features

```rust
use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create time series with missing data
    let series = TimeSeries::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![Some(1.0), None, Some(3.0), Some(2.0), Some(4.0)]
    )?;

    // Handle missing data
    let cleaned = series.handle_missing(
        MissingDataStrategy::LinearInterpolation
            .with_fallback(MissingDataStrategy::ForwardFill)
    )?;

    // Create graph with node features
    let graph = VisibilityGraph::from_series(&cleaned)
        .with_features(
            FeatureSet::new()
                .add_builtin(BuiltinFeature::DeltaForward)
                .add_builtin(BuiltinFeature::LocalSlope)
                .add_function("squared", |series, idx| {
                    series[idx].map(|v| v * v)
                })
        )
        .horizontal_visibility()?;

    // Inspect node features
    for i in 0..graph.node_count {
        if let Some(features) = graph.node_features(i) {
            println!("Node {}: {:?}", i, features);
        }
    }
    
    Ok(())
}
```

## Architecture

The library is organized into several modules:

- **`time_series`**: Time series data structures and preprocessing
- **`visibility_graph`**: Visibility graph construction and representation
- **`features`**: Node feature computation framework
- **`features::missing_data`**: Missing data handling strategies
- **`algorithms`**: Core visibility graph algorithms

## Built-in Features

The library includes several pre-defined node features:

### Temporal Derivatives
- `DeltaForward`: y[i+1] - y[i]
- `DeltaBackward`: y[i] - y[i-1]
- `DeltaSymmetric`: (y[i+1] - y[i-1]) / 2
- `LocalSlope`: (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
- `Acceleration`: Second derivative approximation

### Local Statistics
- `LocalMean`: Average over local window
- `LocalVariance`: Variance over local window
- `ZScore`: (y[i] - mean) / std

### Extrema Detection
- `IsLocalMax`: Detects peaks
- `IsLocalMin`: Detects valleys

## Missing Data Strategies

- **LinearInterpolation**: Average of neighboring valid values
- **ForwardFill**: Use last valid value
- **BackwardFill**: Use next valid value
- **NearestNeighbor**: Use closest valid value
- **MeanImputation**: Local window mean
- **MedianImputation**: Local window median
- **ZeroFill**: Replace with zero
- **Drop**: Skip missing values

Strategies can be chained with fallbacks:

```rust
let strategy = MissingDataStrategy::LinearInterpolation
    .with_fallback(MissingDataStrategy::ForwardFill)
    .with_fallback(MissingDataStrategy::ZeroFill);
```

## Custom Features

### Simple Function

```rust
let features = FeatureSet::new()
    .add_function("log", |series, idx| {
        series[idx].map(|v| v.ln())
    });
```

### Full Trait Implementation

```rust
use rustygraph::features::{Feature, MissingDataHandler};

struct RangeFeature {
    window: usize,
}

impl Feature<f64> for RangeFeature {
    fn compute(
        &self,
        series: &[Option<f64>],
        index: usize,
        missing_handler: &dyn MissingDataHandler<f64>,
    ) -> Option<f64> {
        let start = index.saturating_sub(self.window / 2);
        let end = (index + self.window / 2).min(series.len());
        
        let valid: Vec<f64> = series[start..end]
            .iter()
            .filter_map(|&v| v)
            .collect();
        
        if valid.is_empty() {
            return None;
        }
        
        let min = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        Some(max - min)
    }

    fn name(&self) -> &str {
        "range"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }

    fn window_size(&self) -> Option<usize> {
        Some(self.window)
    }
}
```

## Performance

- **Natural visibility**: O(n) per node using monotonic stack optimization
- **Horizontal visibility**: O(n) average case
- **Memory efficient**: Adjacency list representation for sparse graphs
- **Type generic**: Works with both `f32` and `f64`

## Current Status & Roadmap

### 🎯 Current Status

This library currently provides the **API design and documentation** for a comprehensive visibility graph computation framework. The interfaces are fully specified and documented, but **core implementations are pending**.

### ✅ Completed

- **Complete API design** with full rustdoc documentation
- **Type signatures** for all public interfaces
- **Module organization** and architecture
- **Error types** and handling patterns
- **Trait definitions** for extensibility
- **Builder patterns** for ergonomic API
- **Example code** demonstrating intended usage

### 🚧 Pending Implementations

#### Core Algorithms
- [ ] **Natural visibility algorithm** - O(n) monotonic stack implementation
- [ ] **Horizontal visibility algorithm** - Linear scan approach
- [ ] Edge deduplication and graph construction

#### Feature Computation
- [ ] **Built-in features** implementation:
  - [ ] DeltaForward, DeltaBackward, DeltaSymmetric
  - [ ] LocalSlope, Acceleration
  - [ ] LocalMean, LocalVariance
  - [ ] IsLocalMax, IsLocalMin
  - [ ] ZScore
- [ ] **Feature computation pipeline** integration
- [ ] **Custom function wrapper** for closures

#### Missing Data Handling
- [ ] **LinearInterpolation** implementation
- [ ] **ForwardFill** and **BackwardFill**
- [ ] **NearestNeighbor** selection
- [ ] **MeanImputation** and **MedianImputation** with windows
- [ ] **ZeroFill** and **Drop** strategies
- [ ] **Fallback chain** execution logic
- [ ] Integration with `TimeSeries::handle_missing()`

#### Graph Operations
- [ ] Adjacency list construction from edges
- [ ] Degree computation and caching
- [ ] Adjacency matrix export
- [ ] Graph validation and consistency checks

### 🔮 Future Enhancements

#### Performance Optimizations
- [ ] **Parallel feature computation** using `rayon`
- [ ] **SIMD optimizations** for numerical operations
- [ ] **Lazy evaluation** for expensive features
- [ ] **Caching** for intermediate computations
- [ ] **Memory pooling** for large graphs

#### Advanced Features
- [ ] **Weighted visibility graphs** with edge weights
- [ ] **Directed vs undirected** graph options
- [ ] **Frequency domain features** (FFT coefficients)
- [ ] **Wavelet-based features** for multi-scale analysis
- [ ] **Graph-theoretic features**:
  - [ ] Clustering coefficient
  - [ ] Betweenness centrality
  - [ ] Community detection
  - [ ] Path-based metrics

#### Serialization & I/O
- [ ] **Graph export formats**:
  - [ ] GraphML
  - [ ] JSON
  - [ ] Edge list (CSV)
  - [ ] Adjacency matrix (CSV/NPY)
- [ ] **Feature export**:
  - [ ] CSV with headers
  - [ ] Parquet for large datasets
  - [ ] HDF5 integration
- [ ] **Time series import** from common formats

#### Integration & Interoperability
- [ ] **`petgraph` integration** for advanced algorithms
- [ ] **`ndarray` support** for matrix operations
- [ ] **`polars`/`arrow` integration** for data frames
- [ ] **Python bindings** via PyO3
- [ ] **C API** for cross-language usage

#### Validation & Quality
- [ ] **Comprehensive unit tests** for all algorithms
- [ ] **Property-based testing** with `proptest`
- [ ] **Benchmarking suite** with `criterion`
- [ ] **Example datasets** and reproducible benchmarks
- [ ] **Validation against reference implementations**

#### Documentation & Usability
- [ ] **Tutorial series** for common use cases
- [ ] **Jupyter notebook examples** (via Python bindings)
- [ ] **Performance tuning guide**
- [ ] **Migration guide** from other libraries
- [ ] **API stability guarantees**

### 🎓 Use Cases (Once Implemented)

- **Climate data**: Temperature and precipitation patterns
- **Energy Load Disaggregation**: Consumption pattern analysis
- **Energy Load and Solar Forecasting**: Predictive modeling
- **Financial time series analysis**: Market volatility patterns
- **Physiological signals**: ECG, EEG analysis
- **Industrial monitoring**: Sensor data anomaly detection
- **Network traffic analysis**: Pattern recognition
- **Seismic data**: Earthquake pattern detection

### 📊 Implementation Priority

**Phase 1: Core Functionality** (Essential)
1. Natural visibility algorithm
2. Horizontal visibility algorithm
3. Basic feature computation (DeltaForward, DeltaBackward, LocalSlope)
4. Linear interpolation for missing data
5. Unit tests for core algorithms

**Phase 2: Feature Completeness**
6. All built-in features
7. All missing data strategies
8. Custom feature function support
9. Graph export to adjacency matrix

**Phase 3: Performance & Robustness**
10. Parallel feature computation
11. Comprehensive test suite
12. Benchmarking and optimization
13. Edge case handling

**Phase 4: Advanced Features**
14. Weighted graphs
15. Graph-theoretic metrics
16. Serialization formats
17. External library integration

## Documentation

Generate and view the full documentation:

```bash
cargo doc --open
```

The documentation provides complete API specifications with examples, even though implementations are pending.

## References

- Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuno, J. C. (2008). "From time series to complex networks: The visibility graph." *Proceedings of the National Academy of Sciences*, 105(13), 4972-4975.

- Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009). "Horizontal visibility graphs: Exact results for random time series." *Physical Review E*, 80(4), 046103.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


