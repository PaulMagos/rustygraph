# Project Organization

This document explains the organization of the RustyGraph project.

## Directory Structure

```
rustygraph/
├── src/                    # Source code
│   ├── core/              # Core types (TimeSeries, VisibilityGraph)
│   ├── algorithms/        # Visibility graph algorithms
│   ├── analysis/          # Graph analysis (metrics, communities)
│   ├── performance/       # Optimizations (parallel, SIMD)
│   ├── io/                # Import/Export
│   ├── integrations/      # External library integrations
│   ├── advanced/          # Advanced features
│   └── utils/             # Utilities
├── examples/              # Usage examples
│   ├── testing/          # Test and verification examples
│   ├── benchmarks/       # Performance benchmarks
│   └── gpu/              # GPU examples (experimental)
├── benches/              # Criterion benchmarks
├── tests/                # Integration tests
├── scripts/              # Helper scripts
│   ├── *.sh             # Shell scripts
│   └── *.py             # Python test scripts
├── python/               # Python bindings source
└── docs/                 # Documentation (this folder)

## Examples Organization

### Main Examples (examples/)
- `basic_usage.rs` - Getting started
- `advanced_*.rs` - Advanced usage patterns
- `community_detection.rs`, `weighted_graphs.rs`, etc.

### Testing Examples (examples/testing/)
- Correctness verification
- Edge count comparisons
- Ground truth validation

### Benchmark Examples (examples/benchmarks/)
- Performance measurements
- Validation of claims

### GPU Examples (examples/gpu/)
- GPU/Metal implementations (experimental)
- Not part of main API yet

## Features

- `default` - Parallel + CSV import + SIMD
- `parallel` - Multi-threaded processing via Rayon
- `simd` - SIMD optimizations (NEON/AVX2)
- `python-bindings` - PyO3 Python bindings
- `csv-import` - CSV file import
- `advanced-features` - FFT and frequency analysis
- Various export formats (npy, parquet, hdf5)

## Scripts

- `compare_edge_counts.sh` - Verify sequential vs optimized
- `benchmark_rust_vs_python.py` - Cross-language benchmarks
- `test_python_*.py` - Python binding tests

