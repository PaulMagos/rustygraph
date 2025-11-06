# RustyGraph Visual Guide

## Project Structure

```
rustygraph/
│
├── 📚 Documentation
│   ├── README.md                    ← Start here!
│   ├── ROADMAP.md                   ← Implementation plan
│   ├── TODO.md                      ← Specific tasks
│   ├── ARCHITECTURE.md              ← Design details
│   ├── CHANGELOG.md                 ← Version history
│   └── DOCUMENTATION_SUMMARY.md     ← This overview
│
├── 📦 Source Code (API Complete, Implementation Pending)
│   ├── src/
│   │   ├── lib.rs                   ← Main entry point
│   │   ├── time_series.rs           ← Data container
│   │   ├── visibility_graph.rs      ← Graph structure
│   │   ├── features/
│   │   │   ├── mod.rs               ← Feature framework
│   │   │   ├── builtin.rs           ← Pre-defined features
│   │   │   └── missing_data.rs      ← Imputation strategies
│   │   └── algorithms/
│   │       ├── mod.rs               ← Algorithm exports
│   │       ├── natural.rs           ← Natural visibility
│   │       └── horizontal.rs        ← Horizontal visibility
│   │
│   └── examples/
│       ├── basic_usage.rs           ← Simple example
│       └── with_features.rs         ← Advanced example
│
├── 🔧 Configuration
│   └── Cargo.toml                   ← Package metadata
│
└── 🏗️ Build Artifacts (generated)
    └── target/
        └── doc/                     ← HTML documentation
            └── rustygraph/
                └── index.html       ← Open this in browser!
```

## Data Flow Diagram

```
┌─────────────────┐
│  Raw Time       │
│  Series Data    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   TimeSeries<T>             │
│  ┌──────────────────────┐   │
│  │ timestamps: Vec<f64> │   │
│  │ values: Vec<Option>  │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼ handle_missing()
┌─────────────────────────────┐
│   Clean TimeSeries          │
│   (all Some values)         │
└────────┬────────────────────┘
         │
         ▼ VisibilityGraph::from_series()
┌─────────────────────────────┐
│  VisibilityGraphBuilder     │
│  ┌──────────────────────┐   │
│  │ .with_features()     │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼ .natural_visibility() or .horizontal_visibility()
┌─────────────────────────────┐
│   Algorithms                │
│  ┌──────────────────────┐   │
│  │ compute_edges()      │   │
│  │ → Vec<(usize,usize)> │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼ Build graph + Compute features
┌─────────────────────────────────────────┐
│   VisibilityGraph<T>                    │
│  ┌─────────────────────────────────┐    │
│  │ edges: Vec<(usize, usize)>      │    │
│  │ adjacency: Vec<Vec<usize>>      │    │
│  │ node_features: Vec<HashMap>     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Graph Analysis            │
│  • Degree sequence          │
│  • Neighbors                │
│  • Features per node        │
│  • Adjacency matrix         │
└─────────────────────────────┘
```

## Module Dependencies

```
┌─────────────┐
│   lib.rs    │  ← Main crate
└──────┬──────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌─────────────┐  ┌──────────────┐
│ time_series │  │ algorithms   │
└──────┬──────┘  └──────┬───────┘
       │                │
       │                ├──► natural.rs
       │                └──► horizontal.rs
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌─────────────────┐
│ visibility   │◄─┤    features     │
│    _graph    │  └────────┬────────┘
└──────────────┘           │
                           ├──► builtin.rs
                           └──► missing_data.rs
```

## API Usage Pattern

### Basic Usage

```rust
// 1. Create time series
let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);

// 2. Build graph
let graph = VisibilityGraph::from_series(&series)
    .natural_visibility()?;

// 3. Analyze
println!("Edges: {}", graph.edges().len());
```

### Advanced Usage

```rust
// 1. Create with missing data
let series = TimeSeries::new(timestamps, values)?;

// 2. Handle missing
let clean = series.handle_missing(
    MissingDataStrategy::LinearInterpolation
        .with_fallback(MissingDataStrategy::ForwardFill)
)?;

// 3. Configure features
let features = FeatureSet::new()
    .add_builtin(BuiltinFeature::DeltaForward)
    .add_function("custom", |s, i| s[i].map(|v| v * 2.0));

// 4. Build with features
let graph = VisibilityGraph::from_series(&clean)
    .with_features(features)
    .natural_visibility()?;

// 5. Inspect results
for i in 0..graph.node_count {
    let features = graph.node_features(i)?;
    println!("Node {}: {:?}", i, features);
}
```

## Implementation Status

```
Phase 1 (MVP) ............................ [ ░░░░░░░░░░ ]  0%
Phase 2 (Feature Complete) .............. [ ░░░░░░░░░░ ]  0%
Phase 3 (Performance) ................... [ ░░░░░░░░░░ ]  0%
Phase 4 (Advanced) ...................... [ ░░░░░░░░░░ ]  0%
Phase 5 (Ecosystem) ..................... [ ░░░░░░░░░░ ]  0%

API Design ............................... [██████████] 100%
Documentation ............................ [██████████] 100%
Examples ................................. [██████████] 100%
```

## Getting Started Guide

### For Users (After Implementation)

1. Add to `Cargo.toml`: `rustygraph = "0.2"`
2. Read the [Quick Start](#quick-start) in README
3. Browse examples: `examples/basic_usage.rs`
4. Explore API docs: `cargo doc --open`

### For Contributors (Now)

1. Read **README.md** for overview
2. Study **ARCHITECTURE.md** for design
3. Check **TODO.md** for tasks
4. Pick a task and implement!
5. Add tests
6. Submit PR

## Feature Matrix

| Feature Category | Built-in | Custom | Status |
|-----------------|----------|--------|--------|
| **Algorithms** |
| Natural Visibility | ✓ | - | 🚧 Pending |
| Horizontal Visibility | ✓ | - | 🚧 Pending |
| **Node Features** |
| Temporal (Δ, slope) | ✓ | ✓ | 🚧 Pending |
| Statistical (mean, var) | ✓ | ✓ | 🚧 Pending |
| Extrema (peaks, valleys) | ✓ | ✓ | 🚧 Pending |
| Custom functions | - | ✓ | 🚧 Pending |
| **Missing Data** |
| Interpolation | ✓ | ✓ | 🚧 Pending |
| Fill strategies | ✓ | ✓ | 🚧 Pending |
| Window-based | ✓ | ✓ | 🚧 Pending |
| Custom handlers | - | ✓ | 🚧 Pending |
| **Graph Operations** |
| Degree queries | ✓ | - | ✅ API Ready |
| Neighbor queries | ✓ | - | ✅ API Ready |
| Adjacency matrix | ✓ | - | 🚧 Pending |
| **Performance** |
| Parallel features | ✓ | - | ⏳ Future |
| SIMD optimizations | ✓ | - | ⏳ Future |
| Lazy evaluation | ✓ | - | ⏳ Future |

Legend: ✅ Complete | 🚧 Pending | ⏳ Planned | ✓ Supported | - Not applicable

## Performance Targets

```
Series Size    Target Time    Memory Usage
──────────────────────────────────────────
100 points     < 1 ms         ~10 KB
1,000 points   < 10 ms        ~100 KB
10,000 points  < 1 sec        ~10 MB
100,000 points < 30 sec       ~100 MB
```

## Next Steps

### Immediate (Phase 1 - MVP)
1. ✅ Complete API design ← **Done!**
2. ⏭️ Implement `natural::compute_edges()`
3. ⏭️ Implement `horizontal::compute_edges()`
4. ⏭️ Wire up graph construction
5. ⏭️ Add 3 basic features
6. ⏭️ Add linear interpolation
7. ⏭️ Write unit tests
8. ⏭️ Run examples successfully

### Short Term (Phase 2)
- Complete all built-in features
- Complete all imputation strategies
- Add custom function support
- Comprehensive testing

### Long Term (Phase 3-5)
- Performance optimization
- Advanced features
- Python bindings
- Publications

## Resources

### Generated Documentation
```bash
cargo doc --open
```

Opens: `target/doc/rustygraph/index.html`

### Example Usage
```bash
# After implementation:
cargo run --example basic_usage
cargo run --example with_features
```

### Testing
```bash
# After implementation:
cargo test
cargo test --doc  # Test examples in docs
```

### Benchmarking
```bash
# After implementation:
cargo bench
```

---

**Status**: ✅ API Design & Documentation Complete  
**Next**: 🚧 Begin Phase 1 Implementation  
**Target**: 🎯 v0.2.0 MVP in 4-6 weeks

Last Updated: 2025-11-06

