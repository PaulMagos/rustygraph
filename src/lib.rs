//! # RustyGraph
//!
//! `rustygraph` is a high-performance Rust library for visibility graph computation from time series data.
//!
//! ## Features
//!
//! - **Natural Visibility Graphs**: O(n) implementation using monotonic stack optimization
//! - **Horizontal Visibility Graphs**: Fast O(n) average case algorithm
//! - **Node Feature Computation**: Extensible system for computing node features (basis expansion/data augmentation)
//! - **Missing Data Handling**: Configurable strategies for imputation
//! - **Custom Functions**: Support for user-defined features and imputation strategies
//!
//! ## Quick Start
//!
//! ```rust
//! use rustygraph::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a time series
//! let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0])?;
//!
//! // Build a natural visibility graph
//! let graph = VisibilityGraph::from_series(&series)
//!     .natural_visibility()?;
//!
//! // Access the results
//! println!("Number of edges: {}", graph.edges().len());
//! println!("Degree sequence: {:?}", graph.degree_sequence());
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage with Features
//!
//! ```rust
//! use rustygraph::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create time series with missing data
//! let series = TimeSeries::new(
//!     vec![0.0, 1.0, 2.0, 3.0, 4.0],
//!     vec![Some(1.0), None, Some(3.0), Some(2.0), Some(4.0)]
//! )?;
//!
//! // Handle missing data
//! let cleaned = series.handle_missing(
//!     MissingDataStrategy::LinearInterpolation
//!         .with_fallback(MissingDataStrategy::ForwardFill)
//! )?;
//!
//! // Create graph with node features
//! let graph = VisibilityGraph::from_series(&cleaned)
//!     .with_features(
//!         FeatureSet::new()
//!             .add_builtin(BuiltinFeature::DeltaForward)
//!             .add_builtin(BuiltinFeature::LocalSlope)
//!             .add_function("squared", |series, idx| {
//!                 series[idx].map(|v| v * v)
//!             })
//!     )
//!     .horizontal_visibility()?;
//!
//! // Inspect node features
//! for i in 0..graph.node_count {
//!     if let Some(features) = graph.node_features(i) {
//!         println!("Node {}: {:?}", i, features);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`time_series`]: Time series data structures and preprocessing
//! - [`visibility_graph`]: Visibility graph construction and representation
//! - [`features`]: Node feature computation framework
//! - [`features::missing_data`]: Missing data handling strategies
//! - [`algorithms`]: Core visibility graph algorithms
//!
//! ## Performance
//!
//! - **Natural visibility**: O(n) per node using monotonic stack optimization
//! - **Horizontal visibility**: O(n) average case
//! - **Memory efficient**: Adjacency list representation for sparse graphs
//! - **Type generic**: Works with both `f32` and `f64`

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

// New modular structure
pub mod core;
pub mod analysis;
pub mod io;
pub mod integrations;
pub mod performance;
pub mod advanced;
pub mod utils;

// Backward compatibility: re-export modules with old names
pub use core::time_series;
pub use core::visibility_graph;
pub use core::features;
pub use core::algorithms;
pub use io::export;
pub use analysis::metrics;
pub use analysis::statistics;
pub use io::import;
pub use performance::parallel;
pub use performance::batch;
pub use analysis::community;
pub use utils::datasets;
pub use performance::lazy;
pub use performance::simd;
pub use analysis::motifs;

#[cfg(any(feature = "npy-export", feature = "parquet-export", feature = "hdf5-export"))]
pub use io::export_advanced;

// Integration modules (feature-gated) - backward compatibility
#[cfg(feature = "petgraph-integration")]
pub use integrations::petgraph as petgraph_integration;
#[cfg(feature = "ndarray-support")]
pub use integrations::ndarray as ndarray_support;
#[cfg(feature = "python-bindings")]
pub use integrations::python;

// Re-export main types for convenience
pub use core::{TimeSeries, TimeSeriesError};
pub use core::{VisibilityGraph, VisibilityGraphBuilder, GraphError, GraphDirection};
pub use core::{Feature, FeatureSet, BuiltinFeature};
pub use core::{MissingDataHandler, MissingDataStrategy, ImputationError};
pub use io::{ExportFormat, ExportOptions};
pub use analysis::GraphStatistics;
pub use io::CsvImportOptions;
pub use analysis::Communities;
pub use performance::{BatchProcessor, BatchResults, compare_graphs};
