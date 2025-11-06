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
//! let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
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

pub mod time_series;
pub mod visibility_graph;
pub mod features;
pub mod algorithms;

// Re-export main types for convenience
pub use time_series::{TimeSeries, TimeSeriesError};
pub use visibility_graph::{VisibilityGraph, VisibilityGraphBuilder, GraphError};
pub use features::{Feature, FeatureSet, BuiltinFeature};
pub use features::missing_data::{MissingDataHandler, MissingDataStrategy, ImputationError};

