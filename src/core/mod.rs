//! Core library functionality for visibility graph computation.
//!
//! This module contains the essential types and algorithms that are always available,
//! regardless of feature flags. It includes time series data structures, visibility
//! graph representations, and the core algorithms for graph construction.
pub mod time_series;
pub mod visibility_graph;
pub mod algorithms;
pub mod features;
pub mod data_split;
// Re-export main types for convenience
pub use self::time_series::{TimeSeries, TimeSeriesError};
pub use self::visibility_graph::{VisibilityGraph, VisibilityGraphBuilder, GraphError, GraphDirection};
pub use self::features::{Feature, FeatureSet, BuiltinFeature};
pub use self::features::missing_data::{MissingDataHandler, MissingDataStrategy, ImputationError};
