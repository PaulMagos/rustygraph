//! Core visibility graph algorithms.
//!
//! This module contains the fundamental algorithms for computing visibility graphs
//! from time series data. Both natural and horizontal visibility algorithms are
//! implemented with O(n) time complexity using optimized data structures.

pub mod edges;

pub use self::edges::{VisibilityEdges, VisibilityType};

use crate::core::TimeSeries;

/// Computes natural visibility edges with default unweighted (weight=1.0) edges.
///
/// Uses O(n) envelope optimization for efficiency.
pub fn natural_visibility<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64>,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Natural, |_, _, _, _| 1.0)
        .compute_edges();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes natural visibility edges in parallel (requires `parallel` feature).
///
/// Uses O(n) envelope optimization within each parallel chunk for better efficiency
/// than the naive O(n²) parallel approach.
///
/// # Performance
///
/// Best for large graphs (>1000 nodes) on multi-core systems.
/// Expected speedup: 2-4x on 4-8 cores.
///
/// # Example
///
/// ```rust
/// use rustygraph::{TimeSeries, algorithms::natural_visibility_parallel};
///
/// let data: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.1).sin()).collect();
/// let series = TimeSeries::from_raw(data).unwrap();
/// let edges = natural_visibility_parallel(&series);
/// ```
#[cfg(feature = "parallel")]
pub fn natural_visibility_parallel<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64> + Send + Sync,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Natural, |_, _, _, _| 1.0)
        .compute_edges_parallel();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes horizontal visibility edges with default unweighted (weight=1.0) edges.
///
/// Uses O(n) envelope optimization for efficiency.
pub fn horizontal_visibility<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64>,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Horizontal, |_, _, _, _| 1.0)
        .compute_edges();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes horizontal visibility edges in parallel (requires `parallel` feature).
///
/// Uses O(n) envelope optimization within each parallel chunk for better efficiency
/// than the naive O(n²) parallel approach.
///
/// # Performance
///
/// Best for large graphs (>1000 nodes) on multi-core systems.
/// Expected speedup: 2-4x on 4-8 cores.
#[cfg(feature = "parallel")]
pub fn horizontal_visibility_parallel<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64> + Send + Sync,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Horizontal, |_, _, _, _| 1.0)
        .compute_edges_parallel();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes visibility edges with custom weight function.
///
/// # Arguments
///
/// - `series`: Time series data
/// - `visibility_type`: Natural or Horizontal visibility
/// - `weight_fn`: Custom function to compute edge weights
///
/// # Returns
///
/// HashMap of edges with their weights
pub fn visibility_weighted<T, F>(
    series: &TimeSeries<T>,
    visibility_type: VisibilityType,
    weight_fn: F,
) -> std::collections::HashMap<(usize, usize), f64>
where
    T: Copy + PartialOrd + Into<f64>,
    F: Fn(usize, usize, T, T) -> f64,
{
    VisibilityEdges::new(series, visibility_type, weight_fn).compute_edges()
}
