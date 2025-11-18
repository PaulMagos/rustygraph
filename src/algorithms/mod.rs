//! Visibility graph algorithms.
//!
//! This module contains the core algorithms for computing visibility graphs
//! from time series data. Both unweighted and weighted graph variants are supported.
//!
//! # Algorithms
//!
//! - [`natural`]: Natural visibility graph (O(n) per node)
//! - [`horizontal`]: Horizontal visibility graph (O(n) average)
//!
//! # Weighted Graphs
//!
//! Each algorithm provides two variants:
//! - **Unweighted**: Returns `HashMap<(usize, usize), f64>` of edge pairs
//! - **Weighted**: Returns `HashMap<(usize, usize), f64>` with custom edge weights
//!
//! # Examples
//!
//! ## Basic unweighted graph
//!
//! ```rust
//! use rustygraph::algorithms::VisibilityEdges;
//!
//! let series = vec![1.0, 3.0, 2.0, 4.0];
//! let edges = natural_visibility(&series);
//! ```
//!
//! ## Weighted graph with custom weights
//!
//! ```rust
//! use rustygraph::algorithms::natural::compute_edges_weighted;
//!
//! let series = vec![1.0, 3.0, 2.0, 4.0];
//! 
//! // Weight by value difference
//! let edges = compute_edges_weighted(&series, |_, _, vi, vj| {
//!     (vj - vi).abs()
//! });
//! ```
//!
//! # References
//!
//! - Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuno, J. C. (2008).
//!   "From time series to complex networks: The visibility graph."
//!   Proceedings of the National Academy of Sciences, 105(13), 4972-4975.

mod edges;

use crate::TimeSeries;

pub use self::edges::{VisibilityEdges, VisibilityType};

// Re-export for backward compatibility
pub use self::edges::VisibilityEdges as create_visibility_edges;
pub use self::edges::VisibilityType::Natural as natural_visibility_type;
pub use self::edges::VisibilityType::Horizontal as horizontal_visibility_type;

/// Computes natural visibility edges with default unweighted (weight=1.0) edges.
///
/// # Arguments
///
/// - `series`: Time series data as a slice
///
/// # Returns
///
/// Vector of (source, target, weight) tuples
///
/// # Examples
///
/// ```rust
/// use rustygraph::algorithms::natural_visibility;
/// use rustygraph::TimeSeries;
///
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let edges = natural_visibility(&series);
/// ```
pub fn natural_visibility<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64>,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Natural, |_, _, _, _| 1.0)
        .compute_edges();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes horizontal visibility edges with default unweighted (weight=1.0) edges.
///
/// # Arguments
///
/// - `series`: Time series data as a slice
///
/// # Returns
///
/// Vector of (source, target, weight) tuples
///
/// # Examples
///
/// ```rust
/// use rustygraph::algorithms::horizontal_visibility;
/// use rustygraph::TimeSeries;
///
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let edges = horizontal_visibility(&series);
/// ```
pub fn horizontal_visibility<T>(series: &TimeSeries<T>) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64>,
{
    let edges = VisibilityEdges::new(series, VisibilityType::Horizontal, |_, _, _, _| 1.0)
        .compute_edges();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}

/// Computes visibility edges with a custom weight function.
///
/// # Arguments
///
/// - `series`: Time series data
/// - `vis_type`: Visibility algorithm type (Natural or Horizontal)
/// - `weight_fn`: Function to compute edge weights `(src_idx, dst_idx, src_val, dst_val) -> weight`
///
/// # Returns
///
/// Vector of (source, target, weight) tuples
///
/// # Examples
///
/// ```rust
/// use rustygraph::algorithms::{visibility_weighted, VisibilityType};
/// use rustygraph::TimeSeries;
///
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let edges = visibility_weighted(&series, VisibilityType::Natural, |_, _, vi, vj| {
///     (vj - vi).abs()
/// });
/// ```
pub fn visibility_weighted<T, F>(
    series: &TimeSeries<T>,
    vis_type: VisibilityType,
    weight_fn: F,
) -> Vec<(usize, usize, f64)>
where
    T: Copy + PartialOrd + Into<f64>,
    F: Fn(usize, usize, T, T) -> f64,
{
    let edges = VisibilityEdges::new(series, vis_type, weight_fn).compute_edges();
    edges.into_iter().map(|((src, dst), w)| (src, dst, w)).collect()
}


