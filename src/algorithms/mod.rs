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
//! - **Unweighted**: Returns `Vec<(usize, usize)>` of edge pairs
//! - **Weighted**: Returns `Vec<(usize, usize, f64)>` with custom edge weights
//!
//! # Examples
//!
//! ## Basic unweighted graph
//!
//! ```rust
//! use rustygraph::algorithms::natural_visibility;
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

pub mod natural;
pub mod horizontal;

// Re-export unweighted functions
pub use natural::compute_edges as natural_visibility;
pub use horizontal::compute_edges as horizontal_visibility;

// Re-export weighted functions
pub use natural::compute_edges_weighted as natural_visibility_weighted;
pub use horizontal::compute_edges_weighted as horizontal_visibility_weighted;

