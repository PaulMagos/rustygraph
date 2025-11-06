//! Visibility graph algorithms.
//!
//! This module contains the core algorithms for computing visibility graphs
//! from time series data.
//!
//! # Algorithms
//!
//! - [`natural`]: Natural visibility graph (O(n) per node)
//! - [`horizontal`]: Horizontal visibility graph (O(n) average)
//!
//! # References
//!
//! - Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuno, J. C. (2008).
//!   "From time series to complex networks: The visibility graph."
//!   Proceedings of the National Academy of Sciences, 105(13), 4972-4975.

pub mod natural;
pub mod horizontal;

pub use natural::compute_edges as natural_visibility;
pub use horizontal::compute_edges as horizontal_visibility;

