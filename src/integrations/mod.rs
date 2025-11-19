//! Integrations with external libraries.
//!
//! This module provides integrations with popular Rust ecosystem libraries
//! like petgraph, ndarray, and Python bindings. All integrations are
//! feature-gated.

#[cfg(feature = "petgraph-integration")]
pub mod petgraph;

#[cfg(feature = "ndarray-support")]
pub mod ndarray;

#[cfg(feature = "python-bindings")]
pub mod python;

