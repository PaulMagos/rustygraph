//! Graph analysis tools and algorithms.
//!
//! This module provides various algorithms for analyzing visibility graphs,
//! including metrics computation, statistical analysis, community detection,
//! and motif identification.
pub mod metrics;
pub mod statistics;
pub mod community;
pub mod motifs;
// Re-export main types
pub use self::statistics::GraphStatistics;
pub use self::community::Communities;
pub use self::motifs::{MotifCounts, Motif3};
