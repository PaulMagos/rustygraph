//! Performance optimization features.
//!
//! This module contains various performance optimizations including
//! parallel processing, SIMD operations, lazy evaluation, and batch processing.

pub mod parallel;
pub mod simd;
pub mod lazy;
pub mod batch;
pub mod tuning;
pub mod gpu;

#[cfg(target_os = "macos")]
pub mod metal;

// Re-export main types
pub use self::batch::{BatchProcessor, BatchResults, compare_graphs};
pub use self::lazy::LazyMetrics;
pub use self::tuning::{PerformanceTuning, SystemCapabilities};
pub use self::gpu::{GpuVisibilityGraph, GpuConfig, GpuBackend, GpuCapabilities};

