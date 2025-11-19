//! Performance tuning utilities for optimizing visibility graph computation.
//!
//! This module provides utilities to tune performance parameters based on
//! workload characteristics and hardware capabilities.

/// Performance tuning configuration for visibility graph computation.
#[derive(Debug, Clone)]
pub struct PerformanceTuning {
    /// Threshold distance for enabling SIMD (default: 8)
    pub simd_threshold: usize,

    /// Minimum graph size for parallel edge computation (default: 100)
    pub parallel_edge_threshold: usize,

    /// Minimum number of features for parallel feature computation (default: 3)
    pub parallel_feature_threshold: usize,

    /// Batch size for parallel processing (default: num_cpus)
    pub batch_size: usize,
}

impl Default for PerformanceTuning {
    fn default() -> Self {
        Self {
            simd_threshold: 8,
            parallel_edge_threshold: 100,
            parallel_feature_threshold: 3,
            batch_size: num_cpus::get(),
        }
    }
}

impl PerformanceTuning {
    /// Creates a new performance tuning configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the SIMD threshold.
    ///
    /// SIMD will only be used when the distance between nodes exceeds this threshold.
    pub fn with_simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_threshold = threshold;
        self
    }

    /// Sets the parallel edge computation threshold.
    ///
    /// Parallel edge computation will only be used for graphs larger than this size.
    pub fn with_parallel_edge_threshold(mut self, threshold: usize) -> Self {
        self.parallel_edge_threshold = threshold;
        self
    }

    /// Sets the parallel feature computation threshold.
    ///
    /// Parallel feature computation will only be used when computing more features than this.
    pub fn with_parallel_feature_threshold(mut self, threshold: usize) -> Self {
        self.parallel_feature_threshold = threshold;
        self
    }

    /// Sets the batch size for parallel processing.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Creates a configuration optimized for small graphs (<100 nodes).
    pub fn for_small_graphs() -> Self {
        Self {
            simd_threshold: 16, // Higher threshold, overhead not worth it
            parallel_edge_threshold: 200,
            parallel_feature_threshold: 5,
            batch_size: num_cpus::get(),
        }
    }

    /// Creates a configuration optimized for large graphs (>1000 nodes).
    pub fn for_large_graphs() -> Self {
        Self {
            simd_threshold: 4, // Lower threshold, maximize SIMD usage
            parallel_edge_threshold: 50,
            parallel_feature_threshold: 2,
            batch_size: num_cpus::get() * 2,
        }
    }

    /// Creates a configuration optimized for power efficiency (e.g., laptops).
    pub fn for_power_efficiency() -> Self {
        Self {
            simd_threshold: 12,
            parallel_edge_threshold: 150,
            parallel_feature_threshold: 4,
            batch_size: num_cpus::get().min(4), // Limit parallelism
        }
    }

    /// Creates a configuration optimized for maximum throughput.
    pub fn for_max_throughput() -> Self {
        Self {
            simd_threshold: 4,
            parallel_edge_threshold: 30,
            parallel_feature_threshold: 1,
            batch_size: num_cpus::get() * 4, // Aggressive parallelism
        }
    }
}

/// System capability detection utilities.
pub struct SystemCapabilities {
    cpu_count: usize,
    has_avx2: bool,
    has_neon: bool,
    architecture: String,
}

impl SystemCapabilities {
    /// Detects system capabilities.
    pub fn detect() -> Self {
        let cpu_count = num_cpus::get();

        let has_avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx2")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        let has_neon = {
            #[cfg(target_arch = "aarch64")]
            {
                std::arch::is_aarch64_feature_detected!("neon")
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                false
            }
        };

        let architecture = {
            #[cfg(target_arch = "x86_64")]
            {
                "x86_64".to_string()
            }
            #[cfg(target_arch = "aarch64")]
            {
                "aarch64".to_string()
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                std::env::consts::ARCH.to_string()
            }
        };

        Self {
            cpu_count,
            has_avx2,
            has_neon,
            architecture,
        }
    }

    /// Returns the number of CPU cores available.
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }

    /// Returns true if AVX2 is available.
    pub fn has_avx2(&self) -> bool {
        self.has_avx2
    }

    /// Returns true if NEON is available.
    pub fn has_neon(&self) -> bool {
        self.has_neon
    }

    /// Returns true if any SIMD capability is available.
    pub fn has_simd(&self) -> bool {
        self.has_avx2 || self.has_neon
    }

    /// Returns the CPU architecture name.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Recommends optimal performance tuning based on capabilities.
    pub fn recommend_tuning(&self, typical_graph_size: usize) -> PerformanceTuning {
        if typical_graph_size < 100 {
            PerformanceTuning::for_small_graphs()
        } else if typical_graph_size > 1000 {
            PerformanceTuning::for_large_graphs()
        } else {
            PerformanceTuning::default()
        }
    }

    /// Prints system capability information.
    pub fn print_info(&self) {
        println!("System Capabilities:");
        println!("  Architecture: {}", self.architecture);
        println!("  CPU Cores: {}", self.cpu_count);
        println!("  SIMD Support:");
        println!("    AVX2: {}", if self.has_avx2 { "✓" } else { "✗" });
        println!("    NEON: {}", if self.has_neon { "✓" } else { "✗" });
        println!("  Recommended: {}", if self.cpu_count >= 8 { "High-performance mode" } else { "Balanced mode" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tuning() {
        let tuning = PerformanceTuning::default();
        assert_eq!(tuning.simd_threshold, 8);
    }

    #[test]
    fn test_custom_tuning() {
        let tuning = PerformanceTuning::new()
            .with_simd_threshold(16)
            .with_parallel_edge_threshold(200);
        assert_eq!(tuning.simd_threshold, 16);
        assert_eq!(tuning.parallel_edge_threshold, 200);
    }

    #[test]
    fn test_system_capabilities() {
        let caps = SystemCapabilities::detect();
        assert!(caps.cpu_count() > 0);
        println!("Detected {} CPUs", caps.cpu_count());
    }
}

