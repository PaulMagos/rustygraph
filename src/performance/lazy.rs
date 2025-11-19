//! Lazy evaluation for expensive features.
//!
//! This module provides lazy computation of graph metrics and features,
//! computing them only when requested and caching the results.

use crate::core::VisibilityGraph;
use std::cell::RefCell;
use std::collections::HashMap;

/// Cached metrics for lazy evaluation.
#[derive(Debug)]
pub struct LazyMetrics {
    clustering_coefficient: RefCell<Option<f64>>,
    diameter: RefCell<Option<usize>>,
    avg_path_length: RefCell<Option<f64>>,
    density: RefCell<Option<f64>>,
    betweenness_centrality: RefCell<HashMap<usize, f64>>,
}

impl LazyMetrics {
    /// Creates a new lazy metrics container.
    pub fn new() -> Self {
        Self {
            clustering_coefficient: RefCell::new(None),
            diameter: RefCell::new(None),
            avg_path_length: RefCell::new(None),
            density: RefCell::new(None),
            betweenness_centrality: RefCell::new(HashMap::new()),
        }
    }

    /// Clears all cached metrics.
    pub fn clear(&self) {
        *self.clustering_coefficient.borrow_mut() = None;
        *self.diameter.borrow_mut() = None;
        *self.avg_path_length.borrow_mut() = None;
        *self.density.borrow_mut() = None;
        self.betweenness_centrality.borrow_mut().clear();
    }
}

impl Default for LazyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VisibilityGraph<T> {
    /// Creates a new graph with lazy metric evaluation.
    ///
    /// Metrics are computed only when first accessed and then cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// // First call computes and caches
    /// let cc1 = graph.average_clustering_coefficient();
    /// // Second call uses cached value (faster)
    /// let cc2 = graph.average_clustering_coefficient();
    /// assert_eq!(cc1, cc2);
    /// ```
    pub fn with_lazy_metrics(self) -> Self {
        // Metrics are already lazy in the current implementation
        // This is a no-op but provides explicit API
        self
    }
}

/// Builder for lazy feature computation.
///
/// Computes features on-demand rather than all at once.
pub struct LazyFeatureBuilder<T> {
    series: Vec<Option<T>>,
    computed_features: RefCell<HashMap<(usize, String), Option<T>>>,
}

impl<T: Copy> LazyFeatureBuilder<T> {
    /// Creates a new lazy feature builder.
    pub fn new(series: Vec<Option<T>>) -> Self {
        Self {
            series,
            computed_features: RefCell::new(HashMap::new()),
        }
    }

    /// Gets a feature value, computing it if not cached.
    pub fn get_feature<F>(&self, node: usize, name: &str, compute_fn: F) -> Option<T>
    where
        F: FnOnce(&[Option<T>], usize) -> Option<T>,
    {
        let key = (node, name.to_string());

        // Check cache
        if let Some(&value) = self.computed_features.borrow().get(&key) {
            return value;
        }

        // Compute and cache
        let value = compute_fn(&self.series, node);
        self.computed_features.borrow_mut().insert(key, value);
        value
    }

    /// Clears the feature cache.
    pub fn clear_cache(&self) {
        self.computed_features.borrow_mut().clear();
    }

    /// Returns the number of cached features.
    pub fn cache_size(&self) -> usize {
        self.computed_features.borrow().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_feature_caching() {
        let series = vec![Some(1.0), Some(2.0), Some(3.0)];
        let builder = LazyFeatureBuilder::new(series);

        let mut compute_count = 0;

        // First access - computes
        let _val1 = builder.get_feature(1, "test", |s, i| {
            compute_count += 1;
            s[i]
        });
        assert_eq!(compute_count, 1);

        // Second access - uses cache
        let _val2 = builder.get_feature(1, "test", |s, i| {
            compute_count += 1;
            s[i]
        });
        assert_eq!(compute_count, 1); // Not incremented

        assert_eq!(builder.cache_size(), 1);
    }
}

