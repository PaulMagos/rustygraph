//! Parallel computation utilities.
//!
//! This module provides parallel implementations of compute-intensive
//! operations using the rayon library.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::features::FeatureSet;
use std::collections::HashMap;

/// Computes all features for all nodes in parallel.
///
/// This is a parallel version of the feature computation that can
/// significantly speed up processing for large time series.
///
/// # Arguments
///
/// - `series`: Time series values
/// - `feature_set`: Features to compute
///
/// # Returns
///
/// Vector of feature maps for each node
#[cfg(feature = "parallel")]
pub fn compute_node_features_parallel<T>(
    series: &[Option<T>],
    feature_set: &FeatureSet<T>,
) -> Vec<HashMap<String, T>>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
       + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64>
       + Send + Sync,
{
    (0..series.len())
        .into_par_iter()
        .map(|i| {
            let mut features = HashMap::new();
            
            // Compute each feature for this node
            for feature in &feature_set.features {
                if let Some(value) = feature.compute(series, i, &feature_set.missing_strategy) {
                    features.insert(feature.name().to_string(), value);
                }
            }
            
            features
        })
        .collect()
}

/// Sequential fallback when parallel feature is not enabled.
#[cfg(not(feature = "parallel"))]
pub fn compute_node_features_parallel<T>(
    series: &[Option<T>],
    feature_set: &FeatureSet<T>,
) -> Vec<HashMap<String, T>>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
       + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64>,
{
    // Fall back to sequential computation
    crate::visibility_graph::compute_node_features(series, feature_set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureSet, BuiltinFeature};
    
    #[test]
    fn test_parallel_computation() {
        let series = vec![Some(1.0), Some(2.0), Some(3.0), Some(2.0)];
        let feature_set = FeatureSet::new()
            .add_builtin(BuiltinFeature::DeltaForward);
        
        let features = compute_node_features_parallel(&series, &feature_set);
        assert_eq!(features.len(), 4);
    }
}

