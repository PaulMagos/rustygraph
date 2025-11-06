//! Built-in feature implementations.
//!
//! This module contains the implementations of all built-in features
//! provided by the library.

// Note: Imports will be needed when implementations are added
// use crate::features::Feature;
// use crate::features::missing_data::MissingDataHandler;

/// Forward difference feature: y[i+1] - y[i]
pub struct DeltaForwardFeature;

/// Backward difference feature: y[i] - y[i-1]
pub struct DeltaBackwardFeature;

/// Symmetric difference feature: (y[i+1] - y[i-1]) / 2
pub struct DeltaSymmetricFeature;

/// Local slope feature: (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
pub struct LocalSlopeFeature;

/// Second derivative approximation feature
pub struct AccelerationFeature;

/// Local mean feature over a window
pub struct LocalMeanFeature;

/// Local variance feature over a window
pub struct LocalVarianceFeature;

/// Peak detection feature (returns 1.0 for local maxima, 0.0 otherwise)
pub struct IsLocalMaxFeature;

/// Valley detection feature (returns 1.0 for local minima, 0.0 otherwise)
pub struct IsLocalMinFeature;

/// Z-score normalization feature: (y[i] - mean) / std
pub struct ZScoreFeature;

// Note: Full implementations would go here
// For documentation purposes, we're showing the structure

