//! Built-in feature implementations.
//!
//! This module contains the implementations of all built-in features
//! provided by the library.

use crate::core::features::{Feature, missing_data::MissingDataHandler};
use std::ops::{Add, Sub, Mul, Div};

/// Helper function to get a value at index with missing data handling.
#[inline]
fn get_value_with_handler<T: Copy>(
    series: &[Option<T>],
    index: usize,
    handler: &dyn MissingDataHandler<T>,
) -> Option<T> {
    series[index].or_else(|| handler.handle(series, index))
}

/// Helper function to compute mean of values.
#[inline]
fn compute_mean<T>(values: &[T]) -> Option<T>
where
    T: Copy + Add<Output = T> + Div<Output = T> + From<f64>,
{
    if values.is_empty() {
        return None;
    }
    let sum = values.iter().fold(None, |acc: Option<T>, &v| {
        Some(acc.map_or(v, |a| a + v))
    })?;
    Some(sum / T::from(values.len() as f64))
}

/// Helper function to compute variance of values given their mean.
#[inline]
fn compute_variance<T>(values: &[T], mean: T) -> Option<T>
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + From<f64>,
{
    if values.is_empty() {
        return None;
    }
    let sum_sq_diff = values.iter().fold(None, |acc: Option<T>, &v| {
        let diff = v - mean;
        Some(acc.map_or(diff * diff, |a| a + diff * diff))
    })?;
    Some(sum_sq_diff / T::from(values.len() as f64))
}

/// Helper function to collect valid values from a window with handler fallback.
#[inline]
fn collect_window_values<T: Copy>(
    series: &[Option<T>],
    start: usize,
    end: usize,
    handler: &dyn MissingDataHandler<T>,
) -> Vec<T> {
    (start..end)
        .filter_map(|i| get_value_with_handler(series, i, handler))
        .collect()
}

/// Forward difference feature: y[i+1] - y[i]
pub struct DeltaForwardFeature;

impl<T> Feature<T> for DeltaForwardFeature
where
    T: Copy + Sub<Output = T>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index >= series.len() - 1 {
            return None;
        }
        let curr = get_value_with_handler(series, index, handler)?;
        let next = get_value_with_handler(series, index + 1, handler)?;
        Some(next - curr)
    }

    fn name(&self) -> &str {
        "delta_forward"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Backward difference feature: y[i] - y[i-1]
pub struct DeltaBackwardFeature;

impl<T> Feature<T> for DeltaBackwardFeature
where
    T: Copy + Sub<Output = T>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 {
            return None;
        }
        let curr = get_value_with_handler(series, index, handler)?;
        let prev = get_value_with_handler(series, index - 1, handler)?;
        Some(curr - prev)
    }

    fn name(&self) -> &str {
        "delta_backward"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Symmetric difference feature: (y[i+1] - y[i-1]) / 2
pub struct DeltaSymmetricFeature;

impl<T> Feature<T> for DeltaSymmetricFeature
where
    T: Copy + Sub<Output = T> + Div<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 || index >= series.len() - 1 {
            return None;
        }
        let prev = get_value_with_handler(series, index - 1, handler)?;
        let next = get_value_with_handler(series, index + 1, handler)?;
        Some((next - prev) / T::from(2.0))
    }

    fn name(&self) -> &str {
        "delta_symmetric"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Local slope feature: (y[i+1] - y[i-1]) / 2 (assuming unit time steps)
///
/// Note: This is mathematically equivalent to DeltaSymmetricFeature.
pub struct LocalSlopeFeature;

impl<T> Feature<T> for LocalSlopeFeature
where
    T: Copy + Sub<Output = T> + Div<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        // Delegate to DeltaSymmetricFeature as they compute the same thing
        DeltaSymmetricFeature.compute(series, index, handler)
    }

    fn name(&self) -> &str {
        "local_slope"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Second derivative approximation feature
pub struct AccelerationFeature;

impl<T> Feature<T> for AccelerationFeature
where
    T: Copy + Sub<Output = T> + Add<Output = T> + Mul<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 || index >= series.len() - 1 {
            return None;
        }
        let prev = get_value_with_handler(series, index - 1, handler)?;
        let curr = get_value_with_handler(series, index, handler)?;
        let next = get_value_with_handler(series, index + 1, handler)?;
        // Second derivative: (y[i+1] - 2*y[i] + y[i-1])
        Some(next - curr * T::from(2.0) + prev)
    }

    fn name(&self) -> &str {
        "acceleration"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Local mean feature over a window
pub struct LocalMeanFeature {
    /// Size of the window around the point
    pub window_size: usize,
}

impl<T> Feature<T> for LocalMeanFeature
where
    T: Copy + Add<Output = T> + Div<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        let start = index.saturating_sub(self.window_size / 2);
        let end = (index + self.window_size / 2 + 1).min(series.len());

        let values = collect_window_values(series, start, end, handler);
        compute_mean(&values)
    }

    fn name(&self) -> &str {
        "local_mean"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }

    fn window_size(&self) -> Option<usize> {
        Some(self.window_size)
    }
}

/// Local variance feature over a window
pub struct LocalVarianceFeature {
    /// Size of the window around the point
    pub window_size: usize,
}

impl<T> Feature<T> for LocalVarianceFeature
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        let start = index.saturating_sub(self.window_size / 2);
        let end = (index + self.window_size / 2 + 1).min(series.len());

        let values = collect_window_values(series, start, end, handler);
        let mean = compute_mean(&values)?;
        compute_variance(&values, mean)
    }

    fn name(&self) -> &str {
        "local_variance"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }

    fn window_size(&self) -> Option<usize> {
        Some(self.window_size)
    }
}

/// Peak detection feature (returns 1.0 for local maxima, 0.0 otherwise)
pub struct IsLocalMaxFeature;

impl<T> Feature<T> for IsLocalMaxFeature
where
    T: Copy + PartialOrd + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 || index >= series.len() - 1 {
            return Some(T::from(0.0));
        }

        let curr = get_value_with_handler(series, index, handler)?;
        let prev = get_value_with_handler(series, index - 1, handler)?;
        let next = get_value_with_handler(series, index + 1, handler)?;

        if curr > prev && curr > next {
            Some(T::from(1.0))
        } else {
            Some(T::from(0.0))
        }
    }

    fn name(&self) -> &str {
        "is_local_max"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Valley detection feature (returns 1.0 for local minima, 0.0 otherwise)
pub struct IsLocalMinFeature;

impl<T> Feature<T> for IsLocalMinFeature
where
    T: Copy + PartialOrd + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 || index >= series.len() - 1 {
            return Some(T::from(0.0));
        }

        let curr = get_value_with_handler(series, index, handler)?;
        let prev = get_value_with_handler(series, index - 1, handler)?;
        let next = get_value_with_handler(series, index + 1, handler)?;

        if curr < prev && curr < next {
            Some(T::from(1.0))
        } else {
            Some(T::from(0.0))
        }
    }

    fn name(&self) -> &str {
        "is_local_min"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Z-score normalization feature: (y[i] - mean) / std
pub struct ZScoreFeature;

impl<T> Feature<T> for ZScoreFeature
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<f64> + Into<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        let curr = get_value_with_handler(series, index, handler)?;

        // Collect all valid values from entire series
        let values = collect_window_values(series, 0, series.len(), handler);

        if values.len() < 2 {
            return None;
        }

        let mean = compute_mean(&values)?;
        let variance = compute_variance(&values, mean)?;

        let std_val: f64 = variance.into();
        let std = T::from(std_val.sqrt());

        if std_val > 1e-10 {
            Some((curr - mean) / std)
        } else {
            Some(T::from(0.0))
        }
    }

    fn name(&self) -> &str {
        "z_score"
    }

    fn requires_neighbors(&self) -> bool {
        true
    }
}

/// Enumeration of all built-in features.
#[derive(Debug, Clone, Copy)]
pub enum BuiltinFeature {
    /// Forward difference: y[i+1] - y[i]
    DeltaForward,
    /// Backward difference: y[i] - y[i-1]
    DeltaBackward,
    /// Symmetric difference: (y[i+1] - y[i-1]) / 2
    DeltaSymmetric,
    /// Local slope
    LocalSlope,
    /// Second derivative (acceleration)
    Acceleration,
    /// Local mean with window size
    LocalMean,
    /// Local variance with window size
    LocalVariance,
    /// Is local maximum
    IsLocalMax,
    /// Is local minimum
    IsLocalMin,
    /// Z-score normalization
    ZScore,
}
impl<T> Feature<T> for BuiltinFeature
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<f64> + Into<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        match self {
            BuiltinFeature::DeltaForward => DeltaForwardFeature.compute(series, index, handler),
            BuiltinFeature::DeltaBackward => DeltaBackwardFeature.compute(series, index, handler),
            BuiltinFeature::DeltaSymmetric => DeltaSymmetricFeature.compute(series, index, handler),
            BuiltinFeature::LocalSlope => LocalSlopeFeature.compute(series, index, handler),
            BuiltinFeature::Acceleration => AccelerationFeature.compute(series, index, handler),
            BuiltinFeature::LocalMean => LocalMeanFeature { window_size: 3 }.compute(series, index, handler),
            BuiltinFeature::LocalVariance => LocalVarianceFeature { window_size: 3 }.compute(series, index, handler),
            BuiltinFeature::IsLocalMax => IsLocalMaxFeature.compute(series, index, handler),
            BuiltinFeature::IsLocalMin => IsLocalMinFeature.compute(series, index, handler),
            BuiltinFeature::ZScore => ZScoreFeature.compute(series, index, handler),
        }
    }
    fn name(&self) -> &str {
        match self {
            BuiltinFeature::DeltaForward => "delta_forward",
            BuiltinFeature::DeltaBackward => "delta_backward",
            BuiltinFeature::DeltaSymmetric => "delta_symmetric",
            BuiltinFeature::LocalSlope => "local_slope",
            BuiltinFeature::Acceleration => "acceleration",
            BuiltinFeature::LocalMean => "local_mean",
            BuiltinFeature::LocalVariance => "local_variance",
            BuiltinFeature::IsLocalMax => "is_local_max",
            BuiltinFeature::IsLocalMin => "is_local_min",
            BuiltinFeature::ZScore => "z_score",
        }
    }
    fn requires_neighbors(&self) -> bool {
        !matches!(self, BuiltinFeature::ZScore)
    }
}
