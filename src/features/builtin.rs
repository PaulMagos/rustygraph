//! Built-in feature implementations.
//!
//! This module contains the implementations of all built-in features
//! provided by the library.

use crate::features::{Feature, missing_data::MissingDataHandler};
use std::ops::{Add, Sub, Mul, Div};

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
        let curr = series[index].or_else(|| handler.handle(series, index))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;
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
        let curr = series[index].or_else(|| handler.handle(series, index))?;
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
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
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;
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
pub struct LocalSlopeFeature;

impl<T> Feature<T> for LocalSlopeFeature
where
    T: Copy + Sub<Output = T> + Div<Output = T> + From<f64>,
{
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T> {
        if index == 0 || index >= series.len() - 1 {
            return None;
        }
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;
        Some((next - prev) / T::from(2.0))
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
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
        let curr = series[index].or_else(|| handler.handle(series, index))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;
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

        let mut sum = None;
        let mut count = 0;

        for i in start..end {
            if let Some(val) = series[i].or_else(|| handler.handle(series, i)) {
                sum = Some(sum.map_or(val, |s| s + val));
                count += 1;
            }
        }

        sum.map(|s| s / T::from(count as f64))
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

        let mut values = Vec::new();
        for i in start..end {
            if let Some(val) = series[i].or_else(|| handler.handle(series, i)) {
                values.push(val);
            }
        }

        if values.is_empty() {
            return None;
        }

        let mean = values.iter().fold(None, |acc: Option<T>, &v| {
            Some(acc.map_or(v, |a| a + v))
        })? / T::from(values.len() as f64);

        let variance = values.iter().fold(None, |acc: Option<T>, &v| {
            let diff = v - mean;
            Some(acc.map_or(diff * diff, |a| a + diff * diff))
        })? / T::from(values.len() as f64);

        Some(variance)
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

        let curr = series[index].or_else(|| handler.handle(series, index))?;
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;

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

        let curr = series[index].or_else(|| handler.handle(series, index))?;
        let prev = series[index - 1].or_else(|| handler.handle(series, index - 1))?;
        let next = series[index + 1].or_else(|| handler.handle(series, index + 1))?;

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
        let curr = series[index].or_else(|| handler.handle(series, index))?;

        // Compute mean and std of entire series
        let mut values = Vec::new();
        for i in 0..series.len() {
            if let Some(val) = series[i].or_else(|| handler.handle(series, i)) {
                values.push(val);
            }
        }

        if values.len() < 2 {
            return None;
        }

        let mean = values.iter().fold(None, |acc: Option<T>, &v| {
            Some(acc.map_or(v, |a| a + v))
        })? / T::from(values.len() as f64);

        let variance = values.iter().fold(None, |acc: Option<T>, &v| {
            let diff = v - mean;
            Some(acc.map_or(diff * diff, |a| a + diff * diff))
        })? / T::from(values.len() as f64);

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

