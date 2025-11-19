//! Missing data handling strategies.
//!
//! This module provides tools for handling missing values in time series data
//! through various imputation strategies.
//!
//! # Built-in Strategies
//!
//! - **LinearInterpolation**: Average of neighboring valid values
//! - **ForwardFill**: Use last valid value
//! - **BackwardFill**: Use next valid value
//! - **NearestNeighbor**: Use closest valid value
//! - **MeanImputation**: Local window mean
//! - **MedianImputation**: Local window median
//! - **ZeroFill**: Replace with zero
//! - **Drop**: Skip missing values
//!
//! # Fallback Chains
//!
//! Strategies can be chained with fallbacks for robustness:
//!
//! ```rust
//! use rustygraph::MissingDataStrategy;
//!
//! let strategy = MissingDataStrategy::LinearInterpolation
//!     .with_fallback(MissingDataStrategy::ForwardFill);
//! ```

use std::fmt;

/// Trait for handling missing data in time series.
///
/// Implement this trait to create custom imputation strategies that can be
/// used throughout the feature computation pipeline.
///
/// # Examples
///
/// ```rust
/// use rustygraph::features::missing_data::MissingDataHandler;
///
/// struct CustomHandler;
///
/// impl MissingDataHandler<f64> for CustomHandler {
///     fn handle(&self, series: &[Option<f64>], index: usize) -> Option<f64> {
///         // Custom imputation logic
///         if index > 0 && index < series.len() - 1 {
///             let prev = series[index - 1]?;
///             let next = series[index + 1]?;
///             Some((prev + next) / 2.0)
///         } else {
///             None
///         }
///     }
/// }
/// ```
pub trait MissingDataHandler<T>: Send + Sync {
    /// Handles a missing value at the specified index.
    ///
    /// # Arguments
    ///
    /// - `series`: Time series with possible missing values
    /// - `index`: Index of the missing value to impute
    ///
    /// # Returns
    ///
    /// Imputed value, or `None` if imputation is not possible
    fn handle(&self, series: &[Option<T>], index: usize) -> Option<T>;

    /// Indicates if the handler needs neighboring values.
    ///
    /// Return `false` for strategies that only need the local value (e.g., constant fill).
    fn requires_context(&self) -> bool {
        true
    }

    /// Returns the window size needed for context (if any).
    ///
    /// For window-based strategies like mean/median imputation.
    fn window_size(&self) -> Option<usize> {
        None
    }
}

/// Pre-defined strategies for handling missing data.
///
/// Each variant represents a different approach to imputing missing values.
/// Strategies can be chained using [`with_fallback`](MissingDataStrategy::with_fallback)
/// to create robust pipelines.
///
/// # Examples
///
/// ## Simple strategy
///
/// ```rust
/// use rustygraph::MissingDataStrategy;
///
/// let strategy = MissingDataStrategy::LinearInterpolation;
/// ```
///
/// ## With fallback
///
/// ```rust
/// use rustygraph::MissingDataStrategy;
///
/// let strategy = MissingDataStrategy::LinearInterpolation
///     .with_fallback(MissingDataStrategy::ForwardFill)
///     .with_fallback(MissingDataStrategy::ZeroFill);
/// ```
#[derive(Debug, Clone)]
pub enum MissingDataStrategy {
    /// Average of previous and next valid values: `(prev + next) / 2`
    ///
    /// Best for smooth data with occasional missing points.
    /// Fails at boundaries or when neighbors are also missing.
    LinearInterpolation,

    /// Use the last valid value (carry forward)
    ///
    /// Simple and fast. Works well when values change slowly.
    /// Fails for missing values at the start of the series.
    ForwardFill,

    /// Use the next valid value (carry backward)
    ///
    /// Mirror of ForwardFill. Fails for missing values at the end.
    BackwardFill,

    /// Use the closest valid value (by index distance)
    ///
    /// Chooses between previous and next based on proximity.
    NearestNeighbor,

    /// Use mean of local window
    ///
    /// Computes the average of valid values in a neighborhood.
    /// Good for noisy data with local patterns.
    MeanImputation {
        /// Size of the window around the missing point
        window_size: usize,
    },

    /// Use median of local window
    ///
    /// More robust to outliers than mean imputation.
    /// Requires at least one valid value in the window.
    MedianImputation {
        /// Size of the window around the missing point
        window_size: usize,
    },

    /// Replace with zero
    ///
    /// Simple but may introduce bias. Useful when zero is meaningful.
    ZeroFill,

    /// Skip missing values (return None)
    ///
    /// No imputation. Features will return `None` for missing points.
    Drop,

    /// Chain multiple strategies (try first, fallback to second)
    ///
    /// Use this to create robust imputation pipelines that handle edge cases.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::MissingDataStrategy;
    ///
    /// let strategy = MissingDataStrategy::Fallback {
    ///     primary: Box::new(MissingDataStrategy::LinearInterpolation),
    ///     fallback: Box::new(MissingDataStrategy::ForwardFill),
    /// };
    /// ```
    Fallback {
        /// Primary strategy to try first
        primary: Box<MissingDataStrategy>,
        /// Fallback strategy if primary fails
        fallback: Box<MissingDataStrategy>,
    },
}

impl MissingDataStrategy {
    /// Creates a fallback chain of strategies.
    ///
    /// The primary strategy is tried first. If it returns `None`, the fallback
    /// strategy is attempted. Chains can be arbitrarily long.
    ///
    /// # Arguments
    ///
    /// - `fallback`: Strategy to use if primary fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::MissingDataStrategy;
    ///
    /// // Try interpolation, then forward fill, then zero
    /// let strategy = MissingDataStrategy::LinearInterpolation
    ///     .with_fallback(MissingDataStrategy::ForwardFill)
    ///     .with_fallback(MissingDataStrategy::ZeroFill);
    /// ```
    pub fn with_fallback(self, fallback: MissingDataStrategy) -> Self {
        MissingDataStrategy::Fallback {
            primary: Box::new(self),
            fallback: Box::new(fallback),
        }
    }
}

/// Errors during missing data handling.
///
/// These errors indicate that imputation was not possible with the given
/// strategy and data.
#[derive(Debug, Clone, PartialEq)]
pub enum ImputationError {
    /// Cannot impute at boundaries with insufficient context
    ///
    /// Occurs when a strategy needs neighbors but the missing point is
    /// at the start or end of the series.
    BoundaryImpossible {
        /// Index of the boundary point
        index: usize,
    },

    /// No valid values found in imputation window
    ///
    /// Occurs when all values in the required window are also missing.
    NoValidValues {
        /// Index of the missing point
        index: usize,
        /// Size of the window that was searched
        window: usize,
    },

    /// All strategies in fallback chain failed
    ///
    /// Occurs when even the final fallback strategy cannot impute.
    AllStrategiesFailed {
        /// Index where imputation failed
        index: usize,
    },
}

impl fmt::Display for ImputationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImputationError::BoundaryImpossible { index } => {
                write!(f, "Cannot impute at boundary index {}", index)
            }
            ImputationError::NoValidValues { index, window } => {
                write!(
                    f,
                    "No valid values found in window of size {} around index {}",
                    window, index
                )
            }
            ImputationError::AllStrategiesFailed { index } => {
                write!(f, "All imputation strategies failed at index {}", index)
            }
        }
    }
}

impl std::error::Error for ImputationError {}

// Implement MissingDataHandler for MissingDataStrategy
impl<T> MissingDataHandler<T> for MissingDataStrategy
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<f64>,
{
    fn handle(&self, series: &[Option<T>], index: usize) -> Option<T> {
        match self {
            MissingDataStrategy::LinearInterpolation => {
                if index == 0 || index >= series.len() - 1 {
                    return None;
                }

                // Find previous valid value
                let mut prev_val = None;
                for i in (0..index).rev() {
                    if let Some(v) = series[i] {
                        prev_val = Some(v);
                        break;
                    }
                }

                // Find next valid value
                let next_val = series[(index + 1)..]
                    .iter()
                    .find_map(|&v| v);

                match (prev_val, next_val) {
                    (Some(p), Some(n)) => Some((p + n) / T::from(2.0)),
                    _ => None,
                }
            }

            MissingDataStrategy::ForwardFill => {
                // Find previous valid value
                for i in (0..index).rev() {
                    if let Some(v) = series[i] {
                        return Some(v);
                    }
                }
                None
            }

            MissingDataStrategy::BackwardFill => {
                // Find next valid value
                series[(index + 1)..]
                    .iter()
                    .find_map(|&v| v)
            }

            MissingDataStrategy::NearestNeighbor => {
                let mut prev_val = None;
                let mut prev_dist = usize::MAX;
                for i in (0..index).rev() {
                    if let Some(v) = series[i] {
                        prev_val = Some(v);
                        prev_dist = index - i;
                        break;
                    }
                }

                let mut next_val = None;
                let mut next_dist = usize::MAX;
                for (offset, &val) in series[(index + 1)..].iter().enumerate() {
                    if let Some(v) = val {
                        next_val = Some(v);
                        next_dist = offset + 1;
                        break;
                    }
                }

                match (prev_val, next_val) {
                    (Some(p), Some(n)) => {
                        if prev_dist <= next_dist {
                            Some(p)
                        } else {
                            Some(n)
                        }
                    }
                    (Some(p), None) => Some(p),
                    (None, Some(n)) => Some(n),
                    (None, None) => None,
                }
            }

            MissingDataStrategy::MeanImputation { window_size } => {
                let start = index.saturating_sub(window_size / 2);
                let end = (index + window_size / 2 + 1).min(series.len());

                let mut sum = None;
                let mut count = 0;

                for (i, &val) in series.iter().enumerate().take(end).skip(start) {
                    if i != index {
                        if let Some(v) = val {
                            sum = Some(sum.map_or(v, |s| s + v));
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    sum.map(|s| s / T::from(count as f64))
                } else {
                    None
                }
            }

            MissingDataStrategy::MedianImputation { window_size } => {
                let start = index.saturating_sub(window_size / 2);
                let end = (index + window_size / 2 + 1).min(series.len());

                let mut values: Vec<T> = Vec::new();
                for (i, &val) in series.iter().enumerate().take(end).skip(start) {
                    if i != index {
                        if let Some(v) = val {
                            values.push(v);
                        }
                    }
                }

                if values.is_empty() {
                    return None;
                }

                values.sort_by(|a, b| {
                    if a < b {
                        std::cmp::Ordering::Less
                    } else if a > b {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                });

                let mid = values.len() / 2;
                if values.len().is_multiple_of(2) {
                    Some((values[mid - 1] + values[mid]) / T::from(2.0))
                } else {
                    Some(values[mid])
                }
            }

            MissingDataStrategy::ZeroFill => Some(T::from(0.0)),

            MissingDataStrategy::Drop => None,

            MissingDataStrategy::Fallback { primary, fallback } => {
                primary.handle(series, index).or_else(|| fallback.handle(series, index))
            }
        }
    }
}

