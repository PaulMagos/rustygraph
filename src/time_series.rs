//! Time series data structures and operations.
//!
//! This module provides the [`TimeSeries`] type for representing time series data,
//! with support for missing values and preprocessing operations.
//!
//! # Examples
//!
//! ```rust
//! use rustygraph::time_series::TimeSeries;
//!
//! // Create from raw values (auto-generates timestamps)
//! let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0, 4.0]);
//!
//! // Create with explicit timestamps
//! let series = TimeSeries::new(
//!     vec![0.0, 0.5, 1.0, 1.5],
//!     vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]
//! ).unwrap();
//! ```

use std::fmt;

/// Time series data container with optional missing value support.
///
/// The `TimeSeries` struct holds temporal data points with associated timestamps.
/// It supports missing values through the `Option<T>` wrapper, allowing for
/// flexible data preprocessing and imputation strategies.
///
/// # Type Parameters
///
/// - `T`: Numeric type that implements `Float` trait (typically `f32` or `f64`)
///
/// # Examples
///
/// ## Creating a time series
///
/// ```rust
/// use rustygraph::TimeSeries;
///
/// // From raw values with auto-generated timestamps
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
/// assert_eq!(series.len(), 5);
///
/// // With explicit timestamps and missing values
/// let series = TimeSeries::new(
///     vec![0.0, 1.0, 2.0, 3.0],
///     vec![Some(1.0), None, Some(3.0), Some(2.0)]
/// ).unwrap();
/// ```
///
/// ## Handling missing data
///
/// ```rust
/// use rustygraph::{TimeSeries, MissingDataStrategy};
///
/// let series = TimeSeries::new(
///     vec![0.0, 1.0, 2.0],
///     vec![Some(1.0), None, Some(3.0)]
/// ).unwrap();
///
/// let cleaned = series.handle_missing(MissingDataStrategy::LinearInterpolation).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    /// Timestamps for each data point
    pub timestamps: Vec<T>,
    /// Data values (None indicates missing data)
    pub values: Vec<Option<T>>,
}

impl<T> TimeSeries<T> {
    /// Creates a time series from raw values with auto-generated timestamps.
    ///
    /// Timestamps are generated as sequential integers starting from 0.
    ///
    /// # Arguments
    ///
    /// - `values`: Vector of data points
    ///
    /// # Returns
    ///
    /// A new `TimeSeries` with sequential integer timestamps (0, 1, 2, ...)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::TimeSeries;
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.timestamps, vec![0.0, 1.0, 2.0]);
    /// ```
    pub fn from_raw(values: Vec<T>) -> Result<Self, TimeSeriesError>  where Vec<T>: FromIterator<usize>{
        if values.is_empty() {
            return Err(TimeSeriesError::EmptyData);
        }
        
        let n = values.len();
        let timestamps: Vec<T> = (0..n).map(|i| i).collect();
        let values: Vec<Option<T>> = values.into_iter().map(Some).collect();
        
        Ok(TimeSeries { timestamps, values })
    }

    /// Creates a time series with explicit timestamps.
    ///
    /// # Arguments
    ///
    /// - `timestamps`: Time points for each value
    /// - `values`: Data values (Some(v) for present, None for missing)
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::LengthMismatch`] if timestamps and values have different lengths.
    /// Returns [`TimeSeriesError::EmptyData`] if both vectors are empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::TimeSeries;
    ///
    /// let series = TimeSeries::new(
    ///     vec![0.0, 1.0, 2.0],
    ///     vec![Some(1.0), None, Some(3.0)]
    /// ).unwrap();
    /// ```
    pub fn new(timestamps: Vec<T>, values: Vec<Option<T>>) -> Result<Self, TimeSeriesError> {
        if timestamps.len() != values.len() {
            return Err(TimeSeriesError::LengthMismatch {
                timestamps: timestamps.len(),
                values: values.len(),
            });
        }
        
        if timestamps.is_empty() {
            return Err(TimeSeriesError::EmptyData);
        }
        
        Ok(TimeSeries { timestamps, values })
    }

    /// Returns the length of the time series.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::TimeSeries;
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(series.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the time series contains no data points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::TimeSeries;
    ///
    /// let series = TimeSeries::from_raw(vec![1.0]);
    /// assert!(!series.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// Placeholder for missing data handling - will be implemented in features module
impl<T> TimeSeries<T> {
    /// Handles missing data using the specified strategy.
    ///
    /// This method creates a new time series with imputed values based on the
    /// chosen strategy. The original series is not modified.
    ///
    /// # Arguments
    ///
    /// - `strategy`: The imputation strategy to apply
    ///
    /// # Returns
    ///
    /// A new `TimeSeries` with imputed values
    ///
    /// # Errors
    ///
    /// Returns an error if imputation fails (e.g., insufficient data for the strategy)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, MissingDataStrategy};
    ///
    /// let series = TimeSeries::new(
    ///     vec![0.0, 1.0, 2.0],
    ///     vec![Some(1.0), None, Some(3.0)]
    /// ).unwrap();
    ///
    /// let cleaned = series.handle_missing(
    ///     MissingDataStrategy::LinearInterpolation
    /// ).unwrap();
    /// ```
    pub fn handle_missing(&self, _strategy: crate::features::missing_data::MissingDataStrategy) 
        -> Result<Self, crate::features::missing_data::ImputationError> 
    where
        T: Clone,
    {
        // Implementation will be provided
        todo!("Missing data handling implementation")
    }
}

/// Errors that can occur during time series operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TimeSeriesError {
    /// Timestamps and values have different lengths
    LengthMismatch {
        /// Number of timestamps
        timestamps: usize,
        /// Number of values
        values: usize,
    },
    /// Empty time series
    EmptyData,
    /// Invalid timestamp order (not monotonically increasing)
    NonMonotonicTimestamps,
}

impl fmt::Display for TimeSeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeSeriesError::LengthMismatch { timestamps, values } => {
                write!(f, "Length mismatch: {} timestamps but {} values", timestamps, values)
            }
            TimeSeriesError::EmptyData => write!(f, "Time series is empty"),
            TimeSeriesError::NonMonotonicTimestamps => {
                write!(f, "Timestamps must be monotonically increasing")
            }
        }
    }
}

impl std::error::Error for TimeSeriesError {}
