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
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
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
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.timestamps, vec![0.0, 1.0, 2.0]);
    /// ```
    pub fn from_raw(values: Vec<T>) -> Result<Self, TimeSeriesError>
    where
        T: Copy + From<f64>,
    {
        if values.is_empty() {
            return Err(TimeSeriesError::EmptyData);
        }
        
        let n = values.len();
        let timestamps: Vec<T> = (0..n).map(|i| T::from(i as f64)).collect();
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
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
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
    /// let series = TimeSeries::from_raw(vec![1.0]).unwrap();
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
    pub fn handle_missing(&self, strategy: crate::core::features::missing_data::MissingDataStrategy)
        -> Result<Self, crate::core::features::missing_data::ImputationError>
    where
        T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<f64>,
    {
        use crate::core::features::missing_data::{MissingDataHandler, ImputationError};

        let mut new_values = Vec::with_capacity(self.values.len());
        
        for i in 0..self.values.len() {
            let value = if let Some(v) = self.values[i] {
                Some(v)
            } else {
                // Try to impute the missing value
                Some(strategy.handle(&self.values, i)
                    .ok_or(ImputationError::AllStrategiesFailed { index: i })?)
            };
            new_values.push(value);
        }
        
        Ok(TimeSeries {
            timestamps: self.timestamps.clone(),
            values: new_values,
        })
    }
}

/// Windowed time series data for machine learning.
///
/// Represents a collection of time series windows extracted from one or more
/// time series, suitable for ML training and inference.
///
/// # Examples
///
/// ```rust
/// use rustygraph::{TimeSeries, WindowedTimeSeries};
///
/// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let windows = WindowedTimeSeries::from_series(&series, 3, 1).unwrap();
/// // Creates windows: [1,2,3], [2,3,4], [3,4,5]
/// ```
#[derive(Debug, Clone)]
pub struct WindowedTimeSeries<T> {
    /// Window data: Vec of windows, each window is [feature][timestep]
    pub windows: Vec<Vec<Vec<T>>>,
    /// Optional labels/targets for supervised learning
    pub labels: Option<Vec<T>>,
    /// Original series indices for traceability
    pub series_indices: Vec<usize>,
    /// Window start positions in original series
    pub window_starts: Vec<usize>,
    /// Window size (timesteps per window)
    pub window_size: usize,
    /// Number of features per window
    pub num_features: usize,
}

impl<T> WindowedTimeSeries<T> {
    /// Creates windows from a single time series using sliding window approach.
    ///
    /// # Arguments
    ///
    /// - `series`: Source time series
    /// - `window_size`: Number of timesteps per window
    /// - `stride`: Step size between windows (1 for sliding, window_size for non-overlapping)
    ///
    /// # Returns
    ///
    /// WindowedTimeSeries with extracted windows
    ///
    /// # Errors
    ///
    /// Returns error if series is too short for requested windows
    pub fn from_series(series: &TimeSeries<T>, window_size: usize, stride: usize)
        -> Result<Self, TimeSeriesError>
    where
        T: Copy + Default,
    {
        if series.len() < window_size {
            return Err(TimeSeriesError::EmptyData);
        }

        let num_windows = ((series.len() - window_size) / stride) + 1;
        let mut windows = Vec::with_capacity(num_windows);
        let mut series_indices = Vec::with_capacity(num_windows);
        let mut window_starts = Vec::with_capacity(num_windows);

        for i in 0..num_windows {
            let start_idx = i * stride;
            series_indices.push(0); // Single series
            window_starts.push(start_idx);

            // Extract window data (single feature)
            let mut window = vec![vec![]; 1]; // 1 feature
            window[0] = (start_idx..start_idx + window_size)
                .filter_map(|idx| series.values[idx])
                .collect();

            // Pad with zeros if window has missing values
            while window[0].len() < window_size {
                window[0].push(T::default());
            }

            windows.push(window);
        }

        Ok(WindowedTimeSeries {
            windows,
            labels: None,
            series_indices,
            window_starts,
            window_size,
            num_features: 1,
        })
    }

    /// Creates windows from multiple time series.
    ///
    /// # Arguments
    ///
    /// - `series_vec`: Vector of time series
    /// - `window_size`: Number of timesteps per window
    /// - `stride`: Step size between windows
    ///
    /// # Returns
    ///
    /// WindowedTimeSeries with windows from all series
    pub fn from_multiple_series(series_vec: &[TimeSeries<T>], window_size: usize, stride: usize)
        -> Result<Self, TimeSeriesError>
    where
        T: Copy + Default,
    {
        let mut all_windows = Vec::new();
        let mut all_series_indices = Vec::new();
        let mut all_window_starts = Vec::new();

        for (series_idx, series) in series_vec.iter().enumerate() {
            if series.len() < window_size {
                continue; // Skip series that are too short
            }

            let num_windows = ((series.len() - window_size) / stride) + 1;

            for i in 0..num_windows {
                let start_idx = i * stride;
                all_series_indices.push(series_idx);
                all_window_starts.push(start_idx);

                // Extract window data (single feature)
                let mut window = vec![vec![]; 1]; // 1 feature
                window[0] = (start_idx..start_idx + window_size)
                    .filter_map(|idx| series.values[idx])
                    .collect();

                // Pad with zeros if window has missing values
                while window[0].len() < window_size {
                    window[0].push(T::default());
                }

                all_windows.push(window);
            }
        }

        Ok(WindowedTimeSeries {
            windows: all_windows,
            labels: None,
            series_indices: all_series_indices,
            window_starts: all_window_starts,
            window_size,
            num_features: 1,
        })
    }

    /// Returns the number of windows.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Returns true if no windows exist.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Gets a specific window by index.
    pub fn get_window(&self, index: usize) -> Option<&Vec<Vec<T>>> {
        self.windows.get(index)
    }

    /// Sets labels for supervised learning.
    pub fn with_labels(mut self, labels: Vec<T>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Converts to a flat 3D array format suitable for ML: [N, F, W]
    pub fn to_array(&self) -> Vec<Vec<Vec<T>>>
    where
        T: Clone,
    {
        self.windows.clone()
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
