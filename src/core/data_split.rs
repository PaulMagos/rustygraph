//! Data splitting utilities for machine learning.
//!
//! This module provides strategies for splitting time series datasets into
//! training, validation, and test sets while preserving temporal relationships.

use crate::core::{TimeSeriesError};
use crate::WindowedTimeSeries;
use std::collections::HashMap;

/// Strategies for splitting time series data.
#[derive(Debug, Clone, Copy)]
pub enum SplitStrategy {
    /// Split by time: first portion for training, middle for validation, last for testing
    ///
    /// Preserves temporal order but may not represent future patterns well.
    TimeBased {
        /// Fraction of data for training (0.0 to 1.0)
        train_frac: f64,
        /// Fraction of data for validation (0.0 to 1.0)
        val_frac: f64,
        // Test fraction is 1.0 - train_frac - val_frac
    },
    /// Split by series: randomly assign entire series to train/val/test
    ///
    /// Good for multiple independent time series.
    SeriesBased {
        /// Fraction of series for training
        train_frac: f64,
        /// Fraction of series for validation
        val_frac: f64,
        // Test fraction is 1.0 - train_frac - val_frac
    },
    /// Rolling window split: use early windows for training, later for testing
    ///
    /// Good for time series forecasting.
    RollingWindow {
        /// Number of initial windows for training
        train_windows: usize,
        /// Number of windows for validation (taken before test windows)
        val_windows: usize,
        // Remaining windows for testing
    },
}

/// Result of a data split operation.
#[derive(Debug, Clone)]
pub struct DataSplit<T> {
    /// Training data
    pub train: WindowedTimeSeries<T>,
    /// Validation data (optional)
    pub val: Option<WindowedTimeSeries<T>>,
    /// Test data
    pub test: WindowedTimeSeries<T>,
}

/// Splits windowed time series data according to the specified strategy.
pub fn split_windowed_data<T>(
    data: WindowedTimeSeries<T>,
    strategy: SplitStrategy,
) -> Result<DataSplit<T>, TimeSeriesError>
where
    T: Clone + Default,
{
    match strategy {
        SplitStrategy::TimeBased { train_frac, val_frac } => {
            split_time_based(data, train_frac, val_frac)
        }
        SplitStrategy::SeriesBased { train_frac, val_frac } => {
            split_series_based(data, train_frac, val_frac)
        }
        SplitStrategy::RollingWindow { train_windows, val_windows } => {
            split_rolling_window(data, train_windows, val_windows)
        }
    }
}

fn split_time_based<T>(
    data: WindowedTimeSeries<T>,
    train_frac: f64,
    val_frac: f64,
) -> Result<DataSplit<T>, TimeSeriesError>
where
    T: Clone + Default,
{
    let total_windows = data.len();
    let train_end = (total_windows as f64 * train_frac) as usize;
    let val_end = train_end + (total_windows as f64 * val_frac) as usize;

    let (train_windows, rest) = data.windows.split_at(train_end);
    let (val_windows, test_windows) = rest.split_at(val_end - train_end);

    let train = WindowedTimeSeries {
        windows: train_windows.to_vec(),
        labels: data.labels.as_ref().map(|l| l[..train_end].to_vec()),
        series_indices: data.series_indices[..train_end].to_vec(),
        window_starts: data.window_starts[..train_end].to_vec(),
        window_size: data.window_size,
        num_features: data.num_features,
    };

    let val = if !val_windows.is_empty() {
        Some(WindowedTimeSeries {
            windows: val_windows.to_vec(),
            labels: data.labels.as_ref().map(|l| l[train_end..val_end].to_vec()),
            series_indices: data.series_indices[train_end..val_end].to_vec(),
            window_starts: data.window_starts[train_end..val_end].to_vec(),
            window_size: data.window_size,
            num_features: data.num_features,
        })
    } else {
        None
    };

    let test = WindowedTimeSeries {
        windows: test_windows.to_vec(),
        labels: data.labels.as_ref().map(|l| l[val_end..].to_vec()),
        series_indices: data.series_indices[val_end..].to_vec(),
        window_starts: data.window_starts[val_end..].to_vec(),
        window_size: data.window_size,
        num_features: data.num_features,
    };

    Ok(DataSplit { train, val, test })
}

fn split_series_based<T>(
    data: WindowedTimeSeries<T>,
    train_frac: f64,
    val_frac: f64,
) -> Result<DataSplit<T>, TimeSeriesError>
where
    T: Clone + Default,
{
    // Group windows by series
    let mut series_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &series_idx) in data.series_indices.iter().enumerate() {
        series_groups.entry(series_idx).or_default().push(idx);
    }

    let mut series_indices: Vec<usize> = series_groups.keys().cloned().collect();
    // Simple deterministic shuffle based on series index
    series_indices.sort_by_key(|&x| x.wrapping_mul(2654435761) % (2usize.pow(32)));

    let total_series = series_indices.len();
    let train_end = (total_series as f64 * train_frac) as usize;
    let val_end = train_end + (total_series as f64 * val_frac) as usize;

    let train_series = &series_indices[..train_end];
    let val_series = &series_indices[train_end..val_end];
    let test_series = &series_indices[val_end..];

    let train_indices: Vec<usize> = train_series.iter()
        .flat_map(|&s| series_groups[&s].iter().cloned())
        .collect();
    let val_indices: Vec<usize> = val_series.iter()
        .flat_map(|&s| series_groups[&s].iter().cloned())
        .collect();
    let test_indices: Vec<usize> = test_series.iter()
        .flat_map(|&s| series_groups[&s].iter().cloned())
        .collect();

    let train = extract_windows(&data, &train_indices);
    let val = if !val_indices.is_empty() {
        Some(extract_windows(&data, &val_indices))
    } else {
        None
    };
    let test = extract_windows(&data, &test_indices);

    Ok(DataSplit { train, val, test })
}

fn split_rolling_window<T>(
    data: WindowedTimeSeries<T>,
    train_windows: usize,
    val_windows: usize,
) -> Result<DataSplit<T>, TimeSeriesError>
where
    T: Clone + Default,
{
    let total_windows = data.len();
    if train_windows + val_windows >= total_windows {
        return Err(TimeSeriesError::LengthMismatch {
            timestamps: train_windows + val_windows,
            values: total_windows,
        });
    }

    let train_indices: Vec<usize> = (0..train_windows).collect();
    let val_indices: Vec<usize> = (train_windows..train_windows + val_windows).collect();
    let test_indices: Vec<usize> = (train_windows + val_windows..total_windows).collect();

    let train = extract_windows(&data, &train_indices);
    let val = Some(extract_windows(&data, &val_indices));
    let test = extract_windows(&data, &test_indices);

    Ok(DataSplit { train, val, test })
}

fn extract_windows<T>(
    data: &WindowedTimeSeries<T>,
    indices: &[usize],
) -> WindowedTimeSeries<T>
where
    T: Clone + Default,
{
    let windows: Vec<Vec<Vec<T>>> = indices.iter()
        .map(|&i| data.windows[i].clone())
        .collect();

    let labels = data.labels.as_ref().map(|l| {
        indices.iter().map(|&i| l[i].clone()).collect()
    });

    let series_indices: Vec<usize> = indices.iter()
        .map(|&i| data.series_indices[i])
        .collect();

    let window_starts: Vec<usize> = indices.iter()
        .map(|&i| data.window_starts[i])
        .collect();

    WindowedTimeSeries {
        windows,
        labels,
        series_indices,
        window_starts,
        window_size: data.window_size,
        num_features: data.num_features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;

    #[test]
    fn test_time_based_split() {
        let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let windows = WindowedTimeSeries::from_series(&series, 2, 1).unwrap();
        // Series of length 6, window size 2, stride 1 -> 5 windows

        let split = split_windowed_data(windows, SplitStrategy::TimeBased {
            train_frac: 0.5,
            val_frac: 0.25,
        }).unwrap();

        // 5 * 0.5 = 2.5 -> 2 train windows
        // 5 * 0.25 = 1.25 -> 1 val window
        // Remaining 2 test windows
        assert_eq!(split.train.len(), 2);
        assert_eq!(split.val.as_ref().unwrap().len(), 1);
        assert_eq!(split.test.len(), 2); // Fixed: should be 2, not 1
    }

    #[test]
    fn test_rolling_window_split() {
        let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let windows = WindowedTimeSeries::from_series(&series, 2, 1).unwrap();
        // 5 windows total

        let split = split_windowed_data(windows, SplitStrategy::RollingWindow {
            train_windows: 2,
            val_windows: 1,
        }).unwrap();

        // train: first 2, val: next 1, test: remaining 2
        assert_eq!(split.train.len(), 2);
        assert_eq!(split.val.as_ref().unwrap().len(), 1);
        assert_eq!(split.test.len(), 2); // Fixed: should be 2, not 1
    }
}
