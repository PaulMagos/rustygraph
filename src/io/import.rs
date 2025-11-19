//! Time series import from various formats.
//!
//! This module provides functions to import time series data from
//! CSV files and other common formats.

use crate::{TimeSeries, TimeSeriesError};

#[cfg(feature = "csv-import")]
use std::path::Path;
#[cfg(feature = "csv-import")]
use std::fs::File;
#[cfg(feature = "csv-import")]
use std::io::{BufRead, BufReader};

/// Options for CSV import.
#[derive(Debug, Clone)]
pub struct CsvImportOptions {
    /// Whether the CSV has a header row
    pub has_header: bool,
    /// Column index for timestamps (None for auto-generated)
    pub timestamp_column: Option<usize>,
    /// Column index for values
    pub value_column: usize,
    /// Delimiter character
    pub delimiter: char,
    /// String representing missing values
    pub missing_value: String,
}

impl Default for CsvImportOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            timestamp_column: Some(0),
            value_column: 1,
            delimiter: ',',
            missing_value: String::new(),
        }
    }
}

impl<T> TimeSeries<T>
where
    T: std::str::FromStr + Copy + From<f64>,
{
    /// Imports a time series from a CSV file.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the CSV file
    /// - `options`: Import options
    ///
    /// # Returns
    ///
    /// Result containing the time series or an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use rustygraph::{TimeSeries, CsvImportOptions};
    ///
    /// let options = CsvImportOptions {
    ///     has_header: true,
    ///     timestamp_column: Some(0),
    ///     value_column: 1,
    ///     ..Default::default()
    /// };
    ///
    /// let series = TimeSeries::<f64>::from_csv("data.csv", options).unwrap();
    /// ```
    #[cfg(feature = "csv-import")]
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        options: CsvImportOptions,
    ) -> Result<Self, TimeSeriesError> {
        let file = File::open(path).map_err(|_| TimeSeriesError::EmptyData)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip header if present
        if options.has_header {
            lines.next();
        }

        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        for (idx, line) in lines.enumerate() {
            let line = line.map_err(|_| TimeSeriesError::EmptyData)?;
            let parts: Vec<&str> = line.split(options.delimiter).collect();

            // Parse timestamp
            let timestamp = if let Some(ts_col) = options.timestamp_column {
                if ts_col >= parts.len() {
                    return Err(TimeSeriesError::EmptyData);
                }
                parts[ts_col]
                    .parse::<f64>()
                    .map(T::from)
                    .map_err(|_| TimeSeriesError::EmptyData)?
            } else {
                T::from(idx as f64)
            };

            // Parse value
            let value = if options.value_column >= parts.len() {
                return Err(TimeSeriesError::EmptyData);
            } else if parts[options.value_column].trim() == options.missing_value {
                None
            } else {
                Some(
                    parts[options.value_column]
                        .trim()
                        .parse::<f64>()
                        .map(T::from)
                        .map_err(|_| TimeSeriesError::EmptyData)?,
                )
            };

            timestamps.push(timestamp);
            values.push(value);
        }

        if timestamps.is_empty() {
            return Err(TimeSeriesError::EmptyData);
        }

        TimeSeries::new(timestamps, values)
    }

    /// Imports a time series from a CSV string.
    ///
    /// # Arguments
    ///
    /// - `csv_data`: CSV data as a string
    /// - `options`: Import options
    ///
    /// # Returns
    ///
    /// Result containing the time series or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, CsvImportOptions};
    ///
    /// let csv_data = "timestamp,value\n0,1.0\n1,2.0\n2,3.0";
    /// let series = TimeSeries::<f64>::from_csv_string(
    ///     csv_data,
    ///     CsvImportOptions::default()
    /// ).unwrap();
    /// assert_eq!(series.len(), 3);
    /// ```
    pub fn from_csv_string(
        csv_data: &str,
        options: CsvImportOptions,
    ) -> Result<Self, TimeSeriesError> {
        let mut lines = csv_data.lines();

        // Skip header if present
        if options.has_header {
            lines.next();
        }

        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        for (idx, line) in lines.enumerate() {
            let parts: Vec<&str> = line.split(options.delimiter).collect();

            // Parse timestamp
            let timestamp = if let Some(ts_col) = options.timestamp_column {
                if ts_col >= parts.len() {
                    return Err(TimeSeriesError::EmptyData);
                }
                parts[ts_col]
                    .parse::<f64>()
                    .map(T::from)
                    .map_err(|_| TimeSeriesError::EmptyData)?
            } else {
                T::from(idx as f64)
            };

            // Parse value
            let value = if options.value_column >= parts.len() {
                return Err(TimeSeriesError::EmptyData);
            } else if parts[options.value_column].trim() == options.missing_value {
                None
            } else {
                Some(
                    parts[options.value_column]
                        .trim()
                        .parse::<f64>()
                        .map(T::from)
                        .map_err(|_| TimeSeriesError::EmptyData)?,
                )
            };

            timestamps.push(timestamp);
            values.push(value);
        }

        if timestamps.is_empty() {
            return Err(TimeSeriesError::EmptyData);
        }

        TimeSeries::new(timestamps, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_string_import() {
        let csv_data = "timestamp,value\n0,1.0\n1,2.0\n2,3.0";
        let series = TimeSeries::<f64>::from_csv_string(
            csv_data,
            CsvImportOptions::default(),
        )
        .unwrap();

        assert_eq!(series.len(), 3);
    }

    #[test]
    fn test_csv_with_missing() {
        let csv_data = "timestamp,value\n0,1.0\n1,\n2,3.0";
        let options = CsvImportOptions {
            has_header: true,
            timestamp_column: Some(0),
            value_column: 1,
            delimiter: ',',
            missing_value: String::new(),
        };

        let series = TimeSeries::<f64>::from_csv_string(csv_data, options).unwrap();
        assert_eq!(series.len(), 3);
        assert!(series.values[1].is_none());
    }
}

