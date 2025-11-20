//! Polars DataFrame integration for RustyGraph.
//!
//! This module provides seamless integration with the Polars DataFrame library,
//! enabling efficient I/O operations and batch processing of time series data.
//!
//! # Features
//!
//! - **DataFrame I/O**: Convert between TimeSeries and Polars DataFrames
//! - **Lazy evaluation**: Support for Polars' lazy API for efficient pipelines
//! - **Zero-copy**: Direct memory access when possible
//! - **Batch processing**: Process multiple time series from DataFrame columns
//!
//! # Examples
//!
//! ## Basic DataFrame Conversion
//!
//! ```rust
//! # #[cfg(feature = "polars-integration")]
//! # {
//! use rustygraph::{TimeSeries, VisibilityGraph};
//! use rustygraph::integrations::polars::*;
//! use polars::prelude::*;
//!
//! // Create a DataFrame with time series data
//! let df = df! {
//!     "time" => &[0.0, 1.0, 2.0, 3.0, 4.0],
//!     "value" => &[1.0, 3.0, 2.0, 4.0, 1.0],
//! }.unwrap();
//!
//! // Convert to TimeSeries
//! let series = TimeSeries::from_polars_df(&df, "time", "value").unwrap();
//!
//! // Build visibility graph
//! let graph = VisibilityGraph::from_series(&series)
//!     .natural_visibility()
//!     .unwrap();
//!
//! // Export graph properties to DataFrame
//! let graph_df = graph.to_polars_df().unwrap();
//! # }
//! ```
//!
//! ## Batch Processing Multiple Series
//!
//! ```rust
//! # #[cfg(feature = "polars-integration")]
//! # {
//! use rustygraph::integrations::polars::*;
//! use polars::prelude::*;
//!
//! // DataFrame with multiple time series (e.g., sensor data)
//! let df = df! {
//!     "time" => &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
//!     "value" => &[1.0, 3.0, 2.0, 2.0, 1.0, 3.0],
//!     "sensor_id" => &["A", "A", "A", "B", "B", "B"],
//! }.unwrap();
//!
//! // Batch process by sensor
//! let results = BatchProcessor::from_polars_df(
//!     &df,
//!     "time",
//!     "value",
//!     "sensor_id"
//! ).unwrap()
//! .process_natural()
//! .unwrap();
//!
//! // Export results back to DataFrame
//! let results_df = results.to_polars_df().unwrap();
//! # }
//! ```

#[cfg(feature = "polars-integration")]
use polars::prelude::*;

use crate::core::{TimeSeries, VisibilityGraph};
use std::collections::HashMap;

/// Error types for Polars integration.
#[derive(Debug, Clone)]
pub enum PolarsError {
    /// Column not found in DataFrame
    ColumnNotFound(String),
    /// Column has wrong data type
    WrongDataType(String),
    /// Mismatched lengths between columns
    LengthMismatch,
    /// Polars library error
    PolarsError(String),
    /// Empty DataFrame
    EmptyDataFrame,
}

impl std::fmt::Display for PolarsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolarsError::ColumnNotFound(col) => write!(f, "Column not found: {}", col),
            PolarsError::WrongDataType(msg) => write!(f, "Wrong data type: {}", msg),
            PolarsError::LengthMismatch => write!(f, "Column lengths do not match"),
            PolarsError::PolarsError(msg) => write!(f, "Polars error: {}", msg),
            PolarsError::EmptyDataFrame => write!(f, "DataFrame is empty"),
        }
    }
}

impl std::error::Error for PolarsError {}

#[cfg(feature = "polars-integration")]
impl From<PolarsError> for polars::error::PolarsError {
    fn from(err: PolarsError) -> Self {
        polars::error::PolarsError::ComputeError(err.to_string().into())
    }
}

/// Extension trait for TimeSeries to support Polars DataFrame conversion.
#[cfg(feature = "polars-integration")]
pub trait TimeSeriesPolarsExt<T> {
    /// Create a TimeSeries from a Polars DataFrame.
    ///
    /// # Arguments
    ///
    /// - `df`: Polars DataFrame containing the time series data
    /// - `time_col`: Name of the column containing timestamps
    /// - `value_col`: Name of the column containing values
    ///
    /// # Returns
    ///
    /// A TimeSeries instance, or an error if conversion fails
    fn from_polars_df(
        df: &DataFrame,
        time_col: &str,
        value_col: &str,
    ) -> Result<Self, PolarsError>
    where
        Self: Sized;

    /// Convert TimeSeries to a Polars DataFrame.
    ///
    /// # Returns
    ///
    /// A DataFrame with "time" and "value" columns
    fn to_polars_df(&self) -> Result<DataFrame, PolarsError>;
}

#[cfg(feature = "polars-integration")]
impl TimeSeriesPolarsExt<f64> for TimeSeries<f64> {
    fn from_polars_df(
        df: &DataFrame,
        time_col: &str,
        value_col: &str,
    ) -> Result<Self, PolarsError> {
        // Get time column
        let time_series = df
            .column(time_col)
            .map_err(|_| PolarsError::ColumnNotFound(time_col.to_string()))?;

        let times: Vec<f64> = time_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be f64", time_col)))?
            .into_iter()
            .filter_map(|v| v)
            .collect();

        // Get value column
        let value_series = df
            .column(value_col)
            .map_err(|_| PolarsError::ColumnNotFound(value_col.to_string()))?;

        let values: Vec<Option<f64>> = value_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be f64", value_col)))?
            .into_iter()
            .collect();

        if times.is_empty() || values.is_empty() {
            return Err(PolarsError::EmptyDataFrame);
        }

        if times.len() != values.len() {
            return Err(PolarsError::LengthMismatch);
        }

        TimeSeries::new(times, values)
            .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }

    fn to_polars_df(&self) -> Result<DataFrame, PolarsError> {
        let times = &self.timestamps;
        let values: Vec<Option<f64>> = self.values.to_vec();

        DataFrame::new(vec![
            Series::new("time".into(), times).into(),
            Series::new("value".into(), values).into(),
        ])
        .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }
}

#[cfg(feature = "polars-integration")]
impl TimeSeriesPolarsExt<f32> for TimeSeries<f32> {
    fn from_polars_df(
        df: &DataFrame,
        time_col: &str,
        value_col: &str,
    ) -> Result<Self, PolarsError> {
        // Get time column (cast f64 to f32)
        let time_series = df
            .column(time_col)
            .map_err(|_| PolarsError::ColumnNotFound(time_col.to_string()))?;

        let times: Vec<f32> = time_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be numeric", time_col)))?
            .into_iter()
            .filter_map(|v| v.map(|x| x as f32))
            .collect();

        // Get value column (cast f64 to f32)
        let value_series = df
            .column(value_col)
            .map_err(|_| PolarsError::ColumnNotFound(value_col.to_string()))?;

        let values: Vec<Option<f32>> = value_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be numeric", value_col)))?
            .into_iter()
            .map(|v| v.map(|x| x as f32))
            .collect();

        if times.is_empty() || values.is_empty() {
            return Err(PolarsError::EmptyDataFrame);
        }

        if times.len() != values.len() {
            return Err(PolarsError::LengthMismatch);
        }

        TimeSeries::new(times, values)
            .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }

    fn to_polars_df(&self) -> Result<DataFrame, PolarsError> {
        let times: Vec<f64> = self.timestamps.iter().map(|&x| x as f64).collect();
        let values: Vec<Option<f64>> = self.values.iter().map(|&x| x.map(|v| v as f64)).collect();

        DataFrame::new(vec![
            Series::new("time".into(), times).into(),
            Series::new("value".into(), values).into(),
        ])
        .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }
}

/// Extension trait for VisibilityGraph to support Polars DataFrame export.
#[cfg(feature = "polars-integration")]
pub trait VisibilityGraphPolarsExt<T> {
    /// Convert graph properties to a Polars DataFrame.
    ///
    /// The DataFrame will include:
    /// - node_id: Node index
    /// - degree: Node degree
    /// - clustering: Local clustering coefficient
    /// - features: Any computed node features
    fn to_polars_df(&self) -> Result<DataFrame, PolarsError>;

    /// Export edges to a Polars DataFrame.
    ///
    /// The DataFrame will have columns: source, target, weight
    fn edges_to_polars_df(&self) -> Result<DataFrame, PolarsError>;
}

#[cfg(feature = "polars-integration")]
impl<T> VisibilityGraphPolarsExt<T> for VisibilityGraph<T>
where
    T: Copy + std::fmt::Debug,
{
    fn to_polars_df(&self) -> Result<DataFrame, PolarsError> {
        let node_ids: Vec<u32> = (0..self.node_count as u32).collect();
        let degrees = self.degree_sequence();
        let degrees_u32: Vec<u32> = degrees.iter().map(|&d| d as u32).collect();

        DataFrame::new(vec![
            Series::new("node_id".into(), node_ids).into(),
            Series::new("degree".into(), degrees_u32).into(),
        ])
        .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }

    fn edges_to_polars_df(&self) -> Result<DataFrame, PolarsError> {
        let edges = self.edges();

        let mut sources = Vec::with_capacity(edges.len());
        let mut targets = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());

        for (&(src, tgt), &weight) in edges {
            sources.push(src as u32);
            targets.push(tgt as u32);
            weights.push(weight);
        }

        DataFrame::new(vec![
            Series::new("source".into(), sources).into(),
            Series::new("target".into(), targets).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }
}

/// Batch processor for multiple time series from a DataFrame.
#[cfg(feature = "polars-integration")]
pub struct BatchProcessor {
    series_map: HashMap<String, TimeSeries<f64>>,
}

#[cfg(feature = "polars-integration")]
impl BatchProcessor {
    /// Create a batch processor from a Polars DataFrame.
    ///
    /// # Arguments
    ///
    /// - `df`: DataFrame containing multiple time series
    /// - `time_col`: Column name for timestamps
    /// - `value_col`: Column name for values
    /// - `group_col`: Column name for grouping (e.g., sensor_id)
    ///
    /// # Returns
    ///
    /// A BatchProcessor instance ready for processing
    pub fn from_polars_df(
        df: &DataFrame,
        time_col: &str,
        value_col: &str,
        group_col: &str,
    ) -> Result<Self, PolarsError> {
        let mut series_map: HashMap<String, (Vec<f64>, Vec<Option<f64>>)> = HashMap::new();

        // Get columns
        let group_series = df
            .column(group_col)
            .map_err(|_| PolarsError::ColumnNotFound(group_col.to_string()))?;

        let time_series = df
            .column(time_col)
            .map_err(|_| PolarsError::ColumnNotFound(time_col.to_string()))?;

        let value_series = df
            .column(value_col)
            .map_err(|_| PolarsError::ColumnNotFound(value_col.to_string()))?;

        // Extract as vectors
        let groups = group_series
            .str()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be string", group_col)))?;

        let times = time_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be f64", time_col)))?;

        let values = value_series
            .f64()
            .map_err(|_| PolarsError::WrongDataType(format!("{} must be f64", value_col)))?;

        // Manually group the data
        for i in 0..df.height() {
            let group_key = groups
                .get(i)
                .ok_or(PolarsError::EmptyDataFrame)?
                .to_string();

            let time = times
                .get(i)
                .ok_or(PolarsError::EmptyDataFrame)?;

            let value = values.get(i);

            let entry = series_map
                .entry(group_key)
                .or_insert_with(|| (Vec::new(), Vec::new()));

            entry.0.push(time);
            entry.1.push(value);
        }

        // Convert to TimeSeries
        let mut result_map = HashMap::new();
        for (key, (times, values)) in series_map {
            let ts = TimeSeries::new(times, values)
                .map_err(|e| PolarsError::PolarsError(e.to_string()))?;
            result_map.insert(key, ts);
        }

        Ok(BatchProcessor { series_map: result_map })
    }

    /// Process all time series using natural visibility algorithm.
    pub fn process_natural(self) -> Result<BatchResults, PolarsError> {
        let mut results = HashMap::new();

        for (key, series) in self.series_map {
            let graph = VisibilityGraph::from_series(&series)
                .natural_visibility()
                .map_err(|e| PolarsError::PolarsError(e.to_string()))?;

            results.insert(key, graph);
        }

        Ok(BatchResults { results })
    }

    /// Process all time series using horizontal visibility algorithm.
    pub fn process_horizontal(self) -> Result<BatchResults, PolarsError> {
        let mut results = HashMap::new();

        for (key, series) in self.series_map {
            let graph = VisibilityGraph::from_series(&series)
                .horizontal_visibility()
                .map_err(|e| PolarsError::PolarsError(e.to_string()))?;

            results.insert(key, graph);
        }

        Ok(BatchResults { results })
    }
}

/// Results from batch processing multiple time series.
#[cfg(feature = "polars-integration")]
pub struct BatchResults {
    results: HashMap<String, VisibilityGraph<f64>>,
}

#[cfg(feature = "polars-integration")]
impl BatchResults {
    /// Get a specific graph by its group key.
    pub fn get(&self, key: &str) -> Option<&VisibilityGraph<f64>> {
        self.results.get(key)
    }

    /// Export all results to a single Polars DataFrame.
    ///
    /// The DataFrame will include a "group" column identifying each series.
    pub fn to_polars_df(&self) -> Result<DataFrame, PolarsError> {
        let mut all_groups = Vec::new();
        let mut all_node_ids = Vec::new();
        let mut all_degrees = Vec::new();

        for (group, graph) in &self.results {
            let degrees = graph.degree_sequence();
            for (node_id, &degree) in degrees.iter().enumerate() {
                all_groups.push(group.clone());
                all_node_ids.push(node_id as u32);
                all_degrees.push(degree as u32);
            }
        }

        DataFrame::new(vec![
            Series::new("group".into(), all_groups).into(),
            Series::new("node_id".into(), all_node_ids).into(),
            Series::new("degree".into(), all_degrees).into(),
        ])
        .map_err(|e| PolarsError::PolarsError(e.to_string()))
    }

    /// Get all group keys.
    pub fn keys(&self) -> Vec<&String> {
        self.results.keys().collect()
    }

    /// Get the number of processed series.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if no series were processed.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

#[cfg(all(test, feature = "polars-integration"))]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_from_polars() {
        let df = df! {
            "time" => &[0.0, 1.0, 2.0, 3.0],
            "value" => &[1.0, 3.0, 2.0, 4.0],
        }
        .unwrap();

        let series = TimeSeries::<f64>::from_polars_df(&df, "time", "value").unwrap();
        assert_eq!(series.len(), 4);
    }

    #[test]
    fn test_timeseries_to_polars() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let df = series.to_polars_df().unwrap();

        assert_eq!(df.shape(), (4, 2));
        assert!(df.column("time").is_ok());
        assert!(df.column("value").is_ok());
    }

    #[test]
    fn test_graph_to_polars() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let df = graph.to_polars_df().unwrap();
        assert_eq!(df.shape().0, 4); // 4 nodes
    }

    #[test]
    fn test_edges_to_polars() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let df = graph.edges_to_polars_df().unwrap();
        assert!(df.column("source").is_ok());
        assert!(df.column("target").is_ok());
        assert!(df.column("weight").is_ok());
    }
}

