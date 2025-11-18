//! Advanced export formats for large-scale data and scientific computing.
//!
//! This module provides export capabilities for specialized formats:
//! - NPY: NumPy array format for Python integration (requires `npy-export` feature)
//! - Parquet: Columnar format for big data analytics (requires `parquet-export` feature)
//! - HDF5: Hierarchical data format for scientific computing (requires `hdf5-export` feature and system HDF5 library)
//!
//! # Installation Requirements
//!
//! ## HDF5 Export
//!
//! On macOS:
//! ```bash
//! brew install hdf5
//! ```
//!
//! On Ubuntu/Debian:
//! ```bash
//! sudo apt-get install libhdf5-dev
//! ```
//!
//! On Windows:
//! Download and install from https://www.hdfgroup.org/downloads/hdf5/

use crate::VisibilityGraph;
use std::path::Path;

impl<T> VisibilityGraph<T>
where
    T: Copy,
{
    /// Exports graph to NPY format (NumPy arrays).
    ///
    /// Creates multiple .npy files:
    /// - `{basename}_edges.npy` - Edge list as (N, 3) array
    /// - `{basename}_adjacency.npy` - Adjacency matrix as (N, N) array
    /// - `{basename}_degrees.npy` - Degree sequence as (N,) array
    ///
    /// Requires `npy-export` feature.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path for output files (without extension)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "npy-export")]
    /// # {
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// // Creates: graph_edges.npy, graph_adjacency.npy, graph_degrees.npy
    /// graph.to_npy("graph").unwrap();
    /// # }
    /// ```
    #[cfg(feature = "npy-export")]
    pub fn to_npy(&self, base_path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        T: Into<f64>,
    {
        use ndarray::Array2;

        let base = base_path.as_ref();

        // Export edges as (N, 3) array: [source, target, weight]
        let edges: Vec<(usize, usize, f64)> = self.edges
            .iter()
            .map(|(&(src, dst), &weight)| (src, dst, weight))
            .collect();

        if !edges.is_empty() {
            let mut edge_array = Array2::<f64>::zeros((edges.len(), 3));
            for (i, &(src, dst, weight)) in edges.iter().enumerate() {
                edge_array[[i, 0]] = src as f64;
                edge_array[[i, 1]] = dst as f64;
                edge_array[[i, 2]] = weight;
            }

            let edge_path = format!("{}_edges.npy", base.display());
            ndarray_npy::write_npy(&edge_path, &edge_array)?;
        }

        // Export adjacency matrix as (N, N) array
        let adj_matrix = self.to_adjacency_matrix();
        let n = self.node_count;
        let mut adj_array = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                adj_array[[i, j]] = adj_matrix[i][j];
            }
        }

        let adj_path = format!("{}_adjacency.npy", base.display());
        ndarray_npy::write_npy(&adj_path, &adj_array)?;

        // Export degree sequence as (N,) array
        let degrees = self.degree_sequence();
        let degree_array = ndarray::Array1::from_vec(
            degrees.iter().map(|&d| d as f64).collect()
        );

        let deg_path = format!("{}_degrees.npy", base.display());
        ndarray_npy::write_npy(&deg_path, &degree_array)?;

        Ok(())
    }

    /// Exports graph to Parquet format for big data analytics.
    ///
    /// Creates a single .parquet file with columns:
    /// - source: Source node index
    /// - target: Target node index
    /// - weight: Edge weight
    ///
    /// Requires `parquet-export` feature.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "parquet-export")]
    /// # {
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// graph.to_parquet("graph.parquet").unwrap();
    /// # }
    /// ```
    #[cfg(feature = "parquet-export")]
    pub fn to_parquet(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        use arrow::array::{Float64Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        // Create schema
        let schema = Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
            Field::new("weight", DataType::Float64, false),
        ]);

        // Collect edges
        let edges: Vec<_> = self.edges.iter()
            .map(|(&(src, dst), &weight)| (src, dst, weight))
            .collect();

        if edges.is_empty() {
            return Ok(());
        }

        // Create arrays
        let sources: Vec<u64> = edges.iter().map(|&(src, _, _)| src as u64).collect();
        let targets: Vec<u64> = edges.iter().map(|&(_, dst, _)| dst as u64).collect();
        let weights: Vec<f64> = edges.iter().map(|&(_, _, w)| w).collect();

        let source_array = Arc::new(UInt64Array::from(sources));
        let target_array = Arc::new(UInt64Array::from(targets));
        let weight_array = Arc::new(Float64Array::from(weights));

        // Create record batch
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![source_array, target_array, weight_array],
        )?;

        // Write to parquet
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Exports graph to HDF5 format for scientific computing.
    ///
    /// Creates an HDF5 file with groups and datasets:
    /// - `/edges/sources` - Source node indices
    /// - `/edges/targets` - Target node indices
    /// - `/edges/weights` - Edge weights
    /// - `/graph/adjacency` - Adjacency matrix
    /// - `/graph/degrees` - Degree sequence
    /// - `/metadata/node_count` - Number of nodes
    /// - `/metadata/edge_count` - Number of edges
    ///
    /// Requires `hdf5-export` feature.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "hdf5-export")]
    /// # {
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// graph.to_hdf5("graph.h5").unwrap();
    /// # }
    /// ```
    #[cfg(feature = "hdf5-export")]
    pub fn to_hdf5(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        use hdf5::File;

        let file = File::create(path)?;

        // Create groups
        let edges_group = file.create_group("edges")?;
        let graph_group = file.create_group("graph")?;
        let metadata_group = file.create_group("metadata")?;

        // Export edges
        let edges: Vec<_> = self.edges.iter()
            .map(|(&(src, dst), &weight)| (src, dst, weight))
            .collect();

        if !edges.is_empty() {
            let sources: Vec<usize> = edges.iter().map(|&(src, _, _)| src).collect();
            let targets: Vec<usize> = edges.iter().map(|&(_, dst, _)| dst).collect();
            let weights: Vec<f64> = edges.iter().map(|&(_, _, w)| w).collect();

            edges_group.new_dataset::<usize>()
                .shape(sources.len())
                .create("sources")?
                .write(&sources)?;

            edges_group.new_dataset::<usize>()
                .shape(targets.len())
                .create("targets")?
                .write(&targets)?;

            edges_group.new_dataset::<f64>()
                .shape(weights.len())
                .create("weights")?
                .write(&weights)?;
        }

        // Export adjacency matrix
        let adj_matrix = self.to_adjacency_matrix();
        let n = self.node_count;
        let flat_matrix: Vec<f64> = adj_matrix.into_iter().flatten().collect();

        graph_group.new_dataset::<f64>()
            .shape([n, n])
            .create("adjacency")?
            .write(&flat_matrix)?;

        // Export degree sequence
        let degrees = self.degree_sequence();
        graph_group.new_dataset::<usize>()
            .shape(degrees.len())
            .create("degrees")?
            .write(&degrees)?;

        // Export metadata
        metadata_group.new_dataset::<usize>()
            .shape(1)
            .create("node_count")?
            .write(&[self.node_count])?;

        metadata_group.new_dataset::<usize>()
            .shape(1)
            .create("edge_count")?
            .write(&[self.edges.len()])?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;

    #[test]
    #[cfg(feature = "npy-export")]
    fn test_npy_export() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let base_path = temp_dir.join("test_graph");

        // Should not panic
        let _ = graph.to_npy(base_path);
    }

    #[test]
    #[cfg(feature = "parquet-export")]
    fn test_parquet_export() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_graph.parquet");

        // Should not panic
        let _ = graph.to_parquet(&path);
    }

    #[test]
    #[cfg(feature = "hdf5-export")]
    fn test_hdf5_export() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_graph.h5");

        // Should not panic
        let _ = graph.to_hdf5(&path);
    }
}

