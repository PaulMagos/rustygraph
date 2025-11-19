//! Batch processing utilities for multiple time series.
//!
//! This module provides utilities for processing multiple time series
//! and comparing their visibility graphs.

use crate::core::{TimeSeries, VisibilityGraph, GraphError};
use std::collections::HashMap;

/// Results from batch processing multiple time series.
#[derive(Debug, Clone)]
pub struct BatchResults<T> {
    /// Individual graphs for each time series
    pub graphs: Vec<VisibilityGraph<T>>,
    /// Labels for each time series
    pub labels: Vec<String>,
    /// Aggregate statistics
    pub statistics: BatchStatistics,
}

/// Aggregate statistics across multiple graphs.
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    /// Average number of edges across graphs
    pub avg_edges: f64,
    /// Average clustering coefficient
    pub avg_clustering: f64,
    /// Average path length
    pub avg_path_length: f64,
    /// Average density
    pub avg_density: f64,
}

/// Builder for batch processing visibility graphs.
///
/// # Examples
///
/// ```rust
/// use rustygraph::{TimeSeries, BatchProcessor};
///
/// let series1 = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let series2 = TimeSeries::from_raw(vec![2.0, 1.0, 3.0, 2.0]).unwrap();
///
/// let results = BatchProcessor::new()
///     .add_series(&series1, "Series1")
///     .add_series(&series2, "Series2")
///     .process_natural()
///     .unwrap();
///
/// println!("Processed {} time series", results.graphs.len());
/// ```
pub struct BatchProcessor<'a, T> {
    series_list: Vec<(&'a TimeSeries<T>, String)>,
}

impl<'a, T> BatchProcessor<'a, T>
where
    T: Copy + PartialOrd + Into<f64>,
{
    /// Creates a new batch processor.
    pub fn new() -> Self {
        Self {
            series_list: Vec::new(),
        }
    }

    /// Adds a time series to the batch.
    ///
    /// # Arguments
    ///
    /// - `series`: Reference to time series
    /// - `label`: Label for identification
    pub fn add_series(mut self, series: &'a TimeSeries<T>, label: &str) -> Self {
        self.series_list.push((series, label.to_string()));
        self
    }

    /// Processes all time series with natural visibility algorithm.
    pub fn process_natural(self) -> Result<BatchResults<T>, GraphError>
    where
        T: Copy + PartialOrd + Into<f64> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
           + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Send + Sync,
    {
        let mut graphs = Vec::new();
        let mut labels = Vec::new();

        for (series, label) in &self.series_list {
            let builder = VisibilityGraph::from_series(series);
            let graph = builder.natural_visibility()?;
            graphs.push(graph);
            labels.push(label.clone());
        }

        let statistics = Self::compute_statistics(&graphs);

        Ok(BatchResults {
            graphs,
            labels,
            statistics,
        })
    }

    /// Processes all time series with horizontal visibility algorithm.
    pub fn process_horizontal(self) -> Result<BatchResults<T>, GraphError>
    where
        T: Copy + PartialOrd + Into<f64> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
           + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Send + Sync,
    {
        let mut graphs = Vec::new();
        let mut labels = Vec::new();

        for (series, label) in &self.series_list {
            let builder = VisibilityGraph::from_series(series);
            let graph = builder.horizontal_visibility()?;
            graphs.push(graph);
            labels.push(label.clone());
        }

        let statistics = Self::compute_statistics(&graphs);

        Ok(BatchResults {
            graphs,
            labels,
            statistics,
        })
    }

    fn compute_statistics(graphs: &[VisibilityGraph<T>]) -> BatchStatistics {
        if graphs.is_empty() {
            return BatchStatistics {
                avg_edges: 0.0,
                avg_clustering: 0.0,
                avg_path_length: 0.0,
                avg_density: 0.0,
            };
        }

        let n = graphs.len() as f64;
        let avg_edges = graphs.iter().map(|g| g.edges.len()).sum::<usize>() as f64 / n;
        let avg_clustering = graphs.iter().map(|g| g.average_clustering_coefficient()).sum::<f64>() / n;
        let avg_path_length = graphs.iter().map(|g| g.average_path_length()).sum::<f64>() / n;
        let avg_density = graphs.iter().map(|g| g.density()).sum::<f64>() / n;

        BatchStatistics {
            avg_edges,
            avg_clustering,
            avg_path_length,
            avg_density,
        }
    }
}

impl<'a, T> Default for BatchProcessor<'a, T>
where
    T: Copy + PartialOrd + Into<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Compares two visibility graphs and returns similarity metrics.
///
/// # Arguments
///
/// - `graph1`: First graph
/// - `graph2`: Second graph
///
/// # Returns
///
/// Hashmap of similarity metrics
///
/// # Examples
///
/// ```rust
/// use rustygraph::{TimeSeries, VisibilityGraph, compare_graphs};
///
/// let series1 = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let series2 = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
///
/// let g1 = VisibilityGraph::from_series(&series1).natural_visibility().unwrap();
/// let g2 = VisibilityGraph::from_series(&series2).natural_visibility().unwrap();
///
/// let similarity = compare_graphs(&g1, &g2);
/// println!("Edge overlap: {:.2}%", similarity["edge_overlap"] * 100.0);
/// ```
pub fn compare_graphs<T>(graph1: &VisibilityGraph<T>, graph2: &VisibilityGraph<T>) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();

    // Edge overlap (Jaccard similarity)
    let edges1: std::collections::HashSet<_> = graph1.edges.keys().collect();
    let edges2: std::collections::HashSet<_> = graph2.edges.keys().collect();

    let intersection = edges1.intersection(&edges2).count();
    let union = edges1.union(&edges2).count();

    let edge_overlap = if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    };
    metrics.insert("edge_overlap".to_string(), edge_overlap);

    // Degree sequence correlation
    let degrees1 = graph1.degree_sequence();
    let degrees2 = graph2.degree_sequence();

    if degrees1.len() == degrees2.len() && !degrees1.is_empty() {
        let mean1 = degrees1.iter().sum::<usize>() as f64 / degrees1.len() as f64;
        let mean2 = degrees2.iter().sum::<usize>() as f64 / degrees2.len() as f64;

        let mut covariance = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..degrees1.len() {
            let d1 = degrees1[i] as f64 - mean1;
            let d2 = degrees2[i] as f64 - mean2;
            covariance += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let correlation = if var1 > 0.0 && var2 > 0.0 {
            covariance / (var1.sqrt() * var2.sqrt())
        } else {
            1.0
        };

        metrics.insert("degree_correlation".to_string(), correlation);
    }

    // Metric differences
    let clustering_diff = (graph1.average_clustering_coefficient() -
                          graph2.average_clustering_coefficient()).abs();
    metrics.insert("clustering_diff".to_string(), clustering_diff);

    let density_diff = (graph1.density() - graph2.density()).abs();
    metrics.insert("density_diff".to_string(), density_diff);

    metrics
}

impl<T> BatchResults<T> {
    /// Prints a summary of batch processing results.
    pub fn print_summary(&self) {
        println!("=== Batch Processing Results ===");
        println!("Total graphs: {}", self.graphs.len());
        println!("\nAggregate Statistics:");
        println!("  Average edges: {:.2}", self.statistics.avg_edges);
        println!("  Average clustering: {:.4}", self.statistics.avg_clustering);
        println!("  Average path length: {:.2}", self.statistics.avg_path_length);
        println!("  Average density: {:.4}", self.statistics.avg_density);

        println!("\nIndividual Graphs:");
        for (i, label) in self.labels.iter().enumerate() {
            let graph = &self.graphs[i];
            println!("  {} - {} nodes, {} edges", label, graph.node_count, graph.edges.len());
        }
    }
}

