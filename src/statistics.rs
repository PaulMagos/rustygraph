//! Graph statistics and summary information.
//!
//! This module provides convenient methods to compute and display
//! comprehensive statistics about visibility graphs.

use crate::VisibilityGraph;
use std::fmt;

/// Comprehensive statistics about a visibility graph.
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Whether graph is directed
    pub is_directed: bool,
    /// Average degree
    pub average_degree: f64,
    /// Minimum degree
    pub min_degree: usize,
    /// Maximum degree
    pub max_degree: usize,
    /// Average clustering coefficient
    pub average_clustering: f64,
    /// Average shortest path length
    pub average_path_length: f64,
    /// Graph diameter
    pub diameter: usize,
    /// Graph density
    pub density: f64,
    /// Whether graph is connected
    pub is_connected: bool,
    /// Number of features per node
    pub feature_count: usize,
}

impl fmt::Display for GraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Graph Statistics ===")?;
        writeln!(f, "Structure:")?;
        writeln!(f, "  Nodes: {}", self.node_count)?;
        writeln!(f, "  Edges: {}", self.edge_count)?;
        writeln!(f, "  Directed: {}", self.is_directed)?;
        writeln!(f)?;
        writeln!(f, "Degree:")?;
        writeln!(f, "  Average: {:.2}", self.average_degree)?;
        writeln!(f, "  Min: {}", self.min_degree)?;
        writeln!(f, "  Max: {}", self.max_degree)?;
        writeln!(f)?;
        writeln!(f, "Topology:")?;
        writeln!(f, "  Average Clustering: {:.4}", self.average_clustering)?;
        writeln!(f, "  Average Path Length: {:.2}", self.average_path_length)?;
        writeln!(f, "  Diameter: {}", self.diameter)?;
        writeln!(f, "  Density: {:.4}", self.density)?;
        writeln!(f, "  Connected: {}", self.is_connected)?;
        writeln!(f)?;
        writeln!(f, "Features:")?;
        writeln!(f, "  Features per node: {}", self.feature_count)?;
        Ok(())
    }
}

impl<T> VisibilityGraph<T> {
    /// Computes comprehensive statistics about the graph.
    ///
    /// This is a convenience method that computes all available metrics
    /// in one call for easy analysis and reporting.
    ///
    /// # Returns
    ///
    /// A `GraphStatistics` struct with all computed metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let stats = graph.compute_statistics();
    /// println!("{}", stats);
    /// ```
    pub fn compute_statistics(&self) -> GraphStatistics {
        let degrees = self.degree_sequence();
        let avg_degree = if !degrees.is_empty() {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        } else {
            0.0
        };

        let min_degree = degrees.iter().min().copied().unwrap_or(0);
        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        let feature_count = if !self.node_features.is_empty() {
            self.node_features[0].len()
        } else {
            0
        };

        GraphStatistics {
            node_count: self.node_count,
            edge_count: self.edges.len(),
            is_directed: self.directed,
            average_degree: avg_degree,
            min_degree,
            max_degree,
            average_clustering: self.average_clustering_coefficient(),
            average_path_length: self.average_path_length(),
            diameter: self.diameter(),
            density: self.density(),
            is_connected: self.is_connected(),
            feature_count,
        }
    }

    /// Prints a summary of the graph to stdout.
    ///
    /// This is a convenience method for quick inspection during development.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// graph.print_summary();
    /// ```
    pub fn print_summary(&self) {
        println!("{}", self.compute_statistics());
    }
}

