//! Visibility graph construction and representation.
//!
//! This module provides types for building and working with visibility graphs
//! from time series data. It supports both natural and horizontal visibility algorithms.
//!
//! # Examples
//!
//! ```rust
//! use rustygraph::{TimeSeries, VisibilityGraph};
//!
//! let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
//! let graph = VisibilityGraph::from_series(&series)
//!     .natural_visibility()
//!     .unwrap();
//!
//! println!("Number of edges: {}", graph.edges().len());
//! ```

use crate::time_series::TimeSeries;
use crate::features::FeatureSet;
use std::collections::HashMap;
use std::fmt;

/// Visibility graph representation with node features.
///
/// A visibility graph represents temporal connectivity patterns in time series data.
/// Each data point becomes a node, and edges connect points that have "visibility"
/// to each other according to the chosen algorithm.
///
/// # Examples
///
/// ## Basic usage
///
/// ```rust
/// use rustygraph::{TimeSeries, VisibilityGraph};
///
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
/// let graph = VisibilityGraph::from_series(&series)
///     .natural_visibility()
///     .unwrap();
///
/// // Access graph properties
/// println!("Nodes: {}", graph.node_count);
/// println!("Edges: {}", graph.edges().len());
/// println!("Degree sequence: {:?}", graph.degree_sequence());
/// ```
///
/// ## With features
///
/// ```rust
/// use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};
///
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
/// let graph = VisibilityGraph::from_series(&series)
///     .with_features(
///         FeatureSet::new()
///             .add_builtin(BuiltinFeature::DeltaForward)
///             .add_builtin(BuiltinFeature::LocalSlope)
///     )
///     .natural_visibility()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct VisibilityGraph<T> {
    /// Number of nodes (data points)
    pub node_count: usize,
    /// Graph edges as (source, target) pairs
    edges: HashMap<(usize, usize), f64>,
    /// Adjacency list representation
    adjacency: Vec<Vec<f64>>,
    /// Computed features for each node
    pub node_features: Vec<HashMap<String, T>>,
}

impl<T> VisibilityGraph<T> {
    /// Creates a visibility graph builder from a time series.
    ///
    /// This is the main entry point for constructing visibility graphs.
    /// Use the returned builder to configure options and select an algorithm.
    ///
    /// # Arguments
    ///
    /// - `series`: Reference to time series data
    ///
    /// # Returns
    ///
    /// A [`VisibilityGraphBuilder`] for configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn from_series(series: &TimeSeries<T>) -> VisibilityGraphBuilder<'_, T> {
        VisibilityGraphBuilder {
            series,
            feature_set: None,
        }
    }

    /// Returns all edges in the graph.
    ///
    /// Each edge is represented as a tuple `(source, target)` where both
    /// values are node indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// for (src, dst) in graph.edges() {
    ///     println!("{} -> {}", src, dst);
    /// }
    /// ```
    pub fn edges(&self) -> &HashMap<(usize, usize), f64> {
        &self.edges
    }

    /// Returns the adjacency list for a specific node.
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// Slice of adjacent node indices, or `None` if node doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// if let Some(neighbors) = graph.neighbors(0) {
    ///     println!("Node 0 is connected to: {:?}", neighbors);
    /// }
    /// ```
    pub fn neighbors(&self, node: usize) -> Option<&[f64]> {
        self.adjacency.get(node).map(|v| v.as_slice())
    }

    /// Returns the degree of a node (number of connections).
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// The number of edges connected to this node, or `None` if node doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// if let Some(degree) = graph.degree(0) {
    ///     println!("Node 0 has degree {}", degree);
    /// }
    /// ```
    pub fn degree(&self, node: usize) -> Option<usize> {
        self.adjacency.get(node).map(|v| v.len())
    }

    /// Returns the degree sequence of all nodes.
    ///
    /// The degree sequence is a vector where each element is the degree
    /// of the corresponding node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let degrees = graph.degree_sequence();
    /// println!("Degree sequence: {:?}", degrees);
    /// ```
    pub fn degree_sequence(&self) -> Vec<usize> {
        self.adjacency.iter().map(|v| v.len()).collect()
    }

    /// Returns computed features for a specific node.
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// A map of feature names to values, or `None` if node doesn't exist
    /// or no features were computed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .with_features(
    ///         FeatureSet::new().add_builtin(BuiltinFeature::DeltaForward)
    ///     )
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// if let Some(features) = graph.node_features(0) {
    ///     for (name, value) in features {
    ///         println!("{}: {}", name, value);
    ///     }
    /// }
    /// ```
    pub fn node_features(&self, node: usize) -> Option<&HashMap<String, T>> {
        self.node_features.get(node)
    }

    /// Exports the graph to an adjacency matrix.
    ///
    /// The adjacency matrix is a square boolean matrix where `matrix[i][j]`
    /// is `true` if there is an edge from node `i` to node `j`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let matrix = graph.to_adjacency_matrix();
    /// for row in &matrix {
    ///     println!("{:?}", row);
    /// }
    /// ```
    pub fn to_adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.node_count]; self.node_count];
        for &(src, dst) in self.edges.keys() {
            matrix[src][dst] = self.edges[&(src, dst)];
        }
        matrix
    }
}

/// Builder for constructing visibility graphs with options.
///
/// This builder allows you to configure features and select the visibility
/// algorithm before constructing the graph.
///
/// # Examples
///
/// ```rust
/// use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};
///
/// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
/// let graph = VisibilityGraph::from_series(&series)
///     .with_features(
///         FeatureSet::new().add_builtin(BuiltinFeature::DeltaForward)
///     )
///     .natural_visibility()
///     .unwrap();
/// ```
pub struct VisibilityGraphBuilder<'a, T> {
    #[allow(dead_code)] // Will be used when algorithms are implemented
    series: &'a TimeSeries<T>,
    feature_set: Option<FeatureSet<T>>,
}

impl<'a, T> VisibilityGraphBuilder<'a, T>
{
    /// Specifies features to compute for each node.
    ///
    /// Features are computed after the graph structure is built.
    ///
    /// # Arguments
    ///
    /// - `features`: A configured [`FeatureSet`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .with_features(
    ///         FeatureSet::new()
    ///             .add_builtin(BuiltinFeature::DeltaForward)
    ///             .add_builtin(BuiltinFeature::LocalSlope)
    ///     )
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn with_features(mut self, features: FeatureSet<T>) -> Self {
        self.feature_set = Some(features);
        self
    }

    /// Constructs a natural visibility graph.
    ///
    /// In a natural visibility graph, two nodes (i, yi) and (j, yj) are connected
    /// if all intermediate points (k, yk) satisfy the visibility criterion:
    ///
    /// `yk < yi + (yj - yi) * (tk - ti) / (tj - ti)`
    ///
    /// # Algorithm
    ///
    /// Uses a monotonic stack optimization for O(n) complexity per node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] if the time series is empty or feature computation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn natural_visibility(self) -> Result<VisibilityGraph<T>, GraphError> {
        use crate::algorithms::{create_visibility_edges, natural_visibility};
        // Check for empty series
        if self.series.is_empty() {
            Err(GraphError::EmptyTimeSeries).expect("TODO: panic message");
        }

        // Extract values
        let val : Vec<T> = self.series.values.iter().map(|v| {
            match v {
                Some(x) => *x,
                None => f64::NAN, // Placeholder for missing values
            }
        }).collect();

        // Check for all missing values
        if val.is_empty(){
            return Err(GraphError::AllValuesMissing);
        }

        // Compute edges using natural visibility algorithm
        let edges : HashMap<(usize, usize), f64> =
            create_visibility_edges::new(&val,
                                        natural_visibility,
                                        |_, _, _, _| {1.0})
            .compute_edges();

        {

        // Compute adjacency list
        let adj : Vec<Vec<f64>> = build_adjacency_list(val.len(), &edges);


        Ok(VisibilityGraph{
            node_count: val.len(),
            edges: HashMap::new(),
            adjacency: vec![],
            node_features: vec![],
        })}
    }

    /// Constructs a horizontal visibility graph.
    ///
    /// In a horizontal visibility graph, two nodes are connected if all
    /// intermediate values are strictly lower than both endpoints.
    ///
    /// # Algorithm
    ///
    /// Uses a linear scan approach with O(n) average case complexity.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] if the time series is empty or feature computation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .horizontal_visibility()
    ///     .unwrap();
    /// ```
    pub fn horizontal_visibility(self) -> Result<VisibilityGraph<T>, GraphError> {
        // Implementation will be provided
        todo!("Horizontal visibility algorithm implementation")
    }
}

/// Errors during visibility graph construction.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// Time series is empty
    EmptyTimeSeries,
    /// Feature computation failed
    FeatureComputationFailed {
        /// Name of the feature that failed
        feature: String,
        /// Node index where failure occurred
        node: usize,
    },
    /// All values are missing
    AllValuesMissing,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::EmptyTimeSeries => write!(f, "Time series is empty"),
            GraphError::FeatureComputationFailed { feature, node } => {
                write!(f, "Feature '{}' computation failed at node {}", feature, node)
            }
            GraphError::AllValuesMissing => write!(f, "All values are missing"),
        }
    }
}


fn build_adjacency_list(node_count: usize, edges: &HashMap<(usize, usize), f64>) -> Vec<Vec<f64>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(src, dst) in edges.keys() {
        adjacency[src].push(edges.get(&(src, dst)).copied().unwrap_or(0.0));
        adjacency[dst].push(*adjacency[src].last().unwrap()); // Undirected
    }
    adjacency
}


impl std::error::Error for GraphError {}

