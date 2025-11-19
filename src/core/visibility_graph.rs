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
//! let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
//! let graph = VisibilityGraph::from_series(&series)
//!     .natural_visibility()
//!     .unwrap();
//!
//! println!("Number of edges: {}", graph.edges().len());
//! ```

use crate::core::time_series::TimeSeries;
use crate::core::features::FeatureSet;
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
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
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
/// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
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
    pub(crate) edges: HashMap<(usize, usize), f64>,
    /// Adjacency list representation
    adjacency: Vec<Vec<f64>>,
    /// Computed features for each node
    pub node_features: Vec<HashMap<String, T>>,
    /// Whether the graph is directed
    pub directed: bool,
}

/// Direction mode for visibility graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphDirection {
    /// Undirected graph (default) - edges go both ways
    Undirected,
    /// Directed graph - edges only from earlier to later time points
    Directed,
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
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn from_series(series: &TimeSeries<T>) -> VisibilityGraphBuilder<'_, T> {
        VisibilityGraphBuilder {
            series,
            feature_set: None,
            direction: GraphDirection::Undirected,
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// for (&(src, dst), &weight) in graph.edges() {
    ///     println!("{} -> {}: {}", src, dst, weight);
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
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

    /// Checks if an edge exists between two nodes.
    ///
    /// # Arguments
    ///
    /// - `from`: Source node index
    /// - `to`: Target node index
    ///
    /// # Returns
    ///
    /// `true` if an edge exists, `false` otherwise
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.edges.contains_key(&(from, to)) ||
        (!self.directed && self.edges.contains_key(&(to, from)))
    }

    /// Returns the indices of neighboring nodes.
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// Vector of neighbor node indices
    pub fn neighbor_indices(&self, node: usize) -> Vec<usize> {
        self.edges
            .keys()
            .filter_map(|(i, j)| {
                if *i == node {
                    Some(*j)
                } else if !self.directed && *j == node {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect()
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
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
/// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
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
    direction: GraphDirection,
}

impl<'a, T> VisibilityGraphBuilder<'a, T>
where
    T: Copy + PartialOrd + Into<f64>,
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
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
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

    /// Sets the graph direction mode.
    ///
    /// # Arguments
    ///
    /// - `direction`: Whether the graph should be directed or undirected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph, GraphDirection};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .with_direction(GraphDirection::Directed)
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn with_direction(mut self, direction: GraphDirection) -> Self {
        self.direction = direction;
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    /// ```
    pub fn natural_visibility(self) -> Result<VisibilityGraph<T>, GraphError>
    where
        T: Copy + PartialOrd + Into<f64> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
           + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Send + Sync,
    {
        use crate::core::algorithms::edges::{VisibilityEdges as create_visibility_edges, VisibilityType};

        // Check for empty series
        if self.series.is_empty() {
            return Err(GraphError::EmptyTimeSeries);
        }

        // Check for all missing values
        if self.series.values.iter().all(|v| v.is_none()) {
            return Err(GraphError::AllValuesMissing);
        }

        // Compute edges using natural visibility algorithm
        // Use parallel computation when available (significant speedup for large graphs)
        let edges: HashMap<(usize, usize), f64> = {
            let edge_computer = create_visibility_edges::new(
                self.series,
                VisibilityType::Natural,
                |_, _, _, _| 1.0
            );

            #[cfg(feature = "parallel")]
            {
                edge_computer.compute_edges_parallel()
            }

            #[cfg(not(feature = "parallel"))]
            {
                edge_computer.compute_edges()
            }
        };

        // Compute adjacency list
        let directed = matches!(self.direction, GraphDirection::Directed);
        let adj: Vec<Vec<f64>> = build_adjacency_list(self.series.len(), &edges, directed);

        // Compute node features if feature set is provided
        let node_features = if let Some(feature_set) = self.feature_set {
            compute_node_features(&self.series.values, &feature_set)
        } else {
            vec![]
        };

        Ok(VisibilityGraph {
            node_count: self.series.len(),
            edges,
            adjacency: adj,
            node_features,
            directed,
        })
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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .horizontal_visibility()
    ///     .unwrap();
    /// ```
    pub fn horizontal_visibility(self) -> Result<VisibilityGraph<T>, GraphError>
    where
        T: Copy + PartialOrd + Into<f64> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
           + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Send + Sync,
    {
        use crate::core::algorithms::edges::{VisibilityEdges as create_visibility_edges, VisibilityType};

        // Check for empty series
        if self.series.is_empty() {
            return Err(GraphError::EmptyTimeSeries);
        }

        // Check for all missing values
        if self.series.values.iter().all(|v| v.is_none()) {
            return Err(GraphError::AllValuesMissing);
        }

        // Compute edges using horizontal visibility algorithm
        // Use parallel computation when available (significant speedup for large graphs)
        let edges: HashMap<(usize, usize), f64> = {
            let edge_computer = create_visibility_edges::new(
                self.series,
                VisibilityType::Horizontal,
                |_, _, _, _| 1.0
            );

            #[cfg(feature = "parallel")]
            {
                edge_computer.compute_edges_parallel()
            }

            #[cfg(not(feature = "parallel"))]
            {
                edge_computer.compute_edges()
            }
        };

        // Compute adjacency list
        let directed = matches!(self.direction, GraphDirection::Directed);
        let adj: Vec<Vec<f64>> = build_adjacency_list(self.series.len(), &edges, directed);

        // Compute node features if feature set is provided
        let node_features = if let Some(feature_set) = self.feature_set {
            compute_node_features(&self.series.values, &feature_set)
        } else {
            vec![]
        };

        Ok(VisibilityGraph {
            node_count: self.series.len(),
            edges,
            adjacency: adj,
            node_features,
            directed,
        })
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


fn build_adjacency_list(node_count: usize, edges: &HashMap<(usize, usize), f64>, directed: bool) -> Vec<Vec<f64>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(src, dst) in edges.keys() {
        let weight = edges.get(&(src, dst)).copied().unwrap_or(0.0);
        adjacency[src].push(weight);
        if !directed {
            adjacency[dst].push(weight); // Add reverse edge for undirected
        }
    }
    adjacency
}

/// Computes all features for all nodes in the series.
///
/// This function automatically uses parallel computation when the `parallel` feature is enabled,
/// otherwise falls back to sequential computation.
pub(crate) fn compute_node_features<T>(
    series: &[Option<T>],
    feature_set: &FeatureSet<T>,
) -> Vec<HashMap<String, T>>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
       + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64>
       + Send + Sync,
{
    // Use parallel implementation when parallel feature is enabled
    #[cfg(feature = "parallel")]
    {
        crate::performance::parallel::compute_node_features_parallel(series, feature_set)
    }

    // Fall back to sequential implementation
    #[cfg(not(feature = "parallel"))]
    {
        let mut node_features = Vec::with_capacity(series.len());

        for i in 0..series.len() {
            let mut features = HashMap::new();

            // Compute each feature for this node
            for feature in &feature_set.features {
                if let Some(value) = feature.compute(series, i, &feature_set.missing_strategy) {
                    features.insert(feature.name().to_string(), value);
                }
            }

            node_features.push(features);
        }

        node_features
    }
}

impl std::error::Error for GraphError {}

