//! Python bindings for RustyGraph using PyO3.
//!
//! This module provides Python access to the core RustyGraph functionality.
//!
//! Requires `python-bindings` feature.
//!
//! # Installation
//!
//! ```bash
//! pip install maturin
//! maturin develop --features python-bindings
//! ```
//!
//! # Usage in Python
//!
//! ```python
//! import rustygraph
//!
//! # Create a time series
//! series = rustygraph.TimeSeries([1.0, 3.0, 2.0, 4.0, 3.0])
//!
//! # Build visibility graph
//! graph = series.natural_visibility()
//!
//! # Get graph properties
//! print(f"Nodes: {graph.node_count()}")
//! print(f"Edges: {graph.edge_count()}")
//! print(f"Density: {graph.density()}")
//! ```

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyModule;
#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyArrayMethods};

#[cfg(feature = "python-bindings")]
use crate::{
    TimeSeries as RustTimeSeries,
    VisibilityGraph as RustVisibilityGraph,
    FeatureSet as RustFeatureSet,
    BuiltinFeature as RustBuiltinFeature,
    MissingDataStrategy as RustMissingDataStrategy,
};

/// Python wrapper for TimeSeries.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "TimeSeries")]
pub struct PyTimeSeries {
    inner: RustTimeSeries<f64>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyTimeSeries {
    /// Creates a new time series from a list of values.
    #[new]
    fn new(values: Vec<f64>) -> PyResult<Self> {
        let inner = RustTimeSeries::from_raw(values)
            .map_err(py_error_helpers::to_value_error)?;
        Ok(Self { inner })
    }
    
    /// Returns the length of the time series.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Creates a natural visibility graph.
    fn natural_visibility(&self) -> PyResult<PyVisibilityGraph> {
        let graph = RustVisibilityGraph::from_series(&self.inner)
            .natural_visibility()
            .map_err(py_error_helpers::to_runtime_error)?;
        Ok(PyVisibilityGraph { inner: graph })
    }
    
    /// Creates a horizontal visibility graph.
    fn horizontal_visibility(&self) -> PyResult<PyVisibilityGraph> {
        let graph = RustVisibilityGraph::from_series(&self.inner)
            .horizontal_visibility()
            .map_err(py_error_helpers::to_runtime_error)?;
        Ok(PyVisibilityGraph { inner: graph })
    }
    
    /// Creates a natural visibility graph with features.
    /// Note: This takes ownership of the FeatureSet
    fn natural_visibility_with_features(&self, mut features: PyRefMut<'_, PyFeatureSet>) -> PyResult<PyVisibilityGraph> {
        // Take ownership of the inner FeatureSet
        let feature_set = std::mem::replace(&mut features.inner, RustFeatureSet::new());

        let graph = RustVisibilityGraph::from_series(&self.inner)
            .with_features(feature_set)
            .natural_visibility()
            .map_err(py_error_helpers::to_runtime_error)?;
        Ok(PyVisibilityGraph { inner: graph })
    }

    /// Creates a horizontal visibility graph with features.
    /// Note: This takes ownership of the FeatureSet
    fn horizontal_visibility_with_features(&self, mut features: PyRefMut<'_, PyFeatureSet>) -> PyResult<PyVisibilityGraph> {
        // Take ownership of the inner FeatureSet
        let feature_set = std::mem::replace(&mut features.inner, RustFeatureSet::new());

        let graph = RustVisibilityGraph::from_series(&self.inner)
            .with_features(feature_set)
            .horizontal_visibility()
            .map_err(py_error_helpers::to_runtime_error)?;
        Ok(PyVisibilityGraph { inner: graph })
    }

    /// Returns the values as a NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self.inner.values.iter().filter_map(|&v| v).collect();
        PyArray1::from_vec(py, values)
    }

    /// Handle missing data using the specified strategy.
    fn handle_missing(&self, strategy: &PyMissingDataStrategy) -> PyResult<Self> {
        let cleaned = self.inner.handle_missing(strategy.inner.clone())
            .map_err(py_error_helpers::to_runtime_error)?;
        Ok(Self { inner: cleaned })
    }

    /// Create a time series from values with explicit missing data (None values).
    #[staticmethod]
    fn with_missing(timestamps: Vec<f64>, values: Vec<Option<f64>>) -> PyResult<Self> {
        let inner = RustTimeSeries::new(timestamps, values)
            .map_err(py_error_helpers::to_value_error)?;
        Ok(Self { inner })
    }
}

/// Python wrapper for MissingDataStrategy.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "MissingDataStrategy")]
#[derive(Clone)]
pub struct PyMissingDataStrategy {
    inner: RustMissingDataStrategy,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyMissingDataStrategy {
    /// Linear interpolation: average of neighbors
    #[staticmethod]
    fn linear_interpolation() -> Self {
        Self {
            inner: RustMissingDataStrategy::LinearInterpolation,
        }
    }

    /// Forward fill: use last valid value
    #[staticmethod]
    fn forward_fill() -> Self {
        Self {
            inner: RustMissingDataStrategy::ForwardFill,
        }
    }

    /// Backward fill: use next valid value
    #[staticmethod]
    fn backward_fill() -> Self {
        Self {
            inner: RustMissingDataStrategy::BackwardFill,
        }
    }

    /// Nearest neighbor: use closest valid value
    #[staticmethod]
    fn nearest_neighbor() -> Self {
        Self {
            inner: RustMissingDataStrategy::NearestNeighbor,
        }
    }

    /// Mean imputation with window
    #[staticmethod]
    fn mean_imputation(window_size: usize) -> Self {
        Self {
            inner: RustMissingDataStrategy::MeanImputation { window_size },
        }
    }

    /// Median imputation with window
    #[staticmethod]
    fn median_imputation(window_size: usize) -> Self {
        Self {
            inner: RustMissingDataStrategy::MedianImputation { window_size },
        }
    }

    /// Fill with zeros
    #[staticmethod]
    fn zero_fill() -> Self {
        Self {
            inner: RustMissingDataStrategy::ZeroFill,
        }
    }

    /// Drop missing values (return None)
    #[staticmethod]
    fn drop() -> Self {
        Self {
            inner: RustMissingDataStrategy::Drop,
        }
    }

    /// Chain strategies: try primary, fallback to secondary if it fails
    fn with_fallback(&self, fallback: &PyMissingDataStrategy) -> Self {
        Self {
            inner: self.inner.clone().with_fallback(fallback.inner.clone()),
        }
    }

    fn __repr__(&self) -> String {
        format!("MissingDataStrategy({:?})", self.inner)
    }
}

/// Python wrapper for VisibilityGraph.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VisibilityGraph")]
pub struct PyVisibilityGraph {
    inner: RustVisibilityGraph<f64>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVisibilityGraph {
    /// Returns the number of nodes.
    fn node_count(&self) -> usize {
        self.inner.node_count
    }
    
    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.inner.edges().len()
    }
    
    /// Returns the graph density.
    fn density(&self) -> f64 {
        self.inner.density()
    }
    
    /// Returns the average clustering coefficient.
    fn clustering_coefficient(&self) -> f64 {
        self.inner.average_clustering_coefficient()
    }
    
    /// Returns the graph diameter.
    fn diameter(&self) -> usize {
        self.inner.diameter()
    }
    
    /// Returns the degree sequence as a list.
    fn degree_sequence(&self) -> Vec<usize> {
        self.inner.degree_sequence()
    }
    
    /// Returns edges as a list of tuples (source, target, weight).
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges().iter()
            .map(|(&(src, dst), &weight)| (src, dst, weight))
            .collect()
    }
    
    /// Returns the adjacency matrix as a NumPy array.
    fn adjacency_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let matrix = self.inner.to_adjacency_matrix();
        let n = self.inner.node_count;
        let flat: Vec<f64> = matrix.into_iter().flatten().collect();
        let array = PyArray1::from_vec(py, flat);
        array.reshape([n, n])
    }
    
    /// Exports the graph to JSON.
    fn to_json(&self) -> String {
        self.inner.to_json(crate::io::ExportOptions::default())
    }

    /// Exports edges to CSV string (with weights).
    fn to_edge_list_csv(&self, include_weights: bool) -> String {
        self.inner.to_edge_list_csv(include_weights)
    }

    /// Exports adjacency matrix to CSV string.
    fn to_adjacency_csv(&self) -> String {
        self.inner.to_adjacency_matrix_csv()
    }

    /// Exports node features to CSV string.
    fn to_features_csv(&self) -> String {
        self.inner.features_to_csv()
    }

    /// Exports graph to DOT format (GraphViz).
    fn to_dot(&self) -> String {
        self.inner.to_dot()
    }

    /// Exports graph to GraphML format.
    fn to_graphml(&self) -> String {
        self.inner.to_graphml()
    }

    /// Saves edges to CSV file.
    fn save_edge_list_csv(&self, path: &str, include_weights: bool) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        let csv = self.inner.to_edge_list_csv(include_weights);
        let mut file = File::create(path)
            .map_err(|e| py_error_helpers::to_runtime_error(e))?;
        file.write_all(csv.as_bytes())
            .map_err(|e| py_error_helpers::to_runtime_error(e))
    }

    /// Saves adjacency matrix to CSV file.
    fn save_adjacency_csv(&self, path: &str) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        let csv = self.inner.to_adjacency_matrix_csv();
        let mut file = File::create(path)
            .map_err(|e| py_error_helpers::to_runtime_error(e))?;
        file.write_all(csv.as_bytes())
            .map_err(|e| py_error_helpers::to_runtime_error(e))
    }

    /// Saves to DOT file.
    fn save_dot(&self, path: &str) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        let dot = self.inner.to_dot();
        let mut file = File::create(path)
            .map_err(|e| py_error_helpers::to_runtime_error(e))?;
        file.write_all(dot.as_bytes())
            .map_err(|e| py_error_helpers::to_runtime_error(e))
    }

    /// Saves to GraphML file.
    fn save_graphml(&self, path: &str) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        let graphml = self.inner.to_graphml();
        let mut file = File::create(path)
            .map_err(|e| py_error_helpers::to_runtime_error(e))?;
        file.write_all(graphml.as_bytes())
            .map_err(|e| py_error_helpers::to_runtime_error(e))
    }

    /// Get feature names for a specific node
    fn get_node_feature_names(&self, node: usize) -> PyResult<Vec<String>> {
        if node >= self.inner.node_count {
            return Err(py_error_helpers::to_value_error(
                format!("Node {} out of range (graph has {} nodes)", node, self.inner.node_count)
            ));
        }

        if node >= self.inner.node_features.len() {
            return Err(py_error_helpers::to_value_error(
                "No features computed. Use natural_visibility_with_features() or horizontal_visibility_with_features()"
            ));
        }

        let features = &self.inner.node_features[node];
        Ok(features.keys().cloned().collect())
    }

    /// Get features for a specific node as a dictionary {feature_name: value}
    fn get_node_features(&self, node: usize) -> PyResult<std::collections::HashMap<String, f64>> {
        if node >= self.inner.node_count {
            return Err(py_error_helpers::to_value_error(
                format!("Node {} out of range (graph has {} nodes)", node, self.inner.node_count)
            ));
        }

        if node >= self.inner.node_features.len() {
            return Err(py_error_helpers::to_value_error(
                "No features computed. Use natural_visibility_with_features() or horizontal_visibility_with_features()"
            ));
        }

        Ok(self.inner.node_features[node].clone())
    }

    /// Get all node features as a NumPy array (nodes x features)
    /// Returns array where row i contains features for node i
    fn get_all_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.inner.node_features.is_empty() {
            return Err(py_error_helpers::to_value_error(
                "No features computed. Use natural_visibility_with_features() or horizontal_visibility_with_features()"
            ));
        }

        let n = self.inner.node_count;

        // Get feature names from first node (all nodes should have same features)
        let feature_names: Vec<String> = self.inner.node_features[0].keys().cloned().collect();
        let feature_count = feature_names.len();

        // Build flat array
        let mut flat = Vec::with_capacity(n * feature_count);
        for i in 0..n {
            for name in &feature_names {
                let value = self.inner.node_features.get(i)
                    .and_then(|f| f.get(name))
                    .copied()
                    .unwrap_or(0.0);
                flat.push(value);
            }
        }

        let array = PyArray1::from_vec(py, flat);
        array.reshape([n, feature_count])
    }

    /// Check if features were computed
    fn has_features(&self) -> bool {
        !self.inner.node_features.is_empty()
    }

    /// Get the number of features per node
    fn feature_count(&self) -> usize {
        self.inner.node_features.first()
            .map(|f| f.len())
            .unwrap_or(0)
    }

    /// Computes betweenness centrality for a node.
    fn betweenness_centrality(&self, node: usize) -> PyResult<f64> {
        self.inner.betweenness_centrality(node)
            .ok_or_else(|| py_error_helpers::to_value_error(
                format!("Node {} not found in graph", node)
            ))
    }
    
    /// Detects communities and returns node assignments.
    fn detect_communities(&self) -> Vec<usize> {
        self.inner.detect_communities().node_communities
    }

    /// Computes shortest path length between two nodes.
    fn shortest_path_length(&self, source: usize, target: usize) -> Option<usize> {
        self.inner.shortest_path_length(source, target)
    }

    /// Computes average path length across all node pairs.
    fn average_path_length(&self) -> f64 {
        self.inner.average_path_length()
    }

    /// Returns the graph radius (minimum eccentricity).
    fn radius(&self) -> usize {
        self.inner.radius()
    }

    /// Checks if the graph is connected.
    fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Counts the number of connected components.
    fn count_components(&self) -> usize {
        self.inner.count_components()
    }

    /// Returns the size of the largest component.
    fn largest_component_size(&self) -> usize {
        self.inner.largest_component_size()
    }

    /// Computes assortativity coefficient (degree correlation).
    fn assortativity(&self) -> f64 {
        self.inner.assortativity()
    }

    /// Computes degree variance.
    fn degree_variance(&self) -> f64 {
        self.inner.degree_variance()
    }

    /// Computes degree standard deviation.
    fn degree_std_dev(&self) -> f64 {
        self.inner.degree_std_dev()
    }

    /// Returns degree distribution as a dictionary {degree: count}.
    fn degree_distribution(&self) -> std::collections::HashMap<usize, usize> {
        self.inner.degree_distribution()
    }

    /// Computes degree entropy.
    fn degree_entropy(&self) -> f64 {
        self.inner.degree_entropy()
    }

    /// Computes clustering coefficient for a specific node.
    fn node_clustering_coefficient(&self, node: usize) -> Option<f64> {
        self.inner.clustering_coefficient(node)
    }

    /// Computes global clustering coefficient (transitivity).
    fn global_clustering_coefficient(&self) -> f64 {
        self.inner.global_clustering_coefficient()
    }

    /// Computes betweenness centrality for all nodes.
    fn betweenness_centrality_all(&self) -> Vec<f64> {
        self.inner.betweenness_centrality_all()
    }

    /// Computes degree centrality for all nodes.
    fn degree_centrality(&self) -> Vec<f64> {
        self.inner.degree_centrality()
    }

    /// Computes comprehensive statistics for the graph.
    fn compute_statistics(&self) -> PyGraphStatistics {
        let stats = self.inner.compute_statistics();
        PyGraphStatistics { inner: stats }
    }

    /// Detects motifs in the graph.
    fn detect_motifs(&self) -> PyMotifCounts {
        let motifs = self.inner.detect_3node_motifs();
        PyMotifCounts { inner: motifs }
    }

    /// Get neighbors of a node with their edge weights.
    fn neighbors(&self, node: usize) -> Option<Vec<f64>> {
        self.inner.neighbors(node).map(|n| n.to_vec())
    }

    /// Check if there's an edge between two nodes.
    fn has_edge(&self, from_node: usize, to_node: usize) -> bool {
        self.inner.has_edge(from_node, to_node)
    }

    /// Get neighbor indices for a node.
    fn neighbor_indices(&self, node: usize) -> Vec<usize> {
        self.inner.neighbor_indices(node)
    }

    /// Get the degree of a specific node.
    fn degree(&self, node: usize) -> Option<usize> {
        self.inner.degree(node)
    }

    /// Get features for a specific node (returns the underlying HashMap).
    fn node_features(&self, node: usize) -> Option<std::collections::HashMap<String, f64>> {
        self.inner.node_features(node).map(|f| f.clone())
    }

    /// Check if the graph is directed.
    fn is_directed(&self) -> bool {
        self.inner.directed
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "VisibilityGraph(nodes={}, edges={}, density={:.4})",
            self.inner.node_count,
            self.inner.edges().len(),
            self.inner.density()
        )
    }
}

/// Python wrapper for GraphStatistics.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "GraphStatistics")]
pub struct PyGraphStatistics {
    inner: crate::analysis::statistics::GraphStatistics,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyGraphStatistics {
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count
    }

    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.is_directed
    }

    #[getter]
    fn average_degree(&self) -> f64 {
        self.inner.average_degree
    }

    #[getter]
    fn min_degree(&self) -> usize {
        self.inner.min_degree
    }

    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree
    }

    #[getter]
    fn degree_std_dev(&self) -> f64 {
        self.inner.degree_std_dev
    }

    #[getter]
    fn degree_variance(&self) -> f64 {
        self.inner.degree_variance
    }

    #[getter]
    fn average_clustering(&self) -> f64 {
        self.inner.average_clustering
    }

    #[getter]
    fn global_clustering(&self) -> f64 {
        self.inner.global_clustering
    }

    #[getter]
    fn average_path_length(&self) -> f64 {
        self.inner.average_path_length
    }

    #[getter]
    fn diameter(&self) -> usize {
        self.inner.diameter
    }

    #[getter]
    fn radius(&self) -> usize {
        self.inner.radius
    }

    #[getter]
    fn density(&self) -> f64 {
        self.inner.density
    }

    #[getter]
    fn is_connected(&self) -> bool {
        self.inner.is_connected
    }

    #[getter]
    fn num_components(&self) -> usize {
        self.inner.num_components
    }

    #[getter]
    fn largest_component_size(&self) -> usize {
        self.inner.largest_component_size
    }

    #[getter]
    fn assortativity(&self) -> f64 {
        self.inner.assortativity
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.inner.feature_count
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Python wrapper for MotifCounts.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "MotifCounts")]
pub struct PyMotifCounts {
    inner: crate::analysis::motifs::MotifCounts,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyMotifCounts {
    /// Get counts as a dictionary {motif_name: count}
    fn counts(&self) -> std::collections::HashMap<String, usize> {
        self.inner.counts.clone()
    }

    #[getter]
    fn total_subgraphs(&self) -> usize {
        self.inner.total_subgraphs
    }

    /// Get count for a specific motif type
    fn get(&self, motif_name: &str) -> Option<usize> {
        self.inner.counts.get(motif_name).copied()
    }

    fn __repr__(&self) -> String {
        format!("MotifCounts(counts={:?}, total={})", self.inner.counts, self.inner.total_subgraphs)
    }
}

/// Python wrapper for BuiltinFeature enum
#[cfg(feature = "python-bindings")]
#[pyclass(name = "BuiltinFeature")]
#[derive(Clone)]
pub struct PyBuiltinFeature {
    inner: RustBuiltinFeature,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyBuiltinFeature {
    #[classattr]
    const DELTA_FORWARD: &'static str = "DeltaForward";
    #[classattr]
    const DELTA_BACKWARD: &'static str = "DeltaBackward";
    #[classattr]
    const DELTA_SYMMETRIC: &'static str = "DeltaSymmetric";
    #[classattr]
    const LOCAL_SLOPE: &'static str = "LocalSlope";
    #[classattr]
    const ACCELERATION: &'static str = "Acceleration";
    #[classattr]
    const LOCAL_MEAN: &'static str = "LocalMean";
    #[classattr]
    const LOCAL_VARIANCE: &'static str = "LocalVariance";
    #[classattr]
    const IS_LOCAL_MAX: &'static str = "IsLocalMax";
    #[classattr]
    const IS_LOCAL_MIN: &'static str = "IsLocalMin";
    #[classattr]
    const ZSCORE: &'static str = "ZScore";

    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "DeltaForward" => RustBuiltinFeature::DeltaForward,
            "DeltaBackward" => RustBuiltinFeature::DeltaBackward,
            "DeltaSymmetric" => RustBuiltinFeature::DeltaSymmetric,
            "LocalSlope" => RustBuiltinFeature::LocalSlope,
            "Acceleration" => RustBuiltinFeature::Acceleration,
            "LocalMean" => RustBuiltinFeature::LocalMean,
            "LocalVariance" => RustBuiltinFeature::LocalVariance,
            "IsLocalMax" => RustBuiltinFeature::IsLocalMax,
            "IsLocalMin" => RustBuiltinFeature::IsLocalMin,
            "ZScore" => RustBuiltinFeature::ZScore,
            _ => return Err(py_error_helpers::to_value_error(
                format!("Unknown feature: {}", name)
            )),
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("BuiltinFeature({:?})", self.inner)
    }
}

/// Python wrapper for FeatureSet
#[cfg(feature = "python-bindings")]
#[pyclass(name = "FeatureSet")]
pub struct PyFeatureSet {
    inner: RustFeatureSet<f64>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyFeatureSet {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustFeatureSet::new(),
        }
    }

    /// Add a builtin feature to the set (builder pattern - returns self)
    fn add_builtin(mut slf: PyRefMut<'_, Self>, feature: PyBuiltinFeature) -> PyRefMut<'_, Self> {
        // Take ownership of inner, add feature, put it back
        let inner = std::mem::replace(&mut slf.inner, RustFeatureSet::new());
        slf.inner = inner.add_builtin(feature.inner);
        slf
    }

    fn __repr__(&self) -> String {
        "FeatureSet(...)".to_string()
    }
}

/// Standalone function to create natural visibility graph from array
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn natural_visibility(values: Vec<f64>) -> PyResult<PyVisibilityGraph> {
    let series = RustTimeSeries::from_raw(values)
        .map_err(py_error_helpers::to_value_error)?;
    let graph = crate::VisibilityGraph::from_series(&series)
        .natural_visibility()
        .map_err(py_error_helpers::to_runtime_error)?;
    Ok(PyVisibilityGraph { inner: graph })
}

/// Standalone function to create horizontal visibility graph from array
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn horizontal_visibility(values: Vec<f64>) -> PyResult<PyVisibilityGraph> {
    let series = RustTimeSeries::from_raw(values)
        .map_err(py_error_helpers::to_value_error)?;
    let graph = crate::VisibilityGraph::from_series(&series)
        .horizontal_visibility()
        .map_err(py_error_helpers::to_runtime_error)?;
    Ok(PyVisibilityGraph { inner: graph })
}

/// Python module initialization.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn _rustygraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyVisibilityGraph>()?;
    m.add_class::<PyBuiltinFeature>()?;
    m.add_class::<PyFeatureSet>()?;
    m.add_class::<PyMissingDataStrategy>()?;
    m.add_class::<PyGraphStatistics>()?;
    m.add_class::<PyMotifCounts>()?;
    m.add_function(wrap_pyfunction!(natural_visibility, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_visibility, m)?)?;
    Ok(())
}

/// Helper functions for consistent error handling in Python bindings.
#[cfg(feature = "python-bindings")]
mod py_error_helpers {
    use pyo3::PyErr;
    use pyo3::exceptions::{PyValueError, PyRuntimeError};

    /// Convert a Rust error to PyValueError.
    pub fn to_value_error<E: std::fmt::Display>(error: E) -> PyErr {
        PyErr::new::<PyValueError, _>(error.to_string())
    }

    /// Convert a Rust error to PyRuntimeError.
    pub fn to_runtime_error<E: std::fmt::Display>(error: E) -> PyErr {
        PyErr::new::<PyRuntimeError, _>(error.to_string())
    }
}

#[cfg(test)]
#[cfg(feature = "python-bindings")]
mod tests {
    use super::*;
    
    #[test]
    fn test_py_timeseries_creation() {
        let series = PyTimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(series.__len__(), 4);
    }
}
