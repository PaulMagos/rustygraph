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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyVisibilityGraph { inner: graph })
    }
    
    /// Creates a horizontal visibility graph.
    fn horizontal_visibility(&self) -> PyResult<PyVisibilityGraph> {
        let graph = RustVisibilityGraph::from_series(&self.inner)
            .horizontal_visibility()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyVisibilityGraph { inner: graph })
    }

    /// Returns the values as a NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self.inner.values.iter().filter_map(|&v| v).collect();
        PyArray1::from_vec(py, values)
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
        self.inner.to_json(crate::ExportOptions::default())
    }
    
    /// Get feature names for a specific node
    fn get_node_feature_names(&self, node: usize) -> PyResult<Vec<String>> {
        if node >= self.inner.node_count {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node {} out of range (graph has {} nodes)", node, self.inner.node_count)
            ));
        }

        if node >= self.inner.node_features.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No features computed. Use natural_visibility_with_features() or horizontal_visibility_with_features()"
            ));
        }

        let features = &self.inner.node_features[node];
        Ok(features.keys().cloned().collect())
    }

    /// Get features for a specific node as a dictionary {feature_name: value}
    fn get_node_features(&self, node: usize) -> PyResult<std::collections::HashMap<String, f64>> {
        if node >= self.inner.node_count {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node {} out of range (graph has {} nodes)", node, self.inner.node_count)
            ));
        }

        if node >= self.inner.node_features.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No features computed. Use natural_visibility_with_features() or horizontal_visibility_with_features()"
            ));
        }

        Ok(self.inner.node_features[node].clone())
    }

    /// Get all node features as a NumPy array (nodes x features)
    /// Returns array where row i contains features for node i
    fn get_all_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.inner.node_features.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node {} not found in graph", node)
            ))
    }
    
    /// Detects communities and returns node assignments.
    fn detect_communities(&self) -> Vec<usize> {
        self.inner.detect_communities().node_communities
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
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let graph = crate::VisibilityGraph::from_series(&series)
        .natural_visibility()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyVisibilityGraph { inner: graph })
}

/// Standalone function to create horizontal visibility graph from array
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn horizontal_visibility(values: Vec<f64>) -> PyResult<PyVisibilityGraph> {
    let series = RustTimeSeries::from_raw(values)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let graph = crate::VisibilityGraph::from_series(&series)
        .horizontal_visibility()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
    m.add_function(wrap_pyfunction!(natural_visibility, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_visibility, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "High-performance visibility graph computation for time series data")?;
    
    Ok(())
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

