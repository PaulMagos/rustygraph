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
use pyo3::types::PyList;
#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2};

#[cfg(feature = "python-bindings")]
use crate::{TimeSeries as RustTimeSeries, VisibilityGraph as RustVisibilityGraph};

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
    
    /// Returns the values as a NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
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
    fn adjacency_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let matrix = self.inner.to_adjacency_matrix();
        let n = self.inner.node_count;
        let flat: Vec<f64> = matrix.into_iter().flatten().collect();
        PyArray2::from_vec2(py, &vec![flat; 1]).unwrap()
            .reshape([n, n]).unwrap()
    }
    
    /// Exports the graph to JSON.
    fn to_json(&self) -> String {
        self.inner.to_json(crate::ExportOptions::default())
    }
    
    /// Computes betweenness centrality for a node.
    fn betweenness_centrality(&self, node: usize) -> f64 {
        self.inner.betweenness_centrality(node)
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

/// Python module initialization.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn rustygraph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyVisibilityGraph>()?;
    
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

