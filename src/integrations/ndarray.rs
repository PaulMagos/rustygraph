//! Integration with ndarray for matrix operations.
//!
//! This module provides conversions between RustyGraph and ndarray types,
//! enabling efficient matrix-based computations.
//!
//! Requires `ndarray-support` feature.

#[cfg(feature = "ndarray-support")]
use ndarray::{Array1, Array2};

use crate::{TimeSeries, VisibilityGraph};

#[cfg(feature = "ndarray-support")]
impl TimeSeries<f64> {
    /// Creates a time series from an ndarray Array1.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "ndarray-support")]
    /// # {
    /// use rustygraph::TimeSeries;
    /// use ndarray::array;
    ///
    /// let data = array![1.0, 2.0, 3.0, 4.0];
    /// let series = TimeSeries::from_ndarray(data).unwrap();
    /// # }
    /// ```
    pub fn from_ndarray(data: Array1<f64>) -> Result<Self, Box<dyn std::error::Error>> {
        let values: Vec<f64> = data.to_vec();
        Ok(Self::from_raw(values)?)
    }

    /// Converts the time series to an ndarray Array1.
    ///
    /// Returns only the valid (non-None) values.
    pub fn to_ndarray(&self) -> Array1<f64> {
        let values: Vec<f64> = self.values.iter()
            .filter_map(|&v| v)
            .collect();
        Array1::from_vec(values)
    }

    /// Converts the time series to an ndarray Array1 with Option values.
    pub fn to_ndarray_optional(&self) -> Array1<Option<f64>> {
        Array1::from_vec(self.values.clone())
    }
}

#[cfg(feature = "ndarray-support")]
impl<T: Copy + Into<f64>> VisibilityGraph<T> {
    /// Returns the adjacency matrix as an ndarray Array2.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "ndarray-support")]
    /// # {
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let adj_matrix = graph.to_ndarray_adjacency();
    /// println!("Adjacency matrix shape: {:?}", adj_matrix.shape());
    /// # }
    /// ```
    pub fn to_ndarray_adjacency(&self) -> Array2<f64> {
        let n = self.node_count;
        let mut matrix = Array2::<f64>::zeros((n, n));

        for (&(src, dst), &weight) in &self.edges {
            matrix[[src, dst]] = weight;
            if !self.directed {
                matrix[[dst, src]] = weight;
            }
        }

        matrix
    }

    /// Returns the Laplacian matrix as an ndarray Array2.
    ///
    /// The Laplacian matrix is defined as L = D - A, where D is the degree
    /// matrix and A is the adjacency matrix.
    pub fn to_ndarray_laplacian(&self) -> Array2<f64> {
        let adj = self.to_ndarray_adjacency();
        let n = self.node_count;
        let mut laplacian = Array2::<f64>::zeros((n, n));

        // Compute degree matrix
        for i in 0..n {
            let degree: f64 = adj.row(i).sum();
            laplacian[[i, i]] = degree;
        }

        // Subtract adjacency matrix
        laplacian - adj
    }

    /// Returns node features as an ndarray Array2.
    ///
    /// Each row represents a node, each column a feature.
    pub fn to_ndarray_features(&self) -> Option<Array2<f64>> {
        if self.node_features.is_empty() {
            return None;
        }

        // Get all feature names
        let mut feature_names: Vec<String> = Vec::new();
        for features in &self.node_features {
            for name in features.keys() {
                if !feature_names.contains(name) {
                    feature_names.push(name.clone());
                }
            }
        }

        if feature_names.is_empty() {
            return None;
        }

        // Create matrix
        let n_nodes = self.node_count;
        let n_features = feature_names.len();
        let mut matrix = Array2::<f64>::zeros((n_nodes, n_features));

        for (i, features) in self.node_features.iter().enumerate() {
            for (j, name) in feature_names.iter().enumerate() {
                if let Some(&value) = features.get(name) {
                    matrix[[i, j]] = value.into();
                }
            }
        }

        Some(matrix)
    }

    /// Returns the degree sequence as an ndarray Array1.
    pub fn to_ndarray_degrees(&self) -> Array1<usize> {
        let degrees = self.degree_sequence();
        Array1::from_vec(degrees)
    }
}

/// Advanced matrix operations using ndarray.
#[cfg(feature = "ndarray-support")]
pub mod matrix_ops {
    use super::*;
    use ndarray::Array1;

    impl<T: Copy + Into<f64>> VisibilityGraph<T> {
        /// Computes eigenvalues of the adjacency matrix (approximation).
        ///
        /// Note: This is a simple power iteration method for the dominant eigenvalue.
        /// For full eigendecomposition, use external libraries like ndarray-linalg.
        pub fn dominant_eigenvalue(&self, iterations: usize) -> f64 {
            let adj = self.to_ndarray_adjacency();
            let n = self.node_count;

            if n == 0 {
                return 0.0;
            }

            // Power iteration
            let mut v = Array1::<f64>::ones(n);
            v = &v / v.dot(&v).sqrt();

            for _ in 0..iterations {
                let v_new = adj.dot(&v);
                let norm = v_new.dot(&v_new).sqrt();
                if norm > 0.0 {
                    v = v_new / norm;
                } else {
                    break;
                }
            }

            // Rayleigh quotient
            let av = adj.dot(&v);
            v.dot(&av)
        }

        /// Computes the graph energy (sum of absolute eigenvalues).
        ///
        /// This is an approximation using the trace and norm.
        pub fn graph_energy_approx(&self) -> f64 {
            let adj = self.to_ndarray_adjacency();

            // Energy approximation: sqrt(trace(A^2))
            let adj_squared = adj.dot(&adj);
            let trace: f64 = (0..self.node_count)
                .map(|i| adj_squared[[i, i]])
                .sum();

            trace.sqrt()
        }

        /// Performs random walk on the graph using matrix operations.
        ///
        /// Returns the stationary distribution.
        pub fn random_walk_stationary(&self, iterations: usize) -> Array1<f64> {
            let adj = self.to_ndarray_adjacency();
            let n = self.node_count;

            // Compute transition matrix
            let mut trans = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                let row_sum: f64 = adj.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n {
                        trans[[i, j]] = adj[[i, j]] / row_sum;
                    }
                }
            }

            // Start with uniform distribution
            let mut dist = Array1::<f64>::ones(n) / (n as f64);

            // Iterate
            for _ in 0..iterations {
                dist = trans.t().dot(&dist);
            }

            dist
        }
    }
}

#[cfg(test)]
#[cfg(feature = "ndarray-support")]
mod tests {
    use super::*;

    #[test]
    fn test_from_ndarray() {
        use ndarray::array;

        let data = array![1.0, 2.0, 3.0, 4.0];
        let series = TimeSeries::from_ndarray(data).unwrap();
        assert_eq!(series.len(), 4);
    }

    #[test]
    fn test_adjacency_matrix() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let adj = graph.to_ndarray_adjacency();
        assert_eq!(adj.shape(), &[4, 4]);
    }

    #[test]
    fn test_laplacian() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();

        let lap = graph.to_ndarray_laplacian();
        assert_eq!(lap.shape(), &[4, 4]);
    }
}

