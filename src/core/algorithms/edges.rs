//! Horizontal and Natural visibility algorithm implementation.
//!
//! The horizontal visibility algorithm is a simpler variant where two points
//! can "see" each other if all intermediate values are strictly lower than
//! both endpoints.
//!
//! # Algorithm
//!
//! Two points (i, yi) and (j, yj) are connected if for all k between i and j:
//!
//! ```text
//! yk < min(yi, yj)
//! ```
//!
//! The natural visibility algorithm connects two data points if they can "see"
//! each other - that is, if the line segment between them is not blocked by
//! any intermediate point.
//!
//! # Algorithm
//!
//! For each pair of points (i, yi) and (j, yj), they are connected if all
//! intermediate points (k, yk) satisfy:
//!
//! ```text
//! yk < yi + (yj - yi) * (tk - ti) / (tj - ti)
//!
//! # Examples
//!
//! ```rust
//! use rustygraph::algorithms::horizontal::compute_edges;
//!
//! let series = vec![1.0, 3.0, 2.0, 4.0, 1.0];
//! let edges = compute_edges(&series);
//!
//! println!("Horizontal visibility edges: {:?}", edges);
//! ```
//!
//! See the main `VisibilityGraph` API for usage examples.
//!
//! # References
//!
//! Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
//! "Horizontal visibility graphs: Exact results for random time series."
//! Physical Review E, 80(4), 046103.
//!
//! Lacasa, L., et al. (2008). "From time series to complex networks:
//! The visibility graph." PNAS, 105(13), 4972-4975.
//!

/// Computes visibility edges.
///
/// # Arguments
///
/// - `series`: Input time series data
///
/// # Returns
///
/// Hashmap of edges as (source, target) pairs with weights
use crate::core::TimeSeries;

/// Visibility graph algorithm type.
///
/// Determines which visibility criterion is used to connect nodes.
#[derive(Debug, Clone, Copy)]
pub enum VisibilityType {
    /// Natural visibility: nodes connected if line-of-sight is not blocked
    Natural,
    /// Horizontal visibility: nodes connected if all intermediate values are lower
    Horizontal,
}

/// Visibility edges computation with custom weight function.
///
/// This struct provides a flexible way to compute visibility graph edges
/// with custom edge weights.
///
/// # Type Parameters
///
/// - `T`: Numeric type for time series values
/// - `F`: Weight function type `Fn(usize, usize, T, T) -> f64`
pub struct VisibilityEdges<'a, T, F>
where
    T: Copy + PartialOrd + Into<f64>,
    F: Fn(usize, usize, T, T) -> f64,
{
    series: &'a TimeSeries<T>,
    rule: VisibilityType,
    weight_fn: F,
}

impl<'a, T, F> VisibilityEdges<'a, T, F>
where
    T: Copy + PartialOrd + Into<f64>,
    F: Fn(usize, usize, T, T) -> f64,
{
    /// Creates a new visibility edges computation instance.
    ///
    /// # Arguments
    ///
    /// - `series`: Time series data
    /// - `rule`: Visibility algorithm type
    /// - `weight_fn`: Function to compute edge weights `(src_idx, dst_idx, src_val, dst_val) -> weight`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, algorithms::{VisibilityEdges, VisibilityType}};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let edges = VisibilityEdges::new(
    ///     &series,
    ///     VisibilityType::Natural,
    ///     |_, _, vi: f64, vj: f64| (vj - vi).abs()
    /// ).compute_edges();
    /// ```
    pub fn new(series: &'a TimeSeries<T>, rule: VisibilityType, weight_fn: F) -> Self {
        Self {
            series,
            rule,
            weight_fn,
        }
    }

    /// Computes all visibility edges in the time series.
    ///
    /// Returns a hashmap of directed edges with their computed weights.
    ///
    /// # Returns
    ///
    /// `super::fast::EdgeMap` where keys are `(source, target)` node indices
    /// and values are edge weights computed by the weight function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, algorithms::{VisibilityEdges, VisibilityType}};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    /// let edges = VisibilityEdges::new(
    ///     &series,
    ///     VisibilityType::Natural,
    ///     |_, _, _, _| 1.0
    /// ).compute_edges();
    ///
    /// println!("Found {} edges", edges.len());
    /// ```
    pub fn compute_edges(&self) -> super::fast::EdgeMap {
        self.compute(false)
    }

    /// Runs the fast kernel on the valid (present, finite) points and applies
    /// the weight function.
    ///
    /// Missing (`None`) and non-finite points are treated as absent: they get
    /// no edges and do not block visibility. Positions are the series
    /// timestamps when they are finite and strictly increasing, otherwise the
    /// sample indices.
    fn compute(&self, parallel: bool) -> super::fast::EdgeMap {
        use super::fast::{self, Explicit, Uniform};

        let values = &self.series.values;
        let mut idx: Vec<usize> = Vec::with_capacity(values.len());
        let mut y: Vec<f64> = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            if let Some(v) = v {
                let f: f64 = (*v).into();
                if f.is_finite() {
                    idx.push(i);
                    y.push(f);
                }
            }
        }

        let ts: Vec<f64> = self.series.timestamps.iter().map(|&t| t.into()).collect();
        let ts_ok = ts.len() == values.len()
            && ts.iter().all(|t| t.is_finite())
            && ts.windows(2).all(|w| w[1] > w[0]);
        let dense = idx.len() == values.len();
        let unit = ts_ok && ts.iter().enumerate().all(|(i, &t)| t == i as f64);

        let raw = match self.rule {
            VisibilityType::Horizontal => fast::horizontal_edges(&y),
            VisibilityType::Natural => {
                if dense && (unit || !ts_ok) {
                    run_natural(&y, &Uniform, parallel)
                } else {
                    let x: Vec<f64> = if ts_ok {
                        idx.iter().map(|&i| ts[i]).collect()
                    } else {
                        idx.iter().map(|&i| i as f64).collect()
                    };
                    run_natural(&y, &Explicit(&x), parallel)
                }
            }
        };

        let mut edges = super::fast::EdgeMap::with_capacity_and_hasher(raw.len(), Default::default());
        for (a, b) in raw {
            let (src, dst) = (idx[a as usize], idx[b as usize]);
            let (vs, vd) = (values[src].unwrap(), values[dst].unwrap());
            edges.insert((src, dst), (self.weight_fn)(src, dst, vs, vd));
        }
        edges
    }
}

fn run_natural<P: super::fast::Positions>(y: &[f64], pos: &P, parallel: bool) -> Vec<super::fast::Edge> {
    #[cfg(feature = "parallel")]
    {
        if parallel {
            return super::fast::natural_edges(y, pos);
        }
    }
    let _ = parallel;
    super::fast::natural_edges_seq(y, pos)
}

/// Parallel edge computation (when parallel feature is enabled).
#[cfg(feature = "parallel")]
impl<'a, T, F> VisibilityEdges<'a, T, F>
where
    T: Copy + PartialOrd + Into<f64> + Send + Sync,
    F: Fn(usize, usize, T, T) -> f64 + Send + Sync,
{
    /// Computes edges, running the natural-visibility kernel in parallel for
    /// large series (see [`super::fast::PARALLEL_THRESHOLD`]). Output is
    /// identical to [`Self::compute_edges`].
    pub fn compute_edges_parallel(&self) -> super::fast::EdgeMap {
        self.compute(true)
    }
}
