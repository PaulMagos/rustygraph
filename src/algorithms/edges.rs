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
//!
//! ```rust
//! use rustygraph::algorithms::natural::compute_edges;
//!
//! let series = vec![1.0, 3.0, 2.0, 4.0, 1.0];
//! let edges = compute_edges(&series);
//!
//! println!("Natural visibility edges: {:?}", edges);
//! ```
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
///
/// # Examples
///
/// ```rust
/// use rustygraph::algorithms::{create_visibility_edges, natural_visibility};
///
/// let data = vec![1.0, 3.0, 2.0, 4.0, 1.0];
/// let edges = create_visibility_edges{
///     &data,
///     natural_visibility,
///     |_, _, vi, vj| (vj - vi).abs()
/// }.compute_edges();
///
/// for (src, dst), weight in edges {
///     println!("{} -> {}: {}", src, dst, weight);
/// }
/// ```
use std::collections::HashMap;
use crate::TimeSeries;

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
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
    /// let edges = VisibilityEdges::new(
    ///     &series,
    ///     VisibilityType::Natural,
    ///     |_, _, vi, vj| (vj - vi).abs()
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
    /// `HashMap<(usize, usize), f64>` where keys are `(source, target)` node indices
    /// and values are edge weights computed by the weight function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, algorithms::{VisibilityEdges, VisibilityType}};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]);
    /// let edges = VisibilityEdges::new(
    ///     &series,
    ///     VisibilityType::Natural,
    ///     |_, _, _, _| 1.0
    /// ).compute_edges();
    ///
    /// println!("Found {} edges", edges.len());
    /// ```
    pub fn compute_edges(&self) -> HashMap<(usize, usize), f64> {
        let mut edges = HashMap::new();
        let mut stack: Vec<usize> = Vec::new();

        // Process each point in the series
        for i in 0..self.series.len() {
            // Update the envelope stack based on visibility rule
            self.update_envelope(&mut stack, i);

            // Add visible edges from points in the stack to point i
            self.add_visible_edges(&mut edges, &stack, i);

            // Push the current point onto the stack
            stack.push(i);
        }

        // Return the computed edges
        edges
    }

    fn update_envelope(&self, stack: &mut Vec<usize>, i: usize) {
        // Only update envelope for natural visibility
        if !matches!(self.rule, VisibilityType::Natural) {
            return;
        }

        // Maintain the upper envelope stack
        while stack.len() >= 2 {
            // Get the last two points in the stack
            let j = *stack.last().unwrap();
            let k = stack[stack.len() - 2];
            // Check if point j should be popped
            if self.should_pop(k, j, i) {
                stack.pop();
            } else {
                break;
            }
        }
    }

    // Adds visible edges from points in the stack to point i
    fn add_visible_edges(
        &self,
        edges: &mut HashMap<(usize, usize), f64>,
        stack: &[usize],
        i: usize,
    ) {
        // Check visibility from each point in the stack to point i
        for &j in stack.iter().rev() {
            if self.is_visible(j, i) {
                // Unwrap is safe here as we only process non-None values
                let vj = self.series.values[j].unwrap();
                let vi = self.series.values[i].unwrap();
                let w = (self.weight_fn)(j, i, vj, vi);
                edges.insert((j, i), w);
            } else if matches!(self.rule, VisibilityType::Horizontal) {
                break;
            }
        }
    }

    // Determines if the point at index j is visible from point i based on the visibility rule
    fn is_visible(&self, j: usize, i: usize) -> bool {
        match self.rule {
            VisibilityType::Natural => self.is_visible_natural(j, i),
            VisibilityType::Horizontal => self.is_visible_horizontal(j, i),
        }
    }

    // Determines if the point at index j is visible from point i in natural visibility
    fn is_visible_natural(&self, j: usize, i: usize) -> bool {
        (j + 1..i).all(|k| {
            let vj: f64 = self.series.values[j].unwrap().into();
            let vi: f64 = self.series.values[i].unwrap().into();
            let vk: f64 = self.series.values[k].unwrap().into();
            let line_height = vj + (vi - vj) * ((k - j) as f64 / (i - j) as f64);
            vk < line_height
        })
    }


    // Determines if the point at index j is visible from point i in horizontal visibility
    fn is_visible_horizontal(&self, j: usize, i: usize) -> bool {
        let vj = self.series.values[j].unwrap();
        let vi = self.series.values[i].unwrap();
        let min_h = if vj < vi { vj } else { vi };
        (j + 1..i).all(|k| self.series.values[k].unwrap() < min_h)
    }

    // Determines if the point at index j should be popped from the envelope stack
    fn should_pop(&self, k: usize, j: usize, i: usize) -> bool {
        let vk: f64 = self.series.values[k].unwrap().into();
        let vj: f64 = self.series.values[j].unwrap().into();
        let vi: f64 = self.series.values[i].unwrap().into();
        let slope_kj = (vj - vk) / ((j - k) as f64);
        let slope_ki = (vi - vk) / ((i - k) as f64);
        // Use strict inequality: point j is only hidden if slope_ki is strictly greater
        // Points on the line (equal slopes) should remain visible
        slope_ki > slope_kj
    }
}
