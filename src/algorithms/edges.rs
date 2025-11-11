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

// Visibility type enum
#[derive(Debug, Clone, Copy)]
pub enum VisibilityType {
    Natural,
    Horizontal,
}

// Visibility edges computation struct with custom weight function
pub struct VisibilityEdges<'a, T, F>
where
    F: Fn(usize, usize, f64, f64) -> f64,
{
    series: &'a [T],
    rule: VisibilityType,
    weight_fn: F,
}

// Implementation of the visibility edges computation
impl<'a, T, F> VisibilityEdges<'a, T, F>
where
    F: Fn(usize, usize, f64, f64) -> f64,
{

    // Creates a new VisibilityEdges instance
    pub fn new(series: &'a [T], rule: VisibilityType, weight_fn: F) -> Self {
        Self {
            series,
            rule,
            weight_fn,
        }
    }

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
            if Self::should_pop(self.series, k, j, i) {
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
                let w = (self.weight_fn)(j, i, self.series[j], self.series[i]);
                edges.insert((j, i), w);
            } else if matches!(self.rule, VisibilityType::Horizontal) {
                break;
            }
        }
    }

    // Determines if the point at index j is visible from point i based on the visibility rule
    fn is_visible(&self, j: usize, i: usize) -> bool {
        match self.rule {
            VisibilityType::Natural => Self::is_visible_natural(self.series, j, i),
            VisibilityType::Horizontal => Self::is_visible_horizontal(self.series, j, i),
        }
    }

    // Determines if the point at index j is visible from point i in natural visibility
    fn is_visible_natural(series: &[f64], j: usize, i: usize) -> bool {
        (j + 1..i).all(|k| {
            let line_height =
                series[j] + (series[i] - series[j]) * ((k - j) as f64 / (i - j) as f64);
            series[k] < line_height
        })
    }


    // Determines if the point at index j is visible from point i in horizontal visibility
    fn is_visible_horizontal(series: &[f64], j: usize, i: usize) -> bool {
        let min_h = series[j].min(series[i]);
        (j + 1..i).all(|k| series[k] < min_h)
    }

    // Determines if the point at index j should be popped from the envelope stack
    fn should_pop(series: &[f64], k: usize, j: usize, i: usize) -> bool {
        let slope_kj = (series[j] - series[k]) / ((j - k) as f64);
        let slope_ki = (series[i] - series[k]) / ((i - k) as f64);
        slope_ki >= slope_kj
    }
}
