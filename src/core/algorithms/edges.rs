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
use std::collections::HashMap;
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
    /// `HashMap<(usize, usize), f64>` where keys are `(source, target)` node indices
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
    pub fn compute_edges(&self) -> HashMap<(usize, usize), f64> {

        // Sequential implementation with O(n) envelope optimization
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
        // Remove points that are dominated by the convex hull
        while stack.len() >= 2 {
            let j = *stack.last().unwrap();
            let k = stack[stack.len() - 2];

            // Check if point j should be popped from the envelope
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
        let vj: f64 = self.series.values[j].unwrap().into();
        let vi: f64 = self.series.values[i].unwrap().into();

        // Use SIMD optimization when available and beneficial (x86_64 AVX2 or ARM NEON)
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            if i - j > 8 {
                // Collect intermediate values for SIMD processing
                let intermediate: Vec<f64> = (j + 1..i)
                    .map(|k| self.series.values[k].unwrap().into())
                    .collect();
                return crate::performance::simd::SimdOps::is_visible_natural_simd(
                    vj, vi, &intermediate, j, i
                );
            }
        }

        // Standard scalar implementation
        (j + 1..i).all(|k| {
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

        // Use SIMD optimization when available and beneficial (x86_64 AVX2 or ARM NEON)
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            if i - j > 8 {
                let intermediate: Vec<f64> = (j + 1..i)
                    .map(|k| self.series.values[k].unwrap().into())
                    .collect();
                return crate::performance::simd::SimdOps::is_visible_horizontal_simd(
                    vj.into(), vi.into(), &intermediate
                );
            }
        }

        // Standard scalar implementation
        (j + 1..i).all(|k| self.series.values[k].unwrap() < min_h)
    }

    // Determines if the point at index j should be popped from the envelope stack
    //
    // For natural visibility graphs, we can only safely remove point j if:
    // 1. Point j cannot see point i (blocked by the line from k to i), AND
    // 2. Point j will be blocked from seeing ALL future points beyond i
    //
    // IMPORTANT: We must NEVER remove j if it's adjacent to i (j+1 == i),
    // because adjacent points always have visibility (no intermediate points to block).
    fn should_pop(&self, k: usize, j: usize, i: usize) -> bool {
        // Safety check: NEVER remove a node that's adjacent to the current node
        if self.is_adjacent(j, i) {
            return false;
        }

        let (vk, vj, vi) = self.get_values(k, j, i);
        let (tk, tj, ti) = (k as f64, j as f64, i as f64);

        let expected_height = self.calculate_expected_height(vk, vi, tk, tj, ti);

        if vj < expected_height {
            self.is_permanently_shadowed(vk, vj, vi, tk, tj, ti)
        } else {
            false
        }
    }

    /// Check if two indices are adjacent
    fn is_adjacent(&self, j: usize, i: usize) -> bool {
        j + 1 >= i
    }

    /// Get values for three indices
    fn get_values(&self, k: usize, j: usize, i: usize) -> (f64, f64, f64) {
        (
            self.series.values[k].unwrap().into(),
            self.series.values[j].unwrap().into(),
            self.series.values[i].unwrap().into(),
        )
    }

    /// Calculate expected height of line from k to i at position j
    fn calculate_expected_height(&self, vk: f64, vi: f64, tk: f64, tj: f64, ti: f64) -> f64 {
        vk + (vi - vk) * ((tj - tk) / (ti - tk))
    }

    /// Check if point j is in a permanently shadowed position
    fn is_permanently_shadowed(&self, vk: f64, vj: f64, vi: f64, tk: f64, tj: f64, ti: f64) -> bool {
        let slope_kj = (vj - vk) / (tj - tk);
        let slope_ki = (vi - vk) / (ti - tk);
        slope_kj < slope_ki
    }
}
/// Parallel edge computation (when parallel feature is enabled).
///
/// This implementation splits the work of computing edges across multiple threads,
/// providing significant speedup for large graphs.
#[cfg(feature = "parallel")]
impl<'a, T, F> VisibilityEdges<'a, T, F>
where
    T: Copy + PartialOrd + Into<f64> + Send + Sync,
    F: Fn(usize, usize, T, T) -> f64 + Send + Sync,
{
    /// Computes edges in parallel using Rayon with O(n) envelope optimization per chunk.
    ///
    /// This method processes chunks of the time series in parallel, using the
    /// O(n) envelope optimization within each chunk for efficiency.
    ///
    /// # Strategy
    ///
    /// - Splits the series into chunks (one per thread)
    /// - Each chunk uses the sequential O(n) envelope algorithm
    /// - Results are merged at the end
    ///
    /// # Performance
    ///
    /// Expected speedup: 2-4x on multi-core systems (4-8 cores)
    /// Complexity: O(n²/p) where p is the number of threads
    ///
    /// # Returns
    ///
    /// HashMap of edges with weights, same as sequential version
    pub fn compute_edges_parallel(&self) -> HashMap<(usize, usize), f64> {
        let n = self.series.len();
        if self.should_use_sequential(n) {
            return self.compute_edges();
        }

        let chunk_results = self.process_chunks_in_parallel(n);
        self.merge_chunk_results(chunk_results)
    }

    /// Check if sequential processing is better for small graphs
    fn should_use_sequential(&self, n: usize) -> bool {
        n <= 100
    }

    /// Process chunks in parallel
    fn process_chunks_in_parallel(&self, n: usize) -> Vec<HashMap<(usize, usize), f64>> {
        use rayon::prelude::*;

        (0..n)
            .collect::<Vec<_>>()
            .par_chunks(64)
            .map(|target_chunk| self.process_target_chunk(target_chunk))
            .collect()
    }

    /// Process a single chunk of target nodes
    fn process_target_chunk(&self, target_chunk: &[usize]) -> HashMap<(usize, usize), f64> {
        let mut local_edges = HashMap::new();

        for &i in target_chunk {
            let stack = self.build_envelope_stack_for_target(i);
            self.add_visible_edges_from_stack(&mut local_edges, &stack, i);
        }

        local_edges
    }

    /// Build envelope stack for a target node
    fn build_envelope_stack_for_target(&self, target: usize) -> Vec<usize> {
        let mut stack = Vec::new();

        for j in 0..target {
            self.update_envelope_for_parallel(&mut stack, j);
            stack.push(j);
        }

        stack
    }

    /// Update envelope during parallel processing
    fn update_envelope_for_parallel(&self, stack: &mut Vec<usize>, j: usize) {
        if matches!(self.rule, VisibilityType::Natural) {
            while stack.len() >= 2 {
                let prev_j = *stack.last().unwrap();
                let prev_k = stack[stack.len() - 2];
                if self.should_pop(prev_k, prev_j, j) {
                    stack.pop();
                } else {
                    break;
                }
            }
        }
    }

    /// Add visible edges from stack to target node
    fn add_visible_edges_from_stack(
        &self,
        edges: &mut HashMap<(usize, usize), f64>,
        stack: &[usize],
        target: usize,
    ) {
        for &j in stack.iter().rev() {
            if self.is_visible(j, target) {
                let vj = self.series.values[j].unwrap();
                let vi = self.series.values[target].unwrap();
                let w = (self.weight_fn)(j, target, vj, vi);
                edges.insert((j, target), w);
            } else if matches!(self.rule, VisibilityType::Horizontal) {
                break;
            }
        }
    }

    /// Merge results from all chunks
    fn merge_chunk_results(&self, chunk_results: Vec<HashMap<(usize, usize), f64>>) -> HashMap<(usize, usize), f64> {
        let mut edges = HashMap::new();
        for chunk_edges in chunk_results {
            edges.extend(chunk_edges);
        }
        edges
    }
}
