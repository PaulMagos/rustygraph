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
        // Adjacent nodes (no intermediate points) always see each other
        if j + 1 >= i {
            return false;
        }

        let vk: f64 = self.series.values[k].unwrap().into();
        let vj: f64 = self.series.values[j].unwrap().into();
        let vi: f64 = self.series.values[i].unwrap().into();

        // Calculate the expected height of the line from k to i at position j
        let t_k = k as f64;
        let t_j = j as f64;
        let t_i = i as f64;

        let expected_height = vk + (vi - vk) * ((t_j - t_k) / (t_i - t_k));

        // Only remove j if it's STRICTLY below the line k→i
        // AND the slopes confirm j is in a permanently shadowed valley
        if vj < expected_height {
            // Check if j is in a monotonically shadowed position
            let slope_kj = (vj - vk) / (t_j - t_k);
            let slope_ki = (vi - vk) / (t_i - t_k);

            // Remove j only if the slope to j is less than slope to i
            // This means j is getting increasingly shadowed
            slope_kj < slope_ki
        } else {
            // j is on or above the line, definitely keep it
            false
        }
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
        use rayon::prelude::*;

        let n = self.series.len();
        if n <= 100 {
            // For small graphs, sequential is faster (avoid parallelization overhead)
            return self.compute_edges();
        }

        // Parallel strategy: Process target nodes in parallel
        // For each target node i, check visibility from ALL previous nodes 0..i
        // This is still O(n²) worst case but parallelizes well

        let chunk_results: Vec<HashMap<(usize, usize), f64>> = (0..n)
            .collect::<Vec<_>>()
            .par_chunks(64)  // Process 64 nodes at a time
            .map(|target_chunk| {
                let mut local_edges = HashMap::new();

                for &i in target_chunk {
                    // For each target node i, check visibility from all previous nodes
                    // Use envelope optimization for this single target
                    let mut stack: Vec<usize> = Vec::new();

                    for j in 0..i {
                        // Update envelope for natural visibility
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

                        stack.push(j);
                    }

                    // Now check visibility from nodes in stack to current node i
                    for &j in stack.iter().rev() {
                        if self.is_visible(j, i) {
                            let vj = self.series.values[j].unwrap();
                            let vi = self.series.values[i].unwrap();
                            let w = (self.weight_fn)(j, i, vj, vi);
                            local_edges.insert((j, i), w);
                        } else if matches!(self.rule, VisibilityType::Horizontal) {
                            break;
                        }
                    }
                }

                local_edges
            })
            .collect();

        // Merge all results
        let mut edges = HashMap::new();
        for chunk_edges in chunk_results {
            edges.extend(chunk_edges);
        }

        edges
    }
}
