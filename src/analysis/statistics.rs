//! Graph statistics and summary information.
//!
//! This module provides convenient methods to compute and display
//! comprehensive statistics about visibility graphs.

use crate::core::VisibilityGraph;
use std::fmt;

/// Comprehensive statistics about a visibility graph.
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Whether graph is directed
    pub is_directed: bool,
    /// Average degree
    pub average_degree: f64,
    /// Minimum degree
    pub min_degree: usize,
    /// Maximum degree
    pub max_degree: usize,
    /// Degree standard deviation
    pub degree_std_dev: f64,
    /// Degree variance
    pub degree_variance: f64,
    /// Average clustering coefficient
    pub average_clustering: f64,
    /// Global clustering coefficient
    pub global_clustering: f64,
    /// Average shortest path length
    pub average_path_length: f64,
    /// Graph diameter
    pub diameter: usize,
    /// Graph radius
    pub radius: usize,
    /// Graph density
    pub density: f64,
    /// Whether graph is connected
    pub is_connected: bool,
    /// Number of connected components
    pub num_components: usize,
    /// Size of largest component
    pub largest_component_size: usize,
    /// Assortativity coefficient (degree correlation)
    pub assortativity: f64,
    /// Number of features per node
    pub feature_count: usize,
}

impl fmt::Display for GraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔════════════════════════════════════════╗")?;
        writeln!(f, "║      Graph Statistics Summary          ║")?;
        writeln!(f, "╚════════════════════════════════════════╝")?;
        writeln!(f)?;
        writeln!(f, "📊 Structure:")?;
        writeln!(f, "  Nodes:         {}", self.node_count)?;
        writeln!(f, "  Edges:         {}", self.edge_count)?;
        writeln!(f, "  Directed:      {}", self.is_directed)?;
        writeln!(f, "  Connected:     {}", self.is_connected)?;
        writeln!(f, "  Components:    {}", self.num_components)?;
        writeln!(f, "  Largest comp:  {}", self.largest_component_size)?;
        writeln!(f)?;
        writeln!(f, "📈 Degree Distribution:")?;
        writeln!(f, "  Average:       {:.2}", self.average_degree)?;
        writeln!(f, "  Std Dev:       {:.2}", self.degree_std_dev)?;
        writeln!(f, "  Variance:      {:.2}", self.degree_variance)?;
        writeln!(f, "  Min:           {}", self.min_degree)?;
        writeln!(f, "  Max:           {}", self.max_degree)?;
        writeln!(f)?;
        writeln!(f, "🔗 Clustering:")?;
        writeln!(f, "  Average:       {:.4}", self.average_clustering)?;
        writeln!(f, "  Global:        {:.4}", self.global_clustering)?;
        writeln!(f)?;
        writeln!(f, "📏 Distance Metrics:")?;
        writeln!(f, "  Avg Path Len:  {:.2}", self.average_path_length)?;
        writeln!(f, "  Diameter:      {}", self.diameter)?;
        writeln!(f, "  Radius:        {}", self.radius)?;
        writeln!(f)?;
        writeln!(f, "🎯 Network Properties:")?;
        writeln!(f, "  Density:       {:.4}", self.density)?;
        writeln!(f, "  Assortativity: {:.4}", self.assortativity)?;
        writeln!(f)?;
        if self.feature_count > 0 {
            writeln!(f, "✨ Features:")?;
            writeln!(f, "  Per node:      {}", self.feature_count)?;
        }
        Ok(())
    }
}

impl<T> VisibilityGraph<T> {
    /// Computes comprehensive statistics about the graph.
    ///
    /// This is a convenience method that computes all available metrics
    /// in one call for easy analysis and reporting.
    ///
    /// # Returns
    ///
    /// A `GraphStatistics` struct with all computed metrics
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
    ///
    /// let stats = graph.compute_statistics();
    /// println!("{}", stats);
    /// ```
    pub fn compute_statistics(&self) -> GraphStatistics {
        let degrees = self.degree_sequence();
        let avg_degree = if !degrees.is_empty() {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        } else {
            0.0
        };

        let min_degree = degrees.iter().min().copied().unwrap_or(0);
        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        let feature_count = if !self.node_features.is_empty() {
            self.node_features[0].len()
        } else {
            0
        };

        GraphStatistics {
            node_count: self.node_count,
            edge_count: self.edges.len(),
            is_directed: self.directed,
            average_degree: avg_degree,
            min_degree,
            max_degree,
            degree_std_dev: self.degree_std_dev(),
            degree_variance: self.degree_variance(),
            average_clustering: self.average_clustering_coefficient(),
            global_clustering: self.global_clustering_coefficient(),
            average_path_length: self.average_path_length(),
            diameter: self.diameter(),
            radius: self.radius(),
            density: self.density(),
            is_connected: self.is_connected(),
            num_components: self.count_components(),
            largest_component_size: self.largest_component_size(),
            assortativity: self.assortativity(),
            feature_count,
        }
    }

    /// Prints a summary of the graph to stdout.
    ///
    /// This is a convenience method for quick inspection during development.
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
    /// graph.print_summary();
    /// ```
    pub fn print_summary(&self) {
        println!("{}", self.compute_statistics());
    }

    /// Helper: Computes shortest paths from a single source using BFS.
    fn shortest_paths_from(&self, source: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.node_count];
        distances[source] = 0;

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            for &neighbor in &self.neighbor_indices(node) {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }
}

// ============================================================================
// Advanced Statistical Methods
// ============================================================================
impl<T> VisibilityGraph<T> {
    /// Computes the standard deviation of the degree distribution.
    pub fn degree_std_dev(&self) -> f64 {
        self.degree_variance().sqrt()
    }
    /// Computes the variance of the degree distribution.
    pub fn degree_variance(&self) -> f64 {
        let degrees = self.degree_sequence();
        if degrees.is_empty() {
            return 0.0;
        }
        let mean = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        let variance = degrees.iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / degrees.len() as f64;
        variance
    }
    /// Computes the global clustering coefficient (transitivity).
    /// 
    /// This is the ratio of triangles to connected triples in the graph.
    pub fn global_clustering_coefficient(&self) -> f64 {
        let mut triangles = 0;
        let mut triples = 0;
        for i in 0..self.node_count {
            let neighbors: Vec<usize> = self.neighbor_indices(i);
            let degree = neighbors.len();
            if degree < 2 {
                continue;
            }
            // Count triples centered at i
            triples += degree * (degree - 1) / 2;
            // Count triangles
            for j in 0..neighbors.len() {
                for k in (j + 1)..neighbors.len() {
                    if self.has_edge(neighbors[j], neighbors[k]) {
                        triangles += 1;
                    }
                }
            }
        }
        if triples == 0 {
            0.0
        } else {
            triangles as f64 / triples as f64
        }
    }
    /// Computes the radius of the graph (minimum eccentricity).
    pub fn radius(&self) -> usize {
        if self.node_count == 0 {
            return 0;
        }
        let mut min_eccentricity = usize::MAX;
        for i in 0..self.node_count {
            let distances = self.shortest_paths_from(i);
            let eccentricity = distances.iter()
                .filter(|&&d| d < usize::MAX)
                .max()
                .copied()
                .unwrap_or(0);
            if eccentricity < min_eccentricity {
                min_eccentricity = eccentricity;
            }
        }
        min_eccentricity
    }
    /// Counts the number of connected components in the graph.
    pub fn count_components(&self) -> usize {
        let mut visited = vec![false; self.node_count];
        let mut num_components = 0;
        for start in 0..self.node_count {
            if !visited[start] {
                num_components += 1;
                self.dfs_mark_component(start, &mut visited);
            }
        }
        num_components
    }
    /// Returns the size of the largest connected component.
    pub fn largest_component_size(&self) -> usize {
        let mut visited = vec![false; self.node_count];
        let mut max_size = 0;
        for start in 0..self.node_count {
            if !visited[start] {
                let size = self.dfs_count_component(start, &mut visited);
                if size > max_size {
                    max_size = size;
                }
            }
        }
        max_size
    }
    /// Helper: DFS to mark all nodes in a component.
    fn dfs_mark_component(&self, node: usize, visited: &mut [bool]) {
        visited[node] = true;
        for &neighbor in &self.neighbor_indices(node) {
            if !visited[neighbor] {
                self.dfs_mark_component(neighbor, visited);
            }
        }
    }
    /// Helper: DFS to count nodes in a component.
    fn dfs_count_component(&self, node: usize, visited: &mut [bool]) -> usize {
        visited[node] = true;
        let mut count = 1;
        for &neighbor in &self.neighbor_indices(node) {
            if !visited[neighbor] {
                count += self.dfs_count_component(neighbor, visited);
            }
        }
        count
    }
    /// Computes the assortativity coefficient (degree correlation).
    /// 
    /// Measures the tendency of nodes to connect to others with similar degree.
    /// Positive values indicate assortative mixing, negative values indicate disassortative mixing.
    pub fn assortativity(&self) -> f64 {
        if self.edges.is_empty() {
            return 0.0;
        }
        let degrees = self.degree_sequence();
        let m = self.edges.len() as f64;
        let mut sum_jk = 0.0;
        let mut sum_j_plus_k = 0.0;
        let mut sum_j_sq_plus_k_sq = 0.0;
        for (i, j) in self.edges.keys() {
            let deg_i = degrees[*i] as f64;
            let deg_j = degrees[*j] as f64;
            sum_jk += deg_i * deg_j;
            sum_j_plus_k += deg_i + deg_j;
            sum_j_sq_plus_k_sq += deg_i * deg_i + deg_j * deg_j;
        }
        let numerator = sum_jk / m - (sum_j_plus_k / (2.0 * m)).powi(2);
        let denominator = sum_j_sq_plus_k_sq / (2.0 * m) - (sum_j_plus_k / (2.0 * m)).powi(2);
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }
    /// Returns degree distribution as a histogram.
    /// 
    /// Returns a vector where index i contains the count of nodes with degree i.
    pub fn degree_distribution_histogram(&self) -> Vec<usize> {
        let degrees = self.degree_sequence();
        if degrees.is_empty() {
            return vec![];
        }
        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let mut distribution = vec![0; max_degree + 1];
        for &degree in &degrees {
            distribution[degree] += 1;
        }
        distribution
    }
    /// Computes entropy of the degree distribution.
    /// 
    /// Measures the uncertainty/diversity in the degree distribution.
    pub fn degree_entropy(&self) -> f64 {
        let distribution = self.degree_distribution_histogram();
        let total = self.node_count as f64;
        if total == 0.0 {
            return 0.0;
        }
        distribution.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum()
    }
    /// Returns the degree centrality for all nodes.
    /// 
    /// Degree centrality is normalized by (n-1) where n is the number of nodes.
    pub fn degree_centrality(&self) -> Vec<f64> {
        let degrees = self.degree_sequence();
        let n = self.node_count;
        if n <= 1 {
            return vec![0.0; n];
        }
        let normalizer = (n - 1) as f64;
        degrees.iter()
            .map(|&d| d as f64 / normalizer)
            .collect()
    }
    /// Computes betweenness centrality for all nodes (batch version).
    ///
    /// Measures the extent to which a node lies on paths between other nodes.
    /// Note: This computes for all nodes at once. Use the metrics module for single nodes.
    pub fn betweenness_centrality_batch(&self) -> Vec<f64> {
        let n = self.node_count;
        let mut betweenness = vec![0.0; n];

        for source in 0..n {
            self.compute_betweenness_from_source(source, &mut betweenness);
        }

        self.normalize_betweenness(betweenness, n)
    }

    /// Compute betweenness centrality contribution from a single source node
    fn compute_betweenness_from_source(&self, source: usize, betweenness: &mut [f64]) {
        let n = self.node_count;
        let mut stack = Vec::new();
        let mut paths = vec![Vec::new(); n];
        let mut sigma = vec![0.0; n];
        let mut dist = vec![usize::MAX; n];

        sigma[source] = 1.0;
        dist[source] = 0;

        self.bfs_from_source(source, &mut stack, &mut paths, &mut sigma, &mut dist);
        self.accumulate_betweenness(&stack, &paths, &sigma, betweenness);
    }

    /// Perform BFS from source to compute paths and distances
    fn bfs_from_source(
        &self,
        source: usize,
        stack: &mut Vec<usize>,
        paths: &mut [Vec<usize>],
        sigma: &mut [f64],
        dist: &mut [usize],
    ) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &self.neighbor_indices(v) {
                if dist[w] == usize::MAX {
                    self.process_unvisited_neighbor(v, w, dist, &mut queue);
                }
                if dist[w] == dist[v] + 1 {
                    self.process_neighbor_on_shortest_path(v, w, sigma, paths);
                }
            }
        }
    }

    /// Process an unvisited neighbor during BFS
    fn process_unvisited_neighbor(
        &self,
        v: usize,
        w: usize,
        dist: &mut [usize],
        queue: &mut std::collections::VecDeque<usize>,
    ) {
        dist[w] = dist[v] + 1;
        queue.push_back(w);
    }

    /// Process a neighbor that lies on a shortest path
    fn process_neighbor_on_shortest_path(
        &self,
        v: usize,
        w: usize,
        sigma: &mut [f64],
        paths: &mut [Vec<usize>],
    ) {
        sigma[w] += sigma[v];
        paths[w].push(v);
    }

    /// Accumulate betweenness centrality using dependency accumulation
    fn accumulate_betweenness(
        &self,
        stack: &[usize],
        paths: &[Vec<usize>],
        sigma: &[f64],
        betweenness: &mut [f64],
    ) {
        let n = self.node_count;
        let mut delta = vec![0.0; n];
        let mut temp_stack = stack.to_vec();

        while let Some(w) = temp_stack.pop() {
            for &v in &paths[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            // Don't add contribution for source node
            if !temp_stack.is_empty() || w != stack[0] {
                betweenness[w] += delta[w];
            }
        }
    }

    /// Normalize betweenness centrality values
    fn normalize_betweenness(&self, mut betweenness: Vec<f64>, n: usize) -> Vec<f64> {
        let normalizer = if n > 2 {
            ((n - 1) * (n - 2)) as f64
        } else {
            1.0
        };

        for b in &mut betweenness {
            *b /= normalizer;
        }
        betweenness
    }

    /// Computes closeness centrality for all nodes.
    ///
    /// Measures the average distance from a node to all other nodes.
    pub fn closeness_centrality(&self) -> Vec<f64> {
        let n = self.node_count;
        let mut closeness = vec![0.0; n];
        for (i, closeness_val) in closeness.iter_mut().enumerate().take(n) {
            let distances = self.shortest_paths_from(i);
            let reachable: Vec<usize> = distances.iter()
                .filter(|&&d| d > 0 && d < usize::MAX)
                .copied()
                .collect();
            if !reachable.is_empty() {
                let sum_dist: usize = reachable.iter().sum();
                let num_reachable = reachable.len() as f64;
                *closeness_val = num_reachable / sum_dist as f64;
            }
        }
        closeness
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;
    #[test]
    fn test_degree_variance() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();
        let variance = graph.degree_variance();
        assert!(variance >= 0.0);
    }
    #[test]
    fn test_assortativity() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();
        let assort = graph.assortativity();
        assert!(assort >= -1.0 && assort <= 1.0);
    }
    #[test]
    fn test_centrality_measures() {
        let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let graph = VisibilityGraph::from_series(&series)
            .natural_visibility()
            .unwrap();
        let degree_cent = graph.degree_centrality();
        let between_cent = graph.betweenness_centrality_batch();
        let close_cent = graph.closeness_centrality();
        assert_eq!(degree_cent.len(), 4);
        assert_eq!(between_cent.len(), 4);
        assert_eq!(close_cent.len(), 4);
    }
}
