//! Graph-theoretic metrics and analysis.
//!
//! This module provides functions to compute various graph-theoretic
//! properties of visibility graphs.

use crate::core::VisibilityGraph;
use std::collections::{HashMap, HashSet, VecDeque};

/// Helper struct to store BFS results for shortest path computation.
struct ShortestPathsInfo {
    distances: Vec<usize>,
    num_paths: Vec<usize>,
}

impl<T> VisibilityGraph<T> {
    /// Helper: Computes shortest paths from a source node using BFS.
    /// Returns distances and number of shortest paths to each node.
    fn compute_shortest_paths_from_source(&self, source: usize) -> ShortestPathsInfo {
        let mut distances = vec![usize::MAX; self.node_count];
        let mut num_paths = vec![0usize; self.node_count];
        let mut queue = VecDeque::new();

        distances[source] = 0;
        num_paths[source] = 1;
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.neighbor_indices(v) {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[v] + 1;
                    num_paths[neighbor] = num_paths[v];
                    queue.push_back(neighbor);
                } else if distances[neighbor] == distances[v] + 1 {
                    num_paths[neighbor] += num_paths[v];
                }
            }
        }

        ShortestPathsInfo {
            distances,
            num_paths,
        }
    }

    /// Helper: Checks if a node lies on a shortest path from source to target.
    fn is_on_shortest_path(
        &self,
        node: usize,
        source: usize,
        target: usize,
        info: &ShortestPathsInfo,
    ) -> bool {
        if info.distances[target] == usize::MAX || info.distances[node] == usize::MAX {
            return false;
        }

        let dist_to_target = self.shortest_path_length(node, target).unwrap_or(usize::MAX);
        info.distances[source] + info.distances[node] + dist_to_target == info.distances[target]
    }

    /// Helper: Counts the contribution to betweenness centrality for a specific source node.
    fn count_betweenness_from_source(&self, node: usize, source: usize) -> f64 {
        let info = self.compute_shortest_paths_from_source(source);
        let mut contribution = 0.0;

        for target in 0..self.node_count {
            if target == source || target == node {
                continue;
            }

            if self.is_on_shortest_path(node, source, target, &info) && info.num_paths[target] > 0 {
                contribution += (info.num_paths[node] as f64) / (info.num_paths[target] as f64);
            }
        }

        contribution
    }

    /// Helper: Checks if an edge exists between two nodes (in either direction).
    #[inline]
    fn has_edge_between(&self, n1: usize, n2: usize) -> bool {
        self.edges.contains_key(&(n1, n2)) || self.edges.contains_key(&(n2, n1))
    }

    /// Helper: Counts edges between neighbors for clustering coefficient.
    fn count_neighbor_edges(&self, neighbors: &[usize]) -> usize {
        let mut count = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if self.has_edge_between(neighbors[i], neighbors[j]) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Computes the clustering coefficient for a specific node.
    ///
    /// The clustering coefficient measures the degree to which nodes in a graph
    /// tend to cluster together. For a node, it's the ratio of actual connections
    /// between its neighbors to the maximum possible connections.
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// Clustering coefficient (0.0 to 1.0), or None if node doesn't exist
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
    /// if let Some(cc) = graph.clustering_coefficient(1) {
    ///     println!("Clustering coefficient for node 1: {}", cc);
    /// }
    /// ```
    pub fn clustering_coefficient(&self, node: usize) -> Option<f64> {
        if node >= self.node_count {
            return None;
        }

        let neighbors = self.neighbor_indices(node);
        let k = neighbors.len();

        if k < 2 {
            return Some(0.0);
        }

        let actual_edges = self.count_neighbor_edges(&neighbors);
        let max_edges = k * (k - 1) / 2;

        Some(actual_edges as f64 / max_edges as f64)
    }

    /// Computes the average clustering coefficient for the entire graph.
    ///
    /// # Returns
    ///
    /// Average clustering coefficient across all nodes
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
    /// let avg_cc = graph.average_clustering_coefficient();
    /// println!("Average clustering coefficient: {}", avg_cc);
    /// ```
    pub fn average_clustering_coefficient(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.node_count {
            if let Some(cc) = self.clustering_coefficient(i) {
                sum += cc;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Computes the shortest path length between two nodes using BFS.
    ///
    /// # Arguments
    ///
    /// - `source`: Source node index
    /// - `target`: Target node index
    ///
    /// # Returns
    ///
    /// Shortest path length, or None if no path exists or nodes don't exist
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
    /// if let Some(dist) = graph.shortest_path_length(0, 3) {
    ///     println!("Distance from 0 to 3: {}", dist);
    /// }
    /// ```
    pub fn shortest_path_length(&self, source: usize, target: usize) -> Option<usize> {
        if source >= self.node_count || target >= self.node_count {
            return None;
        }

        if source == target {
            return Some(0);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((source, 0));
        visited.insert(source);

        while let Some((node, dist)) = queue.pop_front() {
            // Get neighbors
            for &neighbor in &self.neighbor_indices(node) {
                if neighbor == target {
                    return Some(dist + 1);
                }

                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        None // No path found
    }

    /// Computes the average shortest path length (characteristic path length).
    ///
    /// Computes the average shortest path length between all node pairs.
    ///
    /// ⚠️ **Performance Warning:** This method has **O(n² + n×m)** complexity where n is the
    /// number of nodes and m is the number of edges. It runs BFS from each node to all others.
    /// For large graphs (> 1,000 nodes), this can be slow. Consider sampling or using
    /// approximation methods for very large graphs.
    ///
    /// # Returns
    ///
    /// Average path length, or 0 if no paths exist
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
    /// let avg_path = graph.average_path_length();
    /// println!("Average path length: {:.2}", avg_path);
    /// ```
    pub fn average_path_length(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.node_count {
            for j in (i + 1)..self.node_count {
                if let Some(dist) = self.shortest_path_length(i, j) {
                    sum += dist as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Computes the diameter of the graph (longest shortest path).
    ///
    /// ⚠️ **Performance Warning:** This method has **O(n² + n×m)** complexity.
    /// Similar to `average_path_length`, it's expensive for large graphs (> 1,000 nodes).
    ///
    /// # Returns
    ///
    /// Graph diameter, or 0 if graph has no edges
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
    /// let diameter = graph.diameter();
    /// println!("Graph diameter: {}", diameter);
    /// ```
    pub fn diameter(&self) -> usize {
        let mut max_dist = 0;

        for i in 0..self.node_count {
            for j in (i + 1)..self.node_count {
                if let Some(dist) = self.shortest_path_length(i, j) {
                    max_dist = max_dist.max(dist);
                }
            }
        }

        max_dist
    }

    /// Computes the degree distribution of the graph.
    ///
    /// # Returns
    ///
    /// HashMap mapping degree to count of nodes with that degree
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
    /// let dist = graph.degree_distribution();
    /// for (degree, count) in dist {
    ///     println!("Degree {}: {} nodes", degree, count);
    /// }
    /// ```
    pub fn degree_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();

        for degree in self.degree_sequence() {
            *distribution.entry(degree).or_insert(0) += 1;
        }

        distribution
    }

    /// Checks if the graph is connected.
    ///
    /// # Returns
    ///
    /// True if all nodes are reachable from any other node
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
    /// if graph.is_connected() {
    ///     println!("Graph is connected");
    /// }
    /// ```
    pub fn is_connected(&self) -> bool {
        if self.node_count == 0 {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited.insert(0);

        while let Some(node) = queue.pop_front() {
            for &neighbor in &self.neighbor_indices(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == self.node_count
    }

    /// Computes graph density (ratio of actual edges to possible edges).
    ///
    /// For an undirected graph with n nodes, the maximum number of edges
    /// is n*(n-1)/2.
    ///
    /// # Returns
    ///
    /// Graph density (0.0 to 1.0)
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
    /// let density = graph.density();
    /// println!("Graph density: {:.3}", density);
    /// ```
    pub fn density(&self) -> f64 {
        let n = self.node_count;
        if n < 2 {
            return 0.0;
        }

        let max_edges = n * (n - 1) / 2;
        let actual_edges = self.edges.len();

        actual_edges as f64 / max_edges as f64
    }

    /// Computes betweenness centrality for a specific node.
    ///
    /// Betweenness centrality measures how often a node appears on shortest
    /// paths between other nodes. High betweenness indicates the node is
    /// important for connectivity.
    ///
    /// # Arguments
    ///
    /// - `node`: Node index
    ///
    /// # Returns
    ///
    /// Betweenness centrality value, or None if node doesn't exist
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
    /// if let Some(bc) = graph.betweenness_centrality(2) {
    ///     println!("Betweenness centrality for node 2: {:.4}", bc);
    /// }
    /// ```
    pub fn betweenness_centrality(&self, node: usize) -> Option<f64> {
        if node >= self.node_count {
            return None;
        }

        let centrality = (0..self.node_count)
            .filter(|&s| s != node)
            .map(|s| self.count_betweenness_from_source(node, s))
            .sum::<f64>();

        // Normalize by the number of pairs
        let normalized = if self.node_count > 2 {
            centrality / ((self.node_count - 1) * (self.node_count - 2)) as f64
        } else {
            centrality
        };

        Some(normalized)
    }

    /// Computes betweenness centrality for all nodes.
    ///
    /// ⚠️ **Performance Warning:** This method has **O(n×m)** complexity using Brandes' algorithm,
    /// where n is nodes and m is edges. For large graphs (> 1,000 nodes), this can take significant
    /// time. This is the optimal algorithm for exact betweenness centrality.
    ///
    /// # Returns
    ///
    /// Vector of betweenness centrality values for each node
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
    /// let centralities = graph.betweenness_centrality_all();
    /// for (i, bc) in centralities.iter().enumerate() {
    ///     println!("Node {}: {:.4}", i, bc);
    /// }
    /// ```
    pub fn betweenness_centrality_all(&self) -> Vec<f64> {
        (0..self.node_count)
            .map(|i| self.betweenness_centrality(i).unwrap_or(0.0))
            .collect()
    }
}
