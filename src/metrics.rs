//! Graph-theoretic metrics and analysis.
//!
//! This module provides functions to compute various graph-theoretic
//! properties of visibility graphs.

use crate::VisibilityGraph;
use std::collections::{HashMap, HashSet, VecDeque};

impl<T> VisibilityGraph<T> {
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

        // Get neighbors of the node
        let neighbors: Vec<usize> = self.edges
            .keys()
            .filter_map(|&(src, dst)| {
                if src == node {
                    Some(dst)
                } else if dst == node {
                    Some(src)
                } else {
                    None
                }
            })
            .collect();

        let k = neighbors.len();
        if k < 2 {
            return Some(0.0);
        }

        // Count edges between neighbors
        let mut actual_edges = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let n1 = neighbors[i];
                let n2 = neighbors[j];
                if self.edges.contains_key(&(n1, n2)) || self.edges.contains_key(&(n2, n1)) {
                    actual_edges += 1;
                }
            }
        }

        // Maximum possible edges between k neighbors is k*(k-1)/2
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
            for &(src, dst) in self.edges.keys() {
                let neighbor = if src == node {
                    dst
                } else if dst == node {
                    src
                } else {
                    continue;
                };

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
    /// This is the average of all shortest paths between all pairs of nodes.
    ///
    /// # Returns
    ///
    /// Average shortest path length
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
    /// println!("Average path length: {}", avg_path);
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
            for &(src, dst) in self.edges.keys() {
                let neighbor = if src == node {
                    dst
                } else if dst == node {
                    src
                } else {
                    continue;
                };

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
        
        let mut centrality = 0.0;
        
        // For each pair of nodes (s, t)
        for s in 0..self.node_count {
            if s == node {
                continue;
            }
            
            // BFS from s to find all shortest paths
            let mut distances = vec![usize::MAX; self.node_count];
            let mut num_paths = vec![0usize; self.node_count];
            let mut queue = VecDeque::new();
            
            distances[s] = 0;
            num_paths[s] = 1;
            queue.push_back(s);
            
            while let Some(v) = queue.pop_front() {
                // Get neighbors of v
                for &(src, dst) in self.edges.keys() {
                    let neighbor = if src == v {
                        dst
                    } else if dst == v {
                        src
                    } else {
                        continue;
                    };
                    
                    if distances[neighbor] == usize::MAX {
                        distances[neighbor] = distances[v] + 1;
                        num_paths[neighbor] = num_paths[v];
                        queue.push_back(neighbor);
                    } else if distances[neighbor] == distances[v] + 1 {
                        num_paths[neighbor] += num_paths[v];
                    }
                }
            }
            
            // For each target t, check if node is on shortest path from s to t
            for t in 0..self.node_count {
                if t == s || t == node {
                    continue;
                }
                
                if distances[t] != usize::MAX && distances[node] != usize::MAX {
                    // Check if node is on a shortest path from s to t
                    if distances[s] + distances[node] + 
                       self.shortest_path_length(node, t).unwrap_or(usize::MAX) == distances[t] {
                        // Count paths through node
                        if num_paths[t] > 0 {
                            centrality += (num_paths[node] as f64) / (num_paths[t] as f64);
                        }
                    }
                }
            }
        }
        
        // Normalize by the number of pairs
        let n = self.node_count;
        if n > 2 {
            centrality /= ((n - 1) * (n - 2)) as f64;
        }
        
        Some(centrality)
    }
    
    /// Computes betweenness centrality for all nodes.
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

