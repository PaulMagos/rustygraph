//! Community detection algorithms for visibility graphs.
//!
//! This module provides algorithms for detecting communities (densely connected
//! subgraphs) in visibility graphs.

use crate::core::VisibilityGraph;
use std::collections::{HashMap, HashSet, VecDeque};

/// Community detection result.
#[derive(Debug, Clone)]
pub struct Communities {
    /// Node to community ID mapping
    pub node_communities: Vec<usize>,
    /// Number of communities found
    pub num_communities: usize,
    /// Modularity score (quality metric)
    pub modularity: f64,
}

impl<T> VisibilityGraph<T> {
    /// Detects communities using the Louvain algorithm (simplified).
    ///
    /// This is a greedy modularity optimization algorithm that finds
    /// densely connected groups of nodes.
    ///
    /// # Returns
    ///
    /// `Communities` struct with node assignments and modularity score
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![
    ///     1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0
    /// ]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let communities = graph.detect_communities();
    /// println!("Found {} communities", communities.num_communities);
    /// println!("Modularity: {:.4}", communities.modularity);
    /// ```
    pub fn detect_communities(&self) -> Communities {
        let mut node_communities = self.initialize_communities();
        let mut improved = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        // Greedy optimization loop
        while improved && iteration < MAX_ITERATIONS {
            improved = self.optimize_communities_one_pass(&mut node_communities);
            iteration += 1;
        }

        // Renumber communities to be contiguous and compute final result
        self.finalize_communities(node_communities)
    }

    /// Initialize each node in its own community
    fn initialize_communities(&self) -> Vec<usize> {
        (0..self.node_count).collect()
    }

    /// Perform one pass of community optimization
    fn optimize_communities_one_pass(&self, node_communities: &mut [usize]) -> bool {
        let mut improved = false;

        for node in 0..self.node_count {
            if let Some(best_community) = self.find_best_community_for_node(node, node_communities) {
                node_communities[node] = best_community;
                improved = true;
            }
        }

        improved
    }

    /// Find the best community for a node by trying all neighbor communities
    fn find_best_community_for_node(&self, node: usize, communities: &[usize]) -> Option<usize> {
        let current_community = communities[node];
        let neighbors = self.get_neighbors(node);
        let candidate_communities = self.get_candidate_communities(&neighbors, communities);

        let mut best_community = current_community;
        let mut best_gain = 0.0;

        for &candidate in &candidate_communities {
            if candidate == current_community {
                continue;
            }

            let gain = self.modularity_gain(node, current_community, candidate, communities);
            if gain > best_gain {
                best_gain = gain;
                best_community = candidate;
            }
        }

        if best_community != current_community {
            Some(best_community)
        } else {
            None
        }
    }

    /// Get unique communities from neighbors
    fn get_candidate_communities(&self, neighbors: &[usize], communities: &[usize]) -> HashSet<usize> {
        neighbors.iter()
            .map(|&neighbor| communities[neighbor])
            .collect()
    }

    /// Renumber communities to be contiguous and compute final result
    fn finalize_communities(&self, mut node_communities: Vec<usize>) -> Communities {
        let unique_communities: HashSet<_> = node_communities.iter().copied().collect();
        let community_map = self.create_community_renumbering_map(&unique_communities);

        // Apply renumbering
        for comm in &mut node_communities {
            *comm = community_map[comm];
        }

        let num_communities = unique_communities.len();
        let modularity = self.compute_modularity(&node_communities, num_communities);

        Communities {
            node_communities,
            num_communities,
            modularity,
        }
    }

    /// Create mapping from old community IDs to new contiguous IDs
    fn create_community_renumbering_map(&self, unique_communities: &HashSet<usize>) -> HashMap<usize, usize> {
        let mut community_map = HashMap::new();
        for (new_id, &old_id) in unique_communities.iter().enumerate() {
            community_map.insert(old_id, new_id);
        }
        community_map
    }

    /// Helper: Get neighbors of a node
    fn get_neighbors(&self, node: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for &(src, dst) in self.edges.keys() {
            if src == node {
                neighbors.push(dst);
            } else if !self.directed && dst == node {
                neighbors.push(src);
            }
        }
        neighbors
    }

    /// Helper: Calculate modularity gain for moving a node
    fn modularity_gain(
        &self,
        node: usize,
        from_comm: usize,
        to_comm: usize,
        communities: &[usize],
    ) -> f64 {
        let m = self.edges.len() as f64;
        if m == 0.0 {
            return 0.0;
        }

        let neighbors = self.get_neighbors(node);
        let degree = neighbors.len() as f64;

        // Count edges to nodes in target community
        let edges_to = neighbors.iter()
            .filter(|&&n| communities[n] == to_comm)
            .count() as f64;

        // Count edges to nodes in source community
        let edges_from = neighbors.iter()
            .filter(|&&n| communities[n] == from_comm)
            .count() as f64;

        // Simplified modularity gain calculation
        (edges_to - edges_from) / (2.0 * m) - degree * degree / (4.0 * m * m)
    }

    /// Helper: Compute modularity score
    fn compute_modularity(&self, communities: &[usize], num_comms: usize) -> f64 {
        let m = self.edges.len() as f64;
        if m == 0.0 {
            return 0.0;
        }

        (0..num_comms)
            .map(|c| self.compute_modularity_for_community(c, communities, m))
            .sum()
    }

    /// Compute modularity contribution for a single community
    fn compute_modularity_for_community(&self, community_id: usize, communities: &[usize], total_edges: f64) -> f64 {
        let nodes_in_comm = self.get_nodes_in_community(community_id, communities);
        if nodes_in_comm.is_empty() {
            return 0.0;
        }

        let internal_edges = self.count_internal_edges(&nodes_in_comm);
        let degree_sum = self.sum_degrees(&nodes_in_comm);

        internal_edges / (2.0 * total_edges) - (degree_sum / (2.0 * total_edges)).powi(2)
    }

    /// Get all nodes belonging to a specific community
    fn get_nodes_in_community(&self, community_id: usize, communities: &[usize]) -> Vec<usize> {
        communities.iter()
            .enumerate()
            .filter(|(_, &comm)| comm == community_id)
            .map(|(node, _)| node)
            .collect()
    }

    /// Count edges within a community
    fn count_internal_edges(&self, nodes: &[usize]) -> f64 {
        let mut count = 0.0;
        for &node1 in nodes {
            for &node2 in nodes {
                if self.edges.contains_key(&(node1, node2)) {
                    count += 1.0;
                }
            }
        }
        count
    }

    /// Sum degrees of nodes in a community
    fn sum_degrees(&self, nodes: &[usize]) -> f64 {
        nodes.iter()
            .map(|&node| self.get_neighbors(node).len() as f64)
            .sum()
    }

    /// Finds connected components in the graph.
    ///
    /// Each connected component is a maximal set of nodes where each node
    /// can reach every other node via some path.
    ///
    /// # Returns
    ///
    /// Vector of component IDs for each node
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
    /// let components = graph.connected_components();
    /// println!("Node component assignments: {:?}", components);
    /// ```
    pub fn connected_components(&self) -> Vec<usize> {
        let mut components = vec![usize::MAX; self.node_count];
        let mut component_id = 0;

        for start_node in 0..self.node_count {
            if self.is_node_visited(start_node, &components) {
                continue;
            }

            self.explore_component(start_node, component_id, &mut components);
            component_id += 1;
        }

        components
    }

    /// Check if a node has been visited (assigned to a component)
    fn is_node_visited(&self, node: usize, components: &[usize]) -> bool {
        components[node] != usize::MAX
    }

    /// Explore a connected component using BFS
    fn explore_component(&self, start_node: usize, component_id: usize, components: &mut [usize]) {
        let mut queue = VecDeque::new();
        queue.push_back(start_node);
        components[start_node] = component_id;

        while let Some(current_node) = queue.pop_front() {
            for neighbor in self.get_neighbors(current_node) {
                if !self.is_node_visited(neighbor, components) {
                    components[neighbor] = component_id;
                    queue.push_back(neighbor);
                }
            }
        }
    }
}

impl Communities {
    /// Returns nodes in a specific community.
    pub fn get_community_nodes(&self, community_id: usize) -> Vec<usize> {
        self.node_communities
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == community_id)
            .map(|(n, _)| n)
            .collect()
    }

    /// Returns the size of each community.
    pub fn community_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.num_communities];
        for &comm in &self.node_communities {
            sizes[comm] += 1;
        }
        sizes
    }
}
