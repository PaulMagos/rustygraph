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
        // Start with each node in its own community
        let mut node_communities: Vec<usize> = (0..self.node_count).collect();
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = 100;

        // Greedy optimization
        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for node in 0..self.node_count {
                let current_community = node_communities[node];
                let mut best_community = current_community;
                let mut best_gain = 0.0;

                // Try moving node to each neighbor's community
                let neighbors = self.get_neighbors(node);
                let mut candidate_communities = HashSet::new();

                for &neighbor in &neighbors {
                    candidate_communities.insert(node_communities[neighbor]);
                }

                for &candidate in &candidate_communities {
                    if candidate == current_community {
                        continue;
                    }

                    // Calculate modularity gain
                    let gain = self.modularity_gain(
                        node,
                        current_community,
                        candidate,
                        &node_communities
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best_community = candidate;
                    }
                }

                if best_community != current_community {
                    node_communities[node] = best_community;
                    improved = true;
                }
            }
        }

        // Renumber communities to be contiguous
        let unique_communities: HashSet<_> = node_communities.iter().copied().collect();
        let mut community_map: HashMap<usize, usize> = HashMap::new();
        for (new_id, &old_id) in unique_communities.iter().enumerate() {
            community_map.insert(old_id, new_id);
        }

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

        let mut modularity = 0.0;

        for c in 0..num_comms {
            let nodes_in_comm: Vec<usize> = communities.iter()
                .enumerate()
                .filter(|(_, &comm)| comm == c)
                .map(|(node, _)| node)
                .collect();

            if nodes_in_comm.is_empty() {
                continue;
            }

            // Count internal edges
            let mut internal_edges = 0.0;
            for &node1 in &nodes_in_comm {
                for &node2 in &nodes_in_comm {
                    if self.edges.contains_key(&(node1, node2)) {
                        internal_edges += 1.0;
                    }
                }
            }

            // Sum of degrees
            let degree_sum: f64 = nodes_in_comm.iter()
                .map(|&n| self.get_neighbors(n).len() as f64)
                .sum();

            modularity += internal_edges / (2.0 * m) - (degree_sum / (2.0 * m)).powi(2);
        }

        modularity
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

        for start in 0..self.node_count {
            if components[start] != usize::MAX {
                continue; // Already visited
            }

            // BFS to find all nodes in this component
            let mut queue = VecDeque::new();
            queue.push_back(start);
            components[start] = component_id;

            while let Some(node) = queue.pop_front() {
                for neighbor in self.get_neighbors(node) {
                    if components[neighbor] == usize::MAX {
                        components[neighbor] = component_id;
                        queue.push_back(neighbor);
                    }
                }
            }

            component_id += 1;
        }

        components
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

