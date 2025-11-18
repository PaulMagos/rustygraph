//! Graph and feature export functionality.
//!
//! This module provides functions to export visibility graphs and features
//! to various formats for analysis and visualization.

use crate::VisibilityGraph;
use std::collections::HashSet;
use std::fmt::Write;

/// Export format for graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format with nodes and edges
    Json,
    /// CSV edge list format
    EdgeList,
    /// Adjacency matrix in CSV format
    AdjacencyMatrix,
}

/// Options for graph export.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include edge weights
    pub include_weights: bool,
    /// Include node features
    pub include_features: bool,
    /// Pretty print JSON
    pub pretty: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_weights: true,
            include_features: true,
            pretty: true,
        }
    }
}

impl<T> VisibilityGraph<T>
where
    T: std::fmt::Display + Copy,
{
    /// Exports the graph to JSON format.
    ///
    /// # Arguments
    ///
    /// - `options`: Export options
    ///
    /// # Returns
    ///
    /// JSON string representation of the graph
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    /// use rustygraph::export::ExportOptions;
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let json = graph.to_json(ExportOptions::default());
    /// println!("{}", json);
    /// ```
    pub fn to_json(&self, options: ExportOptions) -> String {
        let mut output = String::new();
        let indent = if options.pretty { "  " } else { "" };
        let newline = if options.pretty { "\n" } else { "" };

        writeln!(output, "{{{}", newline).unwrap();
        writeln!(output, "{}\"nodes\": {},", indent, self.node_count).unwrap();
        writeln!(output, "{}\"edges\": [", indent).unwrap();

        let edges: Vec<_> = self.edges.iter().collect();
        for (i, (&(src, dst), &weight)) in edges.iter().enumerate() {
            let comma = if i < edges.len() - 1 { "," } else { "" };
            if options.include_weights {
                writeln!(
                    output,
                    "{}{}{{\"source\": {}, \"target\": {}, \"weight\": {}}}{}",
                    indent, indent, src, dst, weight, comma
                ).unwrap();
            } else {
                writeln!(
                    output,
                    "{}{}{{\"source\": {}, \"target\": {}}}{}",
                    indent, indent, src, dst, comma
                ).unwrap();
            }
        }

        writeln!(output, "{}]", indent).unwrap();

        if options.include_features && !self.node_features.is_empty() {
            writeln!(output, "{},\"features\": [", indent).unwrap();
            for (i, features) in self.node_features.iter().enumerate() {
                let comma = if i < self.node_features.len() - 1 { "," } else { "" };
                write!(output, "{}{}{{\"node\": {}, ", indent, indent, i).unwrap();

                let feature_items: Vec<_> = features.iter().collect();
                write!(output, "\"values\": {{").unwrap();
                for (j, (name, value)) in feature_items.iter().enumerate() {
                    let fcomma = if j < feature_items.len() - 1 { ", " } else { "" };
                    write!(output, "\"{}\": {}{}", name, value, fcomma).unwrap();
                }
                writeln!(output, "}}}}{}",  comma).unwrap();
            }
            writeln!(output, "{}]", indent).unwrap();
        }

        writeln!(output, "}}").unwrap();
        output
    }

    /// Exports the graph as a CSV edge list.
    ///
    /// # Arguments
    ///
    /// - `include_weights`: Whether to include edge weights
    ///
    /// # Returns
    ///
    /// CSV string with format: source,target[,weight]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let csv = graph.to_edge_list_csv(true);
    /// println!("{}", csv);
    /// ```
    pub fn to_edge_list_csv(&self, include_weights: bool) -> String {
        let mut output = String::new();

        // Header
        if include_weights {
            writeln!(output, "source,target,weight").unwrap();
        } else {
            writeln!(output, "source,target").unwrap();
        }

        // Edges
        for (&(src, dst), &weight) in &self.edges {
            if include_weights {
                writeln!(output, "{},{},{}", src, dst, weight).unwrap();
            } else {
                writeln!(output, "{},{}", src, dst).unwrap();
            }
        }

        output
    }

    /// Exports the adjacency matrix as CSV.
    ///
    /// # Returns
    ///
    /// CSV string representation of the adjacency matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let csv = graph.to_adjacency_matrix_csv();
    /// println!("{}", csv);
    /// ```
    pub fn to_adjacency_matrix_csv(&self) -> String {
        let matrix = self.to_adjacency_matrix();
        let mut output = String::new();

        // Header with node indices
        write!(output, "node").unwrap();
        for i in 0..self.node_count {
            write!(output, ",{}", i).unwrap();
        }
        writeln!(output).unwrap();

        // Rows
        for (i, row) in matrix.iter().enumerate() {
            write!(output, "{}", i).unwrap();
            for &val in row {
                write!(output, ",{}", val).unwrap();
            }
            writeln!(output).unwrap();
        }

        output
    }

    /// Exports node features to CSV format.
    ///
    /// # Returns
    ///
    /// CSV string with columns: node, feature_name1, feature_name2, ...
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph, FeatureSet, BuiltinFeature};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .with_features(
    ///         FeatureSet::new().add_builtin(BuiltinFeature::DeltaForward)
    ///     )
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let csv = graph.features_to_csv();
    /// println!("{}", csv);
    /// ```
    pub fn features_to_csv(&self) -> String {
        if self.node_features.is_empty() {
            return String::from("node\n");
        }

        let mut output = String::new();

        // Collect all feature names
        let mut all_features: Vec<String> = Vec::new();
        for features in &self.node_features {
            for name in features.keys() {
                if !all_features.contains(name) {
                    all_features.push(name.clone());
                }
            }
        }
        all_features.sort();

        // Header
        write!(output, "node").unwrap();
        for name in &all_features {
            write!(output, ",{}", name).unwrap();
        }
        writeln!(output).unwrap();

        // Data rows
        for (i, features) in self.node_features.iter().enumerate() {
            write!(output, "{}", i).unwrap();
            for name in &all_features {
                if let Some(value) = features.get(name) {
                    write!(output, ",{}", value).unwrap();
                } else {
                    write!(output, ",").unwrap();
                }
            }
            writeln!(output).unwrap();
        }

        output
    }

    /// Exports the graph to GraphViz DOT format for visualization.
    ///
    /// The DOT format can be rendered with GraphViz tools like `dot`, `neato`,
    /// or online tools like GraphvizOnline.
    ///
    /// # Returns
    ///
    /// DOT format string
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let dot = graph.to_dot();
    /// std::fs::write("graph.dot", dot).unwrap();
    /// // Then: dot -Tpng graph.dot -o graph.png
    /// ```
    pub fn to_dot(&self) -> String {
        let mut output = String::new();

        let graph_type = if self.directed { "digraph" } else { "graph" };
        let edge_op = if self.directed { "->" } else { "--" };

        writeln!(output, "{} {{", graph_type).unwrap();
        writeln!(output, "  rankdir=LR;").unwrap();
        writeln!(output, "  node [shape=circle];").unwrap();
        writeln!(output).unwrap();

        // Nodes
        for i in 0..self.node_count {
            write!(output, "  {} [label=\"{}\"]", i, i).unwrap();

            // Add features as tooltip if present
            if let Some(features) = self.node_features.get(i) {
                if !features.is_empty() {
                    write!(output, " [tooltip=\"").unwrap();
                    let mut first = true;
                    for (name, value) in features {
                        if !first {
                            write!(output, "\\n").unwrap();
                        }
                        write!(output, "{}: {:.2}", name, value).unwrap();
                        first = false;
                    }
                    write!(output, "\"]").unwrap();
                }
            }
            writeln!(output, ";").unwrap();
        }

        writeln!(output).unwrap();

        // Edges
        for (&(src, dst), &weight) in &self.edges {
            if self.directed || src < dst {
                write!(output, "  {} {} {}", src, edge_op, dst).unwrap();
                if weight != 1.0 {
                    write!(output, " [label=\"{:.2}\"]", weight).unwrap();
                }
                writeln!(output, ";").unwrap();
            }
        }

        writeln!(output, "}}").unwrap();
        output
    }

    /// Exports the graph to DOT format with custom node labels.
    ///
    /// # Arguments
    ///
    /// - `node_labels`: Function that returns a label for each node
    ///
    /// # Returns
    ///
    /// DOT format string with custom labels
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let dot = graph.to_dot_with_labels(|i| format!("Node {}", i));
    /// ```
    pub fn to_dot_with_labels<F>(&self, node_labels: F) -> String
    where
        F: Fn(usize) -> String,
    {
        let mut output = String::new();

        let graph_type = if self.directed { "digraph" } else { "graph" };
        let edge_op = if self.directed { "->" } else { "--" };

        writeln!(output, "{} {{", graph_type).unwrap();
        writeln!(output, "  rankdir=LR;").unwrap();
        writeln!(output, "  node [shape=circle];").unwrap();
        writeln!(output).unwrap();

        // Nodes with custom labels
        for i in 0..self.node_count {
            writeln!(output, "  {} [label=\"{}\"];", i, node_labels(i)).unwrap();
        }

        writeln!(output).unwrap();

        // Edges
        for (&(src, dst), &weight) in &self.edges {
            if self.directed || src < dst {
                write!(output, "  {} {} {}", src, edge_op, dst).unwrap();
                if weight != 1.0 {
                    write!(output, " [label=\"{:.2}\"]", weight).unwrap();
                }
                writeln!(output, ";").unwrap();
            }
        }

        writeln!(output, "}}").unwrap();
        output
    }

    /// Exports the graph to GraphML format.
    ///
    /// GraphML is an XML-based file format for graphs, supported by many
    /// graph analysis tools including Gephi, Cytoscape, and yEd.
    ///
    /// # Returns
    ///
    /// GraphML format string
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{TimeSeries, VisibilityGraph};
    ///
    /// let series = TimeSeries::from_raw(vec![1.0, 3.0, 2.0]).unwrap();
    /// let graph = VisibilityGraph::from_series(&series)
    ///     .natural_visibility()
    ///     .unwrap();
    ///
    /// let graphml = graph.to_graphml();
    /// std::fs::write("graph.graphml", graphml).unwrap();
    /// ```
    pub fn to_graphml(&self) -> String {
        let mut output = String::new();

        // XML header
        writeln!(output, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>").unwrap();
        writeln!(output, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"").unwrap();
        writeln!(output, "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"").unwrap();
        writeln!(output, "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns").unwrap();
        writeln!(output, "         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">").unwrap();
        writeln!(output).unwrap();

        // Define attributes
        writeln!(output, "  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>").unwrap();

        // Define feature attributes
        if !self.node_features.is_empty() {
            let mut all_features: HashSet<String> = HashSet::new();
            for features in &self.node_features {
                for name in features.keys() {
                    all_features.insert(name.clone());
                }
            }
            let mut sorted_features: Vec<_> = all_features.into_iter().collect();
            sorted_features.sort();

            for feature in &sorted_features {
                writeln!(output, "  <key id=\"{}\" for=\"node\" attr.name=\"{}\" attr.type=\"double\"/>",
                    feature, feature).unwrap();
            }
        }
        writeln!(output).unwrap();

        // Graph element
        let edge_default = if self.directed { "directed" } else { "undirected" };
        writeln!(output, "  <graph id=\"G\" edgedefault=\"{}\">", edge_default).unwrap();

        // Nodes
        for i in 0..self.node_count {
            writeln!(output, "    <node id=\"n{}\">", i).unwrap();

            // Node features
            if let Some(features) = self.node_features.get(i) {
                for (name, value) in features {
                    writeln!(output, "      <data key=\"{}\">{}</data>", name, value).unwrap();
                }
            }

            writeln!(output, "    </node>").unwrap();
        }

        // Edges
        let mut edge_id = 0;
        for (&(src, dst), &weight) in &self.edges {
            if self.directed || src < dst {
                writeln!(output, "    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">",
                    edge_id, src, dst).unwrap();
                writeln!(output, "      <data key=\"weight\">{}</data>", weight).unwrap();
                writeln!(output, "    </edge>").unwrap();
                edge_id += 1;
            }
        }

        writeln!(output, "  </graph>").unwrap();
        writeln!(output, "</graphml>").unwrap();

        output
    }
}

