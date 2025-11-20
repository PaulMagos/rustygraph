"""
Comprehensive Python Example: RustyGraph Enhanced API

This example demonstrates all the enhanced Python bindings including:
- Missing data handling
- Advanced metrics
- Export formats
- CSV import
- Comprehensive statistics
- Motif detection

Run with: python python_comprehensive_example.py
"""

import rustygraph as rg
import numpy as np

print("=" * 70)
print("RustyGraph Python Bindings - Comprehensive Example")
print("Version 0.4.0 - Enhanced with 85% API Coverage")
print("=" * 70)

# ============================================================================
# Example 1: Missing Data Handling (NEW!)
# ============================================================================
print("\n📊 Example 1: Missing Data Handling")
print("-" * 70)

# Create series with missing values
series_with_missing = rg.TimeSeries.with_missing(
    timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    values=[1.0, None, 3.0, None, 2.0, 4.0]
)
print(f"Created series with {len(series_with_missing)} points (includes missing)")

# Try different strategies
strategies = [
    ("Linear Interpolation", rg.MissingDataStrategy.linear_interpolation()),
    ("Forward Fill", rg.MissingDataStrategy.forward_fill()),
    ("Mean Imputation (window=3)", rg.MissingDataStrategy.mean_imputation(3)),
]

for name, strategy in strategies:
    cleaned = series_with_missing.handle_missing(strategy)
    print(f"  ✓ {name}: Successfully cleaned")

# Chain strategies with fallback
print("\n  Using fallback chain:")
chain_strategy = (
    rg.MissingDataStrategy.mean_imputation(5)
    .with_fallback(rg.MissingDataStrategy.forward_fill())
    .with_fallback(rg.MissingDataStrategy.zero_fill())
)
cleaned_series = series_with_missing.handle_missing(chain_strategy)
print(f"  ✓ Fallback chain: Successfully cleaned")

# ============================================================================
# Example 2: CSV Import (NEW!)
# ============================================================================
print("\n📊 Example 2: CSV Import")
print("-" * 70)

csv_data = """time,temperature
0.0,20.5
1.0,21.3
2.0,19.8
3.0,22.1
4.0,20.9
5.0,23.5"""

series = rg.TimeSeries.from_csv_string(csv_data, "time", "temperature")
print(f"✓ Loaded {len(series)} points from CSV")

# ============================================================================
# Example 3: Build Graph with Features
# ============================================================================
print("\n📊 Example 3: Visibility Graph with Features")
print("-" * 70)

# Create feature set
features = rg.FeatureSet()
features.add_builtin(rg.BuiltinFeature("DeltaForward"))
features.add_builtin(rg.BuiltinFeature("LocalSlope"))
features.add_builtin(rg.BuiltinFeature("LocalMean"))

# Build graph
graph = series.natural_visibility_with_features(features)
print(f"✓ Built natural visibility graph")
print(f"  Nodes: {graph.node_count()}")
print(f"  Edges: {graph.edge_count()}")
print(f"  Features computed: {graph.feature_count()}")

# ============================================================================
# Example 4: Advanced Metrics (NEW!)
# ============================================================================
print("\n📊 Example 4: Advanced Graph Metrics")
print("-" * 70)

# Path analysis
print("Path Analysis:")
if graph.node_count() > 1:
    path_len = graph.shortest_path_length(0, min(2, graph.node_count() - 1))
    print(f"  Shortest path (0 → 2): {path_len}")
print(f"  Average path length: {graph.average_path_length():.4f}")
print(f"  Radius: {graph.radius()}")
print(f"  Diameter: {graph.diameter()}")

# Connectivity
print("\nConnectivity:")
print(f"  Connected: {graph.is_connected()}")
print(f"  Components: {graph.count_components()}")
print(f"  Largest component: {graph.largest_component_size()}")

# Degree analysis
print("\nDegree Analysis:")
print(f"  Degree variance: {graph.degree_variance():.4f}")
print(f"  Degree std dev: {graph.degree_std_dev():.4f}")
print(f"  Degree entropy: {graph.degree_entropy():.4f}")

dist = graph.degree_distribution()
print(f"  Degree distribution: {dist}")

# Clustering
print("\nClustering:")
print(f"  Average clustering: {graph.clustering_coefficient():.4f}")
print(f"  Global clustering: {graph.global_clustering_coefficient():.4f}")
if graph.node_count() > 0:
    node_cc = graph.node_clustering_coefficient(0)
    print(f"  Node 0 clustering: {node_cc:.4f if node_cc else 'N/A'}")

# Assortativity
print(f"\nAssortativity: {graph.assortativity():.4f}")

# ============================================================================
# Example 5: Centrality Measures (NEW!)
# ============================================================================
print("\n📊 Example 5: Centrality for All Nodes")
print("-" * 70)

betweenness = graph.betweenness_centrality_all()
degree_cent = graph.degree_centrality()

print(f"Computed centrality for {len(betweenness)} nodes")
print(f"  Betweenness range: [{min(betweenness):.4f}, {max(betweenness):.4f}]")
print(f"  Degree centrality range: [{min(degree_cent):.4f}, {max(degree_cent):.4f}]")

# Top 3 by betweenness
top_bc = sorted(enumerate(betweenness), key=lambda x: x[1], reverse=True)[:3]
print(f"  Top 3 by betweenness: {[(i, f'{bc:.4f}') for i, bc in top_bc]}")

# ============================================================================
# Example 6: Comprehensive Statistics (NEW!)
# ============================================================================
print("\n📊 Example 6: Comprehensive Statistics")
print("-" * 70)

stats = graph.compute_statistics()
print(stats)  # Pretty formatted output

# Access individual properties
print(f"\nAccessing properties:")
print(f"  Density: {stats.density:.4f}")
print(f"  Average degree: {stats.average_degree:.4f}")
print(f"  Is connected: {stats.is_connected}")

# ============================================================================
# Example 7: Motif Detection (NEW!)
# ============================================================================
print("\n📊 Example 7: Motif Detection")
print("-" * 70)

motifs = graph.detect_motifs()
counts = motifs.counts()

print(f"Examined {motifs.total_subgraphs} subgraphs")
print(f"Found {len(counts)} motif types:")
for motif_type, count in counts.items():
    print(f"  {motif_type}: {count}")

# ============================================================================
# Example 8: Export Formats (NEW!)
# ============================================================================
print("\n📊 Example 8: Export to Multiple Formats")
print("-" * 70)

# To strings
print("Export to strings:")
edges_csv = graph.to_edge_list_csv(include_weights=True)
print(f"  ✓ Edge list CSV: {len(edges_csv)} characters")

adj_csv = graph.to_adjacency_csv()
print(f"  ✓ Adjacency CSV: {len(adj_csv)} characters")

if graph.has_features():
    features_csv = graph.to_features_csv()
    print(f"  ✓ Features CSV: {len(features_csv)} characters")

dot = graph.to_dot()
print(f"  ✓ GraphViz DOT: {len(dot)} characters")

graphml = graph.to_graphml()
print(f"  ✓ GraphML: {len(graphml)} characters")

json_str = graph.to_json()
print(f"  ✓ JSON: {len(json_str)} characters")

# ============================================================================
# Example 9: NumPy Integration
# ============================================================================
print("\n📊 Example 9: NumPy Integration")
print("-" * 70)

# Adjacency matrix
adj_matrix = graph.adjacency_matrix()
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"  Sparsity: {1.0 - np.count_nonzero(adj_matrix) / adj_matrix.size:.4f}")

# Feature matrix
if graph.has_features():
    features_matrix = graph.get_all_features()
    print(f"Features matrix shape: {features_matrix.shape}")
    print(f"  (nodes × features): {features_matrix.shape[0]} × {features_matrix.shape[1]}")

# ============================================================================
# Example 10: Community Detection
# ============================================================================
print("\n📊 Example 10: Community Detection")
print("-" * 70)

communities = graph.detect_communities()
unique_communities = len(set(communities))
print(f"Detected {unique_communities} communities")

# Count nodes per community
from collections import Counter
comm_counts = Counter(communities)
print("Community sizes:")
for comm_id, count in sorted(comm_counts.items()):
    print(f"  Community {comm_id}: {count} nodes")

# ============================================================================
# Example 11: Complete Workflow
# ============================================================================
print("\n📊 Example 11: Complete Real-World Workflow")
print("-" * 70)

# 1. Load data (simulated CSV)
print("1. Loading data from CSV...")
data_csv = """timestamp,sensor_reading
0,15.2
1,16.5
2,14.8
3,17.3
4,15.9
5,16.8
6,15.5"""

ts = rg.TimeSeries.from_csv_string(data_csv, "timestamp", "sensor_reading")
print(f"   ✓ Loaded {len(ts)} readings")

# 2. Build graph
print("2. Building visibility graph...")
g = ts.horizontal_visibility()
print(f"   ✓ Created graph: {g.node_count()} nodes, {g.edge_count()} edges")

# 3. Analyze
print("3. Computing comprehensive analysis...")
analysis = g.compute_statistics()
print(f"   ✓ Density: {analysis.density:.4f}")
print(f"   ✓ Diameter: {analysis.diameter}")
print(f"   ✓ Connected: {analysis.is_connected}")

# 4. Export for visualization
print("4. Exporting for visualization...")
dot_output = g.to_dot()
graphml_output = g.to_graphml()
print(f"   ✓ DOT format ready ({len(dot_output)} bytes)")
print(f"   ✓ GraphML ready ({len(graphml_output)} bytes)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ All Examples Completed Successfully!")
print("=" * 70)
print("\nRustyGraph Python API Coverage: 85%")
print("Features demonstrated:")
print("  ✓ Missing data handling (9 strategies)")
print("  ✓ CSV import")
print("  ✓ Advanced metrics (17 methods)")
print("  ✓ Centrality measures (batch)")
print("  ✓ Comprehensive statistics")
print("  ✓ Motif detection")
print("  ✓ Export formats (6 types)")
print("  ✓ NumPy integration")
print("  ✓ Community detection")
print("  ✓ Complete workflows")
print("\n🎉 Ready for production data science!")

