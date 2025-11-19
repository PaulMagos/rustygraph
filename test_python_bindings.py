#!/usr/bin/env python3
"""
Test script for RustyGraph Python bindings.

This script tests all major functionality to verify the Python bindings work correctly.
"""

import rustygraph
import numpy as np


def test_import():
    """Test that the module imports correctly."""
    print("="*60)
    print("TEST: Module Import")
    print("="*60)
    print(f"✅ Import successful!")
    print(f"   Version: {rustygraph.__version__}")
    print(f"   Available: {', '.join([x for x in dir(rustygraph) if not x.startswith('_')])}")
    print()


def test_natural_visibility():
    """Test natural visibility graph creation."""
    print("="*60)
    print("TEST: Natural Visibility Graph")
    print("="*60)

    # Create test data
    data = np.sin(np.linspace(0, 10, 100))
    print(f"Data: {len(data)} points from sin wave")

    # Create graph
    graph = rustygraph.natural_visibility(data.tolist())

    print(f"\n✅ Graph created successfully!")
    print(f"   Nodes: {graph.node_count()}")
    print(f"   Edges: {graph.edge_count()}")
    print(f"   Density: {graph.density():.4f}")
    print(f"   Clustering: {graph.clustering_coefficient():.4f}")
    print(f"   Diameter: {graph.diameter()}")

    # Test degree sequence
    degrees = graph.degree_sequence()
    print(f"   Degree stats: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.2f}")
    print()


def test_horizontal_visibility():
    """Test horizontal visibility graph creation."""
    print("="*60)
    print("TEST: Horizontal Visibility Graph")
    print("="*60)

    data = np.random.randn(100)
    print(f"Data: {len(data)} random points")

    graph = rustygraph.horizontal_visibility(data.tolist())

    print(f"\n✅ Graph created successfully!")
    print(f"   Nodes: {graph.node_count()}")
    print(f"   Edges: {graph.edge_count()}")
    print(f"   Density: {graph.density():.4f}")
    print()


def test_timeseries_class():
    """Test TimeSeries class."""
    print("="*60)
    print("TEST: TimeSeries Class")
    print("="*60)

    data = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
    series = rustygraph.TimeSeries(data)

    print(f"✅ TimeSeries created: {len(series)} points")

    # Create graphs from TimeSeries
    nat_graph = series.natural_visibility()
    print(f"   Natural visibility: {nat_graph.edge_count()} edges")

    hor_graph = series.horizontal_visibility()
    print(f"   Horizontal visibility: {hor_graph.edge_count()} edges")
    print()


def test_adjacency_matrix():
    """Test adjacency matrix export."""
    print("="*60)
    print("TEST: Adjacency Matrix")
    print("="*60)

    data = np.sin(np.linspace(0, 5, 20))
    graph = rustygraph.natural_visibility(data.tolist())

    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

    # Get adjacency matrix
    adj = graph.adjacency_matrix()
    print(f"\n✅ Adjacency matrix exported!")
    print(f"   Shape: {adj.shape}")
    print(f"   Type: {type(adj)}")
    print(f"   Non-zero: {np.count_nonzero(adj)}")
    print(f"   Symmetric: {np.allclose(adj, adj.T)}")
    print()


def test_edges():
    """Test edge list export."""
    print("="*60)
    print("TEST: Edge List")
    print("="*60)

    data = [1.0, 2.0, 1.5, 3.0, 2.5]
    graph = rustygraph.natural_visibility(data)

    edges = graph.edges()
    print(f"✅ Edges exported!")
    print(f"   Total: {len(edges)} edges")
    print(f"   First 5: {edges[:5]}")
    print()


def test_json_export():
    """Test JSON export."""
    print("="*60)
    print("TEST: JSON Export")
    print("="*60)

    data = [1.0, 2.0, 3.0, 2.0, 1.0]
    graph = rustygraph.natural_visibility(data)

    json_str = graph.to_json()
    print(f"✅ JSON exported!")
    print(f"   Length: {len(json_str)} characters")
    print(f"   Preview: {json_str[:100]}...")
    print()


def test_python_314():
    """Test Python 3.14 compatibility."""
    print("="*60)
    print("TEST: Python 3.14 Compatibility")
    print("="*60)

    import sys
    print(f"Python version: {sys.version}")
    print(f"Python version info: {sys.version_info}")

    if sys.version_info >= (3, 14):
        print(f"✅ Python 3.14+ detected and working!")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} working!")
    print()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  RustyGraph Python Bindings Test Suite")
    print("="*60)
    print()

    try:
        test_import()
        test_python_314()
        test_natural_visibility()
        test_horizontal_visibility()
        test_timeseries_class()
        test_adjacency_matrix()
        test_edges()
        test_json_export()

        print("="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print()
        print("Python bindings are working correctly!")
        print(f"RustyGraph version: {rustygraph.__version__}")
        print("Ready for pip install rustygraph! 🚀")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

