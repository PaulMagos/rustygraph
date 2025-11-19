"""
RustyGraph - High-performance visibility graph computation for time series analysis.

This package provides fast Rust-based implementations for computing visibility graphs
from time series data, with seamless NumPy integration.

Examples:
    >>> import rustygraph
    >>> import numpy as np
    >>>
    >>> # Create time series
    >>> data = np.sin(np.linspace(0, 10, 100))
    >>>
    >>> # Create visibility graph
    >>> graph = rustygraph.natural_visibility(data)
    >>>
    >>> # Get graph properties
    >>> print(f"Nodes: {graph.node_count()}")
    >>> print(f"Edges: {graph.edge_count()}")
    >>> print(f"Density: {graph.density():.4f}")
"""

from ._rustygraph import (
    TimeSeries,
    VisibilityGraph,
    BuiltinFeature,
    FeatureSet,
    natural_visibility,
    horizontal_visibility,
)

__version__ = "0.4.0"
__all__ = [
    "TimeSeries",
    "VisibilityGraph",
    "BuiltinFeature",
    "FeatureSet",
    "natural_visibility",
    "horizontal_visibility",
]

