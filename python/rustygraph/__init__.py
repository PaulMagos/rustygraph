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

from importlib.metadata import PackageNotFoundError, version as _version

from ._rustygraph import (
    TimeSeries,
    VisibilityGraph,
    BuiltinFeature,
    FeatureSet,
    MissingDataStrategy,
    GraphStatistics,
    MotifCounts,
    natural_visibility,
    horizontal_visibility,
    natural_visibility_edges,
    natural_visibility_plan,
    horizontal_visibility_edges,
    natural_visibility_batch,
    horizontal_visibility_batch,
    natural_vector_visibility_edges,
    horizontal_vector_visibility_edges,
    natural_vector_visibility_batch,
    horizontal_vector_visibility_batch,
    VisibilityStream,
    visibility_motifs,
)

from .auto import visibility
from .metrics import vg_descriptors, vg_fidelity

_TORCH_NAMES = {
    "exact_visibility", "soft_visibility", "VisibilityLayer", "to_undirected",
    "edge_index_from_adjacency", "soft_degree", "soft_degree_histogram",
}


def __getattr__(name):  # lazy: torch is optional
    if name in _TORCH_NAMES or name == "torch_vg":
        import importlib

        torch_vg = importlib.import_module(".torch_vg", __name__)
        return torch_vg if name == "torch_vg" else getattr(torch_vg, name)
    raise AttributeError(f"module 'rustygraph' has no attribute {name!r}")


try:
    __version__ = _version("pyrustygraph")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+local"

__all__ = [
    "TimeSeries",
    "VisibilityGraph",
    "BuiltinFeature",
    "FeatureSet",
    "MissingDataStrategy",
    "GraphStatistics",
    "MotifCounts",
    "natural_visibility",
    "horizontal_visibility",
    "natural_visibility_edges",
    "natural_visibility_plan",
    "horizontal_visibility_edges",
    "natural_visibility_batch",
    "horizontal_visibility_batch",
    "natural_vector_visibility_edges",
    "horizontal_vector_visibility_edges",
    "natural_vector_visibility_batch",
    "horizontal_vector_visibility_batch",
    "VisibilityStream",
    "visibility_motifs",
    "visibility",
    "vg_descriptors",
    "vg_fidelity",
]
