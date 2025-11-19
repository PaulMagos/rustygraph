"""Type stubs for rustygraph Python bindings."""

from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

class BuiltinFeature:
    """Built-in node feature types."""

    DELTA_FORWARD: str
    DELTA_BACKWARD: str
    DELTA_SYMMETRIC: str
    LOCAL_SLOPE: str
    ACCELERATION: str
    LOCAL_MEAN: str
    LOCAL_VARIANCE: str
    IS_LOCAL_MAX: str
    IS_LOCAL_MIN: str
    ZSCORE: str

    def __init__(self, name: str) -> None:
        """
        Create a builtin feature by name.

        Args:
            name: Feature name (e.g., "DeltaForward", "LocalSlope")
        """
        ...

class FeatureSet:
    """Set of features to compute for nodes."""

    def __init__(self) -> None:
        """Create an empty feature set."""
        ...

    def add_builtin(self, feature: BuiltinFeature) -> 'FeatureSet':
        """
        Add a builtin feature to the set.

        Args:
            feature: BuiltinFeature instance

        Returns:
            Self for method chaining
        """
        ...

class TimeSeries:
    """Time series data container."""

    def __init__(self, values: NDArray[np.float64]) -> None:
        """
        Create a time series from a NumPy array.

        Args:
            values: 1D NumPy array of time series values
        """
        ...

    def natural_visibility(self) -> VisibilityGraph:
        """
        Compute natural visibility graph.

        Returns:
            Visibility graph object
        """
        ...

    def horizontal_visibility(self) -> VisibilityGraph:
        """
        Compute horizontal visibility graph.

        Returns:
            Visibility graph object
        """
        ...

    def natural_visibility_with_features(self, features: FeatureSet) -> VisibilityGraph:
        """
        Compute natural visibility graph with node features.

        Args:
            features: FeatureSet defining which features to compute

        Returns:
            Visibility graph object with node features
        """
        ...

    def horizontal_visibility_with_features(self, features: FeatureSet) -> VisibilityGraph:
        """
        Compute horizontal visibility graph with node features.

        Args:
            features: FeatureSet defining which features to compute

        Returns:
            Visibility graph object with node features
        """
        ...

class VisibilityGraph:
    """Visibility graph representation."""

    def node_count(self) -> int:
        """Get number of nodes in the graph."""
        ...

    def edge_count(self) -> int:
        """Get number of edges in the graph."""
        ...

    def density(self) -> float:
        """Compute graph density."""
        ...

    def degree_sequence(self) -> List[int]:
        """Get degree sequence."""
        ...

    def clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        ...

    def adjacency_matrix(self) -> NDArray[np.float64]:
        """
        Get adjacency matrix as NumPy array.

        Returns:
            2D NumPy array representing the adjacency matrix
        """
        ...

    def edges(self) -> List[Tuple[int, int, float]]:
        """
        Get list of edges.

        Returns:
            List of (source, target, weight) tuples
        """
        ...

    def has_features(self) -> bool:
        """Check if node features were computed."""
        ...

    def feature_count(self) -> int:
        """Get the number of features per node."""
        ...

    def get_node_features(self, node: int) -> Dict[str, float]:
        """
        Get features for a specific node.

        Args:
            node: Node index

        Returns:
            Dictionary mapping feature name to value
        """
        ...

    def get_node_feature_names(self, node: int) -> List[str]:
        """
        Get feature names for a specific node.

        Args:
            node: Node index

        Returns:
            List of feature names
        """
        ...

    def get_all_features(self) -> NDArray[np.float64]:
        """
        Get all node features as a NumPy array.

        Returns:
            2D array of shape (nodes, features)
        """
        ...

def natural_visibility(data: NDArray[np.float64]) -> VisibilityGraph:
    """
    Compute natural visibility graph from time series data.

    Args:
        data: 1D NumPy array of time series values

    Returns:
        Visibility graph object

    Examples:
        >>> import numpy as np
        >>> data = np.sin(np.linspace(0, 10, 100))
        >>> graph = natural_visibility(data)
        >>> print(graph.node_count())
    """
    ...

def horizontal_visibility(data: NDArray[np.float64]) -> VisibilityGraph:
    """
    Compute horizontal visibility graph from time series data.

    Args:
        data: 1D NumPy array of time series values

    Returns:
        Visibility graph object

    Examples:
        >>> import numpy as np
        >>> data = np.sin(np.linspace(0, 10, 100))
        >>> graph = horizontal_visibility(data)
        >>> print(graph.node_count())
    """
    ...

