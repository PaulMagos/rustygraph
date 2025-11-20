"""Type stubs for rustygraph Python bindings.

Enhanced with full API coverage including missing data handling,
advanced metrics, and export formats.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

class MissingDataStrategy:
    """Strategy for handling missing data in time series."""

    @staticmethod
    def linear_interpolation() -> 'MissingDataStrategy':
        """Average of neighboring valid values."""
        ...

    @staticmethod
    def forward_fill() -> 'MissingDataStrategy':
        """Use last valid value (carry forward)."""
        ...

    @staticmethod
    def backward_fill() -> 'MissingDataStrategy':
        """Use next valid value (carry backward)."""
        ...

    @staticmethod
    def nearest_neighbor() -> 'MissingDataStrategy':
        """Use closest valid value by distance."""
        ...

    @staticmethod
    def mean_imputation(window_size: int) -> 'MissingDataStrategy':
        """Use mean of local window."""
        ...

    @staticmethod
    def median_imputation(window_size: int) -> 'MissingDataStrategy':
        """Use median of local window."""
        ...

    @staticmethod
    def zero_fill() -> 'MissingDataStrategy':
        """Replace with zero."""
        ...

    @staticmethod
    def drop() -> 'MissingDataStrategy':
        """Skip missing values (return None)."""
        ...

    def with_fallback(self, fallback: 'MissingDataStrategy') -> 'MissingDataStrategy':
        """Chain strategies: try primary, fallback to secondary if it fails."""
        ...

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

    def handle_missing(self, strategy: MissingDataStrategy) -> 'TimeSeries':
        """
        Handle missing data using the specified strategy.

        Args:
            strategy: MissingDataStrategy to apply

        Returns:
            New TimeSeries with missing data handled
        """
        ...

    @staticmethod
    def with_missing(timestamps: List[float], values: List[Optional[float]]) -> 'TimeSeries':
        """
        Create a time series with explicit missing data (None values).

        Args:
            timestamps: Time points
            values: Data values (None for missing)

        Returns:
            TimeSeries instance
        """
        ...

    @staticmethod
    def from_csv_file(path: str, time_col: str, value_col: str) -> 'TimeSeries':
        """
        Load time series from CSV file.

        Args:
            path: Path to CSV file
            time_col: Name of time column
            value_col: Name of value column

        Returns:
            TimeSeries instance
        """
        ...

    @staticmethod
    def from_csv_string(csv: str, time_col: str, value_col: str) -> 'TimeSeries':
        """
        Load time series from CSV string.

        Args:
            csv: CSV string
            time_col: Name of time column
            value_col: Name of value column

        Returns:
            TimeSeries instance
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

    def shortest_path_length(self, source: int, target: int) -> Optional[int]:
        """Compute shortest path length between two nodes."""
        ...

    def average_path_length(self) -> float:
        """Compute average path length across all node pairs."""
        ...

    def radius(self) -> int:
        """Return graph radius (minimum eccentricity)."""
        ...

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        ...

    def count_components(self) -> int:
        """Count number of connected components."""
        ...

    def largest_component_size(self) -> int:
        """Return size of largest component."""
        ...

    def assortativity(self) -> float:
        """Compute assortativity coefficient (degree correlation)."""
        ...

    def degree_variance(self) -> float:
        """Compute degree variance."""
        ...

    def degree_std_dev(self) -> float:
        """Compute degree standard deviation."""
        ...

    def degree_distribution(self) -> Dict[int, int]:
        """Return degree distribution as {degree: count} dictionary."""
        ...

    def degree_entropy(self) -> float:
        """Compute entropy of degree distribution."""
        ...

    def node_clustering_coefficient(self, node: int) -> Optional[float]:
        """Compute clustering coefficient for specific node."""
        ...

    def global_clustering_coefficient(self) -> float:
        """Compute global clustering coefficient (transitivity)."""
        ...

    def betweenness_centrality_all(self) -> List[float]:
        """Compute betweenness centrality for all nodes."""
        ...

    def degree_centrality(self) -> List[float]:
        """Compute degree centrality for all nodes."""
        ...

    def compute_statistics(self) -> 'GraphStatistics':
        """Compute comprehensive statistics for the graph."""
        ...

    def detect_motifs(self) -> 'MotifCounts':
        """Detect motifs in the graph."""
        ...

    def to_edge_list_csv(self, include_weights: bool = True) -> str:
        """Export edges to CSV string."""
        ...

    def to_adjacency_csv(self) -> str:
        """Export adjacency matrix to CSV string."""
        ...

    def to_features_csv(self) -> str:
        """Export node features to CSV string."""
        ...

    def to_dot(self) -> str:
        """Export to DOT format (GraphViz)."""
        ...

    def to_graphml(self) -> str:
        """Export to GraphML format."""
        ...

    def save_edge_list_csv(self, path: str, include_weights: bool = True) -> None:
        """Save edges to CSV file."""
        ...

    def save_adjacency_csv(self, path: str) -> None:
        """Save adjacency matrix to CSV file."""
        ...

    def save_dot(self, path: str) -> None:
        """Save to DOT file."""
        ...

    def save_graphml(self, path: str) -> None:
        """Save to GraphML file."""
        ...

class GraphStatistics:
    """Comprehensive graph statistics."""

    node_count: int
    edge_count: int
    is_directed: bool
    average_degree: float
    min_degree: int
    max_degree: int
    degree_std_dev: float
    degree_variance: float
    average_clustering: float
    global_clustering: float
    average_path_length: float
    diameter: int
    radius: int
    density: float
    is_connected: bool
    num_components: int
    largest_component_size: int
    assortativity: float
    feature_count: int

class MotifCounts:
    """Motif detection results."""

    total_subgraphs: int

    def counts(self) -> Dict[str, int]:
        """Get counts as a dictionary {motif_name: count}."""
        ...

    def get(self, motif_name: str) -> Optional[int]:
        """Get count for a specific motif type."""
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

