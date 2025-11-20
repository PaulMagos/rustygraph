"""
Comprehensive test suite for RustyGraph Python bindings.

Tests all enhanced functionality including missing data, advanced metrics,
export formats, and statistics.
"""

import pytest
import numpy as np

try:
    import rustygraph as rg
    HAS_RUSTYGRAPH = True
except ImportError:
    HAS_RUSTYGRAPH = False
    pytestmark = pytest.mark.skip("rustygraph not installed")


class TestBasicFunctionality:
    """Test core visibility graph functionality."""

    def test_create_time_series(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        assert len(series) == 4

    def test_natural_visibility(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()
        assert graph.node_count() > 0
        assert graph.edge_count() > 0

    def test_horizontal_visibility(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.horizontal_visibility()
        assert graph.node_count() > 0
        assert graph.edge_count() > 0


class TestMissingDataHandling:
    """Test missing data handling strategies (NEW!)."""

    def test_with_missing(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=[1.0, None, 3.0, None, 2.0]
        )
        assert len(series) == 5

    def test_linear_interpolation(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0, 3.0],
            values=[1.0, None, 3.0, 4.0]
        )
        strategy = rg.MissingDataStrategy.linear_interpolation()
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_forward_fill(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0],
            values=[1.0, None, 3.0]
        )
        strategy = rg.MissingDataStrategy.forward_fill()
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_backward_fill(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0],
            values=[1.0, None, 3.0]
        )
        strategy = rg.MissingDataStrategy.backward_fill()
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_nearest_neighbor(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0],
            values=[1.0, None, 3.0]
        )
        strategy = rg.MissingDataStrategy.nearest_neighbor()
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_mean_imputation(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=[1.0, None, 3.0, None, 2.0]
        )
        strategy = rg.MissingDataStrategy.mean_imputation(3)
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_median_imputation(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=[1.0, None, 3.0, None, 2.0]
        )
        strategy = rg.MissingDataStrategy.median_imputation(3)
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_zero_fill(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0],
            values=[1.0, None, 3.0]
        )
        strategy = rg.MissingDataStrategy.zero_fill()
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None

    def test_strategy_fallback(self):
        series = rg.TimeSeries.with_missing(
            timestamps=[0.0, 1.0, 2.0],
            values=[1.0, None, 3.0]
        )
        strategy = rg.MissingDataStrategy.mean_imputation(5).with_fallback(
            rg.MissingDataStrategy.forward_fill()
        )
        cleaned = series.handle_missing(strategy)
        assert cleaned is not None


class TestCSVImport:
    """Test CSV import functionality (NEW!)."""

    def test_from_csv_string(self):
        csv_data = "time,value\n0.0,1.0\n1.0,3.0\n2.0,2.0\n3.0,4.0"
        series = rg.TimeSeries.from_csv_string(csv_data, "time", "value")
        assert len(series) == 4


class TestAdvancedMetrics:
    """Test advanced graph metrics (NEW!)."""

    def setup_method(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0, 1.0, 3.0])
        self.graph = series.natural_visibility()

    def test_shortest_path_length(self):
        path_len = self.graph.shortest_path_length(0, 2)
        assert path_len is None or path_len >= 0

    def test_average_path_length(self):
        avg_path = self.graph.average_path_length()
        assert avg_path >= 0.0

    def test_radius(self):
        radius = self.graph.radius()
        assert radius >= 0

    def test_is_connected(self):
        connected = self.graph.is_connected()
        assert isinstance(connected, bool)

    def test_count_components(self):
        components = self.graph.count_components()
        assert components >= 1

    def test_largest_component_size(self):
        size = self.graph.largest_component_size()
        assert size > 0

    def test_assortativity(self):
        assort = self.graph.assortativity()
        assert -1.0 <= assort <= 1.0

    def test_degree_variance(self):
        variance = self.graph.degree_variance()
        assert variance >= 0.0

    def test_degree_std_dev(self):
        std_dev = self.graph.degree_std_dev()
        assert std_dev >= 0.0

    def test_degree_distribution(self):
        dist = self.graph.degree_distribution()
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_degree_entropy(self):
        entropy = self.graph.degree_entropy()
        assert entropy >= 0.0

    def test_node_clustering_coefficient(self):
        cc = self.graph.node_clustering_coefficient(0)
        assert cc is None or (0.0 <= cc <= 1.0)

    def test_global_clustering_coefficient(self):
        gcc = self.graph.global_clustering_coefficient()
        assert 0.0 <= gcc <= 1.0

    def test_betweenness_centrality_all(self):
        bc = self.graph.betweenness_centrality_all()
        assert isinstance(bc, list)
        assert len(bc) == self.graph.node_count()
        assert all(x >= 0.0 for x in bc)

    def test_degree_centrality(self):
        dc = self.graph.degree_centrality()
        assert isinstance(dc, list)
        assert len(dc) == self.graph.node_count()
        assert all(0.0 <= x <= 1.0 for x in dc)


class TestStatistics:
    """Test comprehensive statistics (NEW!)."""

    def test_compute_statistics(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0, 1.0])
        graph = series.natural_visibility()
        stats = graph.compute_statistics()

        # Check properties exist and have reasonable values
        assert stats.node_count > 0
        assert stats.edge_count >= 0
        assert isinstance(stats.is_directed, bool)
        assert stats.average_degree >= 0.0
        assert stats.density >= 0.0
        assert stats.diameter >= 0
        assert isinstance(stats.is_connected, bool)
        assert stats.num_components >= 1

    def test_statistics_repr(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()
        stats = graph.compute_statistics()
        repr_str = repr(stats)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


class TestMotifDetection:
    """Test motif detection (NEW!)."""

    def test_detect_motifs(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0, 1.0, 3.0])
        graph = series.natural_visibility()
        motifs = graph.detect_motifs()

        # Check that we can get counts
        counts = motifs.counts()
        assert isinstance(counts, dict)

        # Check total subgraphs
        assert motifs.total_subgraphs >= 0

    def test_motif_get(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()
        motifs = graph.detect_motifs()

        # Try to get a specific motif (may be None)
        triangle_count = motifs.get("triangle")
        assert triangle_count is None or triangle_count >= 0


class TestExportFormats:
    """Test export functionality (NEW!)."""

    def setup_method(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        self.graph = series.natural_visibility()

    def test_to_edge_list_csv(self):
        csv = self.graph.to_edge_list_csv(include_weights=True)
        assert isinstance(csv, str)
        assert len(csv) > 0
        assert "source" in csv or "," in csv

    def test_to_adjacency_csv(self):
        csv = self.graph.to_adjacency_csv()
        assert isinstance(csv, str)
        assert len(csv) > 0

    def test_to_dot(self):
        dot = self.graph.to_dot()
        assert isinstance(dot, str)
        assert "digraph" in dot or "graph" in dot

    def test_to_graphml(self):
        graphml = self.graph.to_graphml()
        assert isinstance(graphml, str)
        assert "graphml" in graphml or "graph" in graphml

    def test_to_json(self):
        json_str = self.graph.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestFeatures:
    """Test feature computation."""

    def test_builtin_features(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0, 1.0])

        features = rg.FeatureSet()
        features.add_builtin(rg.BuiltinFeature("DeltaForward"))
        features.add_builtin(rg.BuiltinFeature("LocalSlope"))

        graph = series.natural_visibility_with_features(features)

        assert graph.has_features()
        assert graph.feature_count() > 0

    def test_get_all_features(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])

        features = rg.FeatureSet()
        features.add_builtin(rg.BuiltinFeature("DeltaForward"))

        graph = series.natural_visibility_with_features(features)
        features_array = graph.get_all_features()

        assert features_array.shape[0] == graph.node_count()
        assert features_array.shape[1] > 0


class TestNumPyIntegration:
    """Test NumPy integration."""

    def test_adjacency_matrix(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()

        adj = graph.adjacency_matrix()
        assert isinstance(adj, np.ndarray)
        assert adj.shape[0] == adj.shape[1]
        assert adj.shape[0] == graph.node_count()

    def test_to_numpy(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        array = series.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) > 0


class TestCommunityDetection:
    """Test community detection."""

    def test_detect_communities(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        graph = series.natural_visibility()

        communities = graph.detect_communities()
        assert isinstance(communities, list)
        assert len(communities) == graph.node_count()
        assert all(c >= 0 for c in communities)


class TestBasicMetrics:
    """Test basic metrics."""

    def test_basic_properties(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()

        assert graph.node_count() == 4
        assert graph.edge_count() > 0
        assert 0.0 <= graph.density() <= 1.0
        assert 0.0 <= graph.clustering_coefficient() <= 1.0
        assert graph.diameter() >= 0

    def test_degree_sequence(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()

        degrees = graph.degree_sequence()
        assert isinstance(degrees, list)
        assert len(degrees) == graph.node_count()
        assert all(d >= 0 for d in degrees)

    def test_edges(self):
        series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
        graph = series.natural_visibility()

        edges = graph.edges()
        assert isinstance(edges, list)
        assert all(len(e) == 3 for e in edges)  # (source, target, weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

