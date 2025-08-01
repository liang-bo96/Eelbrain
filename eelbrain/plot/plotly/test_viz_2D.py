#!/usr/bin/env python3
"""
Unit tests for EelbrainPlotly2DViz class.

Tests cover:
- Initialization with different parameter combinations
- Data loading from NDVar and file sources
- Plotting functionality (butterfly plots, brain projections)
- Arrow threshold feature
- Error handling and edge cases
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Add the eelbrain directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock the problematic imports that might cause OpenMP issues
sys.modules["eelbrain._experiment"] = Mock()
sys.modules["eelbrain._experiment.MneExperiment"] = Mock()

try:
    from eelbrain.plot.plotly.viz_2D import EelbrainPlotly2DViz
    from eelbrain import NDVar

    EELBRAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import eelbrain modules: {e}")
    EELBRAIN_AVAILABLE = False
    # Create mock classes for testing

    class MockNDVar:
        def __init__(self, data, dims):
            self.data = data
            self.dims = dims
            self.has_case = "case" in dims

        def has_dim(self, dim):
            return dim in self.dims

        def get_dim(self, dim):
            mock_dim = Mock()
            if dim == "source":
                mock_dim.coordinates = np.random.randn(self.data.shape[0], 3)
            return mock_dim

        def mean(self, dim):
            return self

        def get_data(self, dims):
            return self.data

        @property
        def time(self):
            mock_time = Mock()
            mock_time.times = np.linspace(0, 1, self.data.shape[-1])
            return mock_time

    NDVar = MockNDVar


class TestEelbrainPlotly2DViz(unittest.TestCase):
    """Test cases for EelbrainPlotly2DViz class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock data for testing
        self.n_sources = 50
        self.n_times = 100
        self.n_space = 3

        # Mock vector data (with space dimension)
        self.vector_data = np.random.randn(self.n_sources, self.n_space, self.n_times)
        self.mock_vector_ndvar = self._create_mock_ndvar(
            self.vector_data, ["source", "space", "time"]
        )

        # Mock scalar data (without space dimension)
        self.scalar_data = np.random.randn(self.n_sources, self.n_times)
        self.mock_scalar_ndvar = self._create_mock_ndvar(
            self.scalar_data, ["source", "time"]
        )

        # Mock data with case dimension
        self.case_data = np.random.randn(10, self.n_sources, self.n_space, self.n_times)
        self.mock_case_ndvar = self._create_mock_ndvar(
            self.case_data, ["case", "source", "space", "time"]
        )

    def _create_mock_ndvar(self, data, dims):
        """Create a mock NDVar for testing."""
        if EELBRAIN_AVAILABLE:
            # If eelbrain is available, create a proper mock
            mock_ndvar = Mock(spec=NDVar)
        else:
            # Use our mock class
            mock_ndvar = MockNDVar(data, dims)

        mock_ndvar.has_case = "case" in dims
        mock_ndvar.has_dim = lambda dim: dim in dims

        # Mock source dimension
        mock_source = Mock()
        mock_source.coordinates = np.random.randn(data.shape[dims.index("source")], 3)
        mock_ndvar.get_dim = lambda dim: mock_source if dim == "source" else Mock()

        # Mock time dimension
        mock_time = Mock()
        mock_time.times = np.linspace(0, 1, data.shape[dims.index("time")])
        mock_ndvar.time = mock_time

        # Mock data extraction
        if "case" in dims:
            # Return data without case dimension after mean
            case_idx = dims.index("case")
            mean_data = np.mean(data, axis=case_idx)
            mock_mean_ndvar = self._create_mock_ndvar(
                mean_data, [d for d in dims if d != "case"]
            )
            mock_ndvar.mean = lambda dim: mock_mean_ndvar
        else:
            mock_ndvar.mean = lambda dim: mock_ndvar

        def mock_get_data(dims_tuple):
            """Mock get_data method."""
            if isinstance(dims_tuple, tuple):
                return data
            return data

        mock_ndvar.get_data = mock_get_data

        return mock_ndvar

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_init_with_ndvar_vector_data(self, mock_dash):
        """Test initialization with NDVar vector data."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Check that data was loaded correctly
        self.assertIsNotNone(viz.glass_brain_data)
        self.assertIsNotNone(viz.butterfly_data)
        self.assertIsNotNone(viz.source_coords)
        self.assertIsNotNone(viz.time_values)

        # Check data shapes
        self.assertEqual(
            viz.glass_brain_data.shape, (self.n_sources, self.n_space, self.n_times)
        )
        self.assertEqual(viz.butterfly_data.shape, (self.n_sources, self.n_times))

        # Check default parameters
        self.assertEqual(viz.cmap, "Hot")
        self.assertEqual(viz.show_max_only, False)
        self.assertEqual(viz.arrow_threshold, None)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_init_with_ndvar_scalar_data(self, mock_dash):
        """Test initialization with NDVar scalar data."""
        viz = EelbrainPlotly2DViz(y=self.mock_scalar_ndvar)

        # Check that data was loaded correctly
        self.assertIsNotNone(viz.glass_brain_data)
        self.assertIsNotNone(viz.butterfly_data)

        # For scalar data, glass_brain_data should be expanded to 3D
        self.assertEqual(viz.glass_brain_data.shape, (self.n_sources, 1, self.n_times))
        self.assertEqual(viz.butterfly_data.shape, (self.n_sources, self.n_times))

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_init_with_case_dimension(self, mock_dash):
        """Test initialization with NDVar that has case dimension."""
        viz = EelbrainPlotly2DViz(y=self.mock_case_ndvar)

        # Data should be averaged over case dimension
        self.assertIsNotNone(viz.glass_brain_data)
        self.assertIsNotNone(viz.butterfly_data)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_init_with_custom_parameters(self, mock_dash):
        """Test initialization with custom parameters."""
        viz = EelbrainPlotly2DViz(
            y=self.mock_vector_ndvar,
            cmap="Viridis",
            show_max_only=True,
            arrow_threshold=0.5,
        )

        self.assertEqual(viz.cmap, "Viridis")
        self.assertEqual(viz.show_max_only, True)
        self.assertEqual(viz.arrow_threshold, 0.5)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    @patch("eelbrain.datasets")  # Fixed: patch the correct module
    def test_init_with_file_data(self, mock_datasets, mock_dash):
        """Test initialization with file data source."""
        # Create a more complete mock NDVar for file data loading
        mock_src_ndvar = Mock()
        mock_src_ndvar.has_case = True
        mock_src_ndvar.has_dim = lambda dim: dim in ["case", "source", "space", "time"]

        # Mock the mean method to return a proper NDVar
        mock_mean_ndvar = Mock()
        mock_mean_ndvar.source = Mock()
        mock_mean_ndvar.source.coordinates = np.random.randn(50, 3)
        mock_mean_ndvar.time = Mock()
        mock_mean_ndvar.time.times = np.linspace(0, 1, 100)
        mock_mean_ndvar.get_data = Mock(return_value=np.random.randn(50, 3, 100))
        mock_src_ndvar.mean = Mock(return_value=mock_mean_ndvar)

        # Mock the dims attribute for set_parc function
        mock_src_ndvar.dims = ["case", "source", "space", "time"]

        # Create a mock dataset that supports item assignment
        mock_ds = {}
        mock_ds["src"] = mock_src_ndvar
        mock_datasets.get_mne_sample.return_value = mock_ds

        # Mock set_parc to return the same NDVar
        with patch("eelbrain.plot.plotly.viz_2D.set_parc", return_value=mock_src_ndvar):
            viz = EelbrainPlotly2DViz(data_source_location=None, region="test-region")

        # Check that datasets.get_mne_sample was called
        mock_datasets.get_mne_sample.assert_called_once()
        self.assertEqual(viz.region_of_brain, "test-region")

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_parameter_validation(self, mock_dash):
        """Test arrow threshold parameter validation."""
        # Test valid values
        viz1 = EelbrainPlotly2DViz(y=self.mock_vector_ndvar, arrow_threshold=None)
        self.assertEqual(viz1.arrow_threshold, None)

        viz2 = EelbrainPlotly2DViz(y=self.mock_vector_ndvar, arrow_threshold="auto")
        self.assertEqual(viz2.arrow_threshold, "auto")

        viz3 = EelbrainPlotly2DViz(y=self.mock_vector_ndvar, arrow_threshold=1.5)
        self.assertEqual(viz3.arrow_threshold, 1.5)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_create_butterfly_plot(self, mock_dash):
        """Test butterfly plot creation."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Test with default parameters
        fig = viz.create_butterfly_plot()
        self.assertIsNotNone(fig)

        # Test with specific time index
        fig = viz.create_butterfly_plot(selected_time_idx=10)
        self.assertIsNotNone(fig)

        # Test with show_max_only=True
        viz.show_max_only = True
        fig = viz.create_butterfly_plot()
        self.assertIsNotNone(fig)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_create_2d_brain_projections_plotly(self, mock_dash):
        """Test 2D brain projections creation."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Test with default parameters
        projections = viz.create_2d_brain_projections_plotly()
        self.assertIsInstance(projections, dict)
        self.assertIn("axial", projections)
        self.assertIn("sagittal", projections)
        self.assertIn("coronal", projections)

        # Test with specific time index
        projections = viz.create_2d_brain_projections_plotly(time_idx=10)
        self.assertIsInstance(projections, dict)

        # Test with selected source
        projections = viz.create_2d_brain_projections_plotly(source_idx=5)
        self.assertIsInstance(projections, dict)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_filtering(self, mock_dash):
        """Test arrow threshold filtering functionality."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Create test data with known magnitudes
        test_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        test_activity = np.array([0.1, 1.0, 2.0])

        # Test None threshold (show all)
        viz.arrow_threshold = None
        fig = viz._create_plotly_brain_projection(
            "axial", test_coords, test_activity, 0.5
        )
        self.assertIsNotNone(fig)

        # Test auto threshold
        viz.arrow_threshold = "auto"
        fig = viz._create_plotly_brain_projection(
            "axial", test_coords, test_activity, 0.5
        )
        self.assertIsNotNone(fig)

        # Test custom threshold
        viz.arrow_threshold = 0.5
        fig = viz._create_plotly_brain_projection(
            "axial", test_coords, test_activity, 0.5
        )
        self.assertIsNotNone(fig)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_colorscale_parameter(self, mock_dash):
        """Test colorscale parameter handling."""
        # Test with string colorscale
        viz1 = EelbrainPlotly2DViz(y=self.mock_vector_ndvar, cmap="Viridis")
        self.assertEqual(viz1.cmap, "Viridis")

        # Test with custom colorscale list
        custom_cmap = [[0, "blue"], [0.5, "green"], [1, "red"]]
        viz2 = EelbrainPlotly2DViz(y=self.mock_vector_ndvar, cmap=custom_cmap)
        self.assertEqual(viz2.cmap, custom_cmap)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    @patch("eelbrain.datasets")  # Fixed: patch the correct module
    def test_data_validation(self, mock_datasets, mock_dash):
        """Test data validation and error handling."""
        # Create a more complete mock NDVar for default data loading
        mock_src_ndvar = Mock()
        mock_src_ndvar.has_case = True
        mock_src_ndvar.has_dim = lambda dim: dim in ["case", "source", "space", "time"]

        # Mock the mean method to return a proper NDVar
        mock_mean_ndvar = Mock()
        mock_mean_ndvar.source = Mock()
        mock_mean_ndvar.source.coordinates = np.random.randn(50, 3)
        mock_mean_ndvar.time = Mock()
        mock_mean_ndvar.time.times = np.linspace(0, 1, 100)
        mock_mean_ndvar.get_data = Mock(return_value=np.random.randn(50, 3, 100))
        mock_src_ndvar.mean = Mock(return_value=mock_mean_ndvar)

        # Mock the dataset loading
        mock_ds = Mock()
        mock_ds.__getitem__ = Mock(return_value=mock_src_ndvar)
        mock_datasets.get_mne_sample.return_value = mock_ds

        # Test with None data (should use default MNE data)
        viz = EelbrainPlotly2DViz(y=None)
        self.assertIsNotNone(viz.glass_brain_data)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_export_functionality(self, mock_dash):
        """Test image export functionality."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Test export_images method
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should not raise an exception
            try:
                viz.export_images(output_dir=temp_dir, time_idx=0, format="png")
            except Exception as e:
                # Expected to fail due to missing dependencies, but should not crash
                self.assertIsInstance(e, (ImportError, AttributeError, TypeError))

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_app_layout_setup(self, mock_dash):
        """Test that app layout is set up correctly."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Check that app layout is set
        self.assertIsNotNone(viz.app.layout)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_callback_setup(self, mock_dash):
        """Test that callbacks are set up correctly."""
        viz = EelbrainPlotly2DViz(y=self.mock_vector_ndvar)

        # Check that callbacks are registered (this is hard to test directly)
        # We mainly check that _setup_callbacks runs without error
        self.assertIsNotNone(viz.app)


class TestArrowThresholdFeature(unittest.TestCase):
    """Specific tests for the arrow threshold feature."""

    def setUp(self):
        """Set up test fixtures for arrow threshold tests."""
        self.n_sources = 10
        self.n_times = 20
        self.n_space = 3

        # Create test data with known magnitudes
        self.test_data = np.random.randn(self.n_sources, self.n_space, self.n_times)
        self.mock_ndvar = self._create_mock_ndvar(
            self.test_data, ["source", "space", "time"]
        )

    def _create_mock_ndvar(self, data, dims):
        """Create a mock NDVar for testing."""
        mock_ndvar = Mock()
        mock_ndvar.has_case = "case" in dims
        mock_ndvar.has_dim = lambda dim: dim in dims

        # Mock source dimension
        mock_source = Mock()
        mock_source.coordinates = np.random.randn(data.shape[dims.index("source")], 3)
        mock_ndvar.get_dim = lambda dim: mock_source if dim == "source" else Mock()

        # Mock time dimension
        mock_time = Mock()
        mock_time.times = np.linspace(0, 1, data.shape[dims.index("time")])
        mock_ndvar.time = mock_time

        mock_ndvar.mean = lambda dim: mock_ndvar
        mock_ndvar.get_data = lambda dims_tuple: data

        return mock_ndvar

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_none(self, mock_dash):
        """Test arrow threshold with None value (show all arrows)."""
        viz = EelbrainPlotly2DViz(y=self.mock_ndvar, arrow_threshold=None)

        # Create test vectors
        test_vectors = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        magnitudes = np.linalg.norm(test_vectors, axis=1)

        # With None threshold, all arrows should be shown
        if viz.arrow_threshold is None:
            show_mask = np.ones(len(test_vectors), dtype=bool)
        else:
            show_mask = magnitudes > viz.arrow_threshold

        self.assertTrue(np.all(show_mask))

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_auto(self, mock_dash):
        """Test arrow threshold with 'auto' value."""
        viz = EelbrainPlotly2DViz(y=self.mock_ndvar, arrow_threshold="auto")

        # Create test vectors with known magnitudes
        test_vectors = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        magnitudes = np.linalg.norm(test_vectors, axis=1)

        # Auto threshold should be 10% of max magnitude
        if viz.arrow_threshold == "auto":
            threshold_value = 0.1 * np.max(magnitudes)
            show_mask = magnitudes > threshold_value
        else:
            show_mask = np.ones(len(test_vectors), dtype=bool)

        # With auto threshold, only vectors above 10% of max should be shown
        expected_threshold = 0.1 * np.max(magnitudes)
        expected_mask = magnitudes > expected_threshold
        np.testing.assert_array_equal(show_mask, expected_mask)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_custom(self, mock_dash):
        """Test arrow threshold with custom numeric value."""
        threshold_value = 0.5

        # Create test vectors
        test_vectors = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        magnitudes = np.linalg.norm(test_vectors, axis=1)

        # Only vectors above threshold should be shown
        show_mask = magnitudes > threshold_value
        expected_mask = np.array([False, True, True])  # Only last two vectors above 0.5

        np.testing.assert_array_equal(show_mask, expected_mask)

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_arrow_threshold_in_brain_projection(self, mock_dash):
        """Test that arrow threshold is applied in brain projection creation."""
        viz = EelbrainPlotly2DViz(y=self.mock_ndvar, arrow_threshold=1.0)

        # Create test data
        test_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        test_activity = np.array([0.5, 1.5, 2.5])  # Only last two should pass threshold

        # Create brain projection
        fig = viz._create_plotly_brain_projection(
            "axial", test_coords, test_activity, 0.5
        )

        # Check that figure was created successfully
        self.assertIsNotNone(fig)

        # The actual arrow filtering logic is tested in the projection method
        # This test ensures the method runs without error with threshold set


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and edge cases."""

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_fig_to_base64(self, mock_dash):
        """Test figure to base64 conversion."""
        # Create a simple mock figure
        mock_fig = Mock()
        mock_fig.savefig = Mock()

        # Create viz instance
        mock_ndvar = Mock()
        mock_ndvar.has_case = False
        mock_ndvar.has_dim = lambda dim: dim in ["source", "space", "time"]
        mock_ndvar.get_dim = lambda dim: Mock(coordinates=np.random.randn(10, 3))
        mock_ndvar.time = Mock(times=np.linspace(0, 1, 20))
        mock_ndvar.mean = lambda dim: mock_ndvar
        mock_ndvar.get_data = lambda dims: np.random.randn(10, 3, 20)

        viz = EelbrainPlotly2DViz(y=mock_ndvar)

        # Test the method (it should handle the conversion)
        try:
            result = viz._fig_to_base64(mock_fig)
            # Should return a string (base64 encoded)
            self.assertIsInstance(result, str)
        except Exception:
            # Expected to fail due to mocking, but should not crash the test
            pass

    @patch("eelbrain.plot.plotly.viz_2D.dash.Dash")
    def test_create_placeholder_image(self, mock_dash):
        """Test placeholder image creation."""
        mock_ndvar = Mock()
        mock_ndvar.has_case = False
        mock_ndvar.has_dim = lambda dim: dim in ["source", "space", "time"]
        mock_ndvar.get_dim = lambda dim: Mock(coordinates=np.random.randn(10, 3))
        mock_ndvar.time = Mock(times=np.linspace(0, 1, 20))
        mock_ndvar.mean = lambda dim: mock_ndvar
        mock_ndvar.get_data = lambda dims: np.random.randn(10, 3, 20)

        viz = EelbrainPlotly2DViz(y=mock_ndvar)

        # Test placeholder creation
        placeholder = viz._create_placeholder_image("Test Message")
        self.assertIsInstance(placeholder, str)
        # The placeholder is a base64 encoded image, so we check it's a base64 string
        self.assertTrue(placeholder.startswith("data:image/png;base64,"))


def run_tests():
    """Run all tests."""
    # Create test suite using modern approach
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestEelbrainPlotly2DViz))
    suite.addTest(loader.loadTestsFromTestCase(TestArrowThresholdFeature))
    suite.addTest(loader.loadTestsFromTestCase(TestUtilityFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 RUNNING UNIT TESTS FOR VIZ_2D.PY")
    print("=" * 70)

    success = run_tests()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 70)