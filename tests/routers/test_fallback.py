"""
Tests for the fallback router.
"""
import unittest
from unittest.mock import MagicMock

from src.routers.fallback import FallbackRouter


class TestFallbackRouter(unittest.TestCase):
    """Tests for the fallback router."""

    def setUp(self):
        """Set up test fixtures."""
        self.primary_router = MagicMock()
        self.fallback_router = MagicMock()
        self.router = FallbackRouter(
            primary_router=self.primary_router,
            fallback_router=self.fallback_router
        )
        self.query = "test query"
        self.available_models = ["model1", "model2", "model3"]

    def test_route_primary_success(self):
        """Test routing when primary router returns a result."""
        self.primary_router.route.return_value = "model1"
        
        result = self.router.route(self.query, self.available_models)
        
        self.assertEqual(result, "model1")
        self.primary_router.route.assert_called_once_with(self.query, self.available_models)
        self.fallback_router.route.assert_not_called()

    def test_route_primary_fails(self):
        """Test routing when primary router fails and fallback succeeds."""
        self.primary_router.route.return_value = None
        self.fallback_router.route.return_value = "model2"
        
        result = self.router.route(self.query, self.available_models)
        
        self.assertEqual(result, "model2")
        self.primary_router.route.assert_called_once_with(self.query, self.available_models)
        self.fallback_router.route.assert_called_once_with(self.query, self.available_models)

    def test_route_both_fail(self):
        """Test routing when both routers fail."""
        self.primary_router.route.return_value = None
        self.fallback_router.route.return_value = None
        
        result = self.router.route(self.query, self.available_models)
        
        self.assertIsNone(result)
        self.primary_router.route.assert_called_once_with(self.query, self.available_models)
        self.fallback_router.route.assert_called_once_with(self.query, self.available_models)

    def test_route_multiple_primary_success(self):
        """Test routing multiple when primary router returns results."""
        self.primary_router.route_multiple.return_value = ["model1", "model2"]
        
        result = self.router.route_multiple(self.query, self.available_models, k=2)
        
        self.assertEqual(result, ["model1", "model2"])
        self.primary_router.route_multiple.assert_called_once_with(self.query, self.available_models, 2)
        self.fallback_router.route_multiple.assert_not_called()

    def test_route_multiple_primary_fails(self):
        """Test routing multiple when primary router fails and fallback succeeds."""
        self.primary_router.route_multiple.return_value = []
        self.fallback_router.route_multiple.return_value = ["model2", "model3"]
        
        result = self.router.route_multiple(self.query, self.available_models, k=2)
        
        self.assertEqual(result, ["model2", "model3"])
        self.primary_router.route_multiple.assert_called_once_with(self.query, self.available_models, 2)
        self.fallback_router.route_multiple.assert_called_once_with(self.query, self.available_models, 2)

    def test_get_confidence_scores(self):
        """Test getting confidence scores from both routers."""
        self.primary_router.get_confidence_scores.return_value = {
            "model1": 0.8,
            "model2": 0.6
        }
        self.fallback_router.get_confidence_scores.return_value = {
            "model1": 0.3,
            "model2": 0.4,
            "model3": 0.5
        }
        
        result = self.router.get_confidence_scores(self.query, self.available_models)
        
        expected = {
            "model1": 0.8,  # From primary router
            "model2": 0.6,  # From primary router
            "model3": 0.5   # From fallback router
        }
        self.assertEqual(result, expected)
        self.primary_router.get_confidence_scores.assert_called_once_with(self.query, self.available_models)
        self.fallback_router.get_confidence_scores.assert_called_once_with(self.query, self.available_models) 