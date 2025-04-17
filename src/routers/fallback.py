"""
Fallback router implementation.

This module implements a router that tries one router first,
and if it doesn't return a result, falls back to another router.
"""

from typing import Dict, List, Optional

from src.routers.base import Router


class FallbackRouter(Router):
    """
    Router that provides fallback behavior.
    
    This router first tries to route using a primary router.
    If the primary router doesn't return a result, it falls back
    to using a secondary router.
    """
    
    def __init__(self, primary_router: Router, fallback_router: Router):
        """
        Initialize the fallback router.
        
        Args:
            primary_router: The first router to try
            fallback_router: The router to use if the primary router fails
        """
        self.primary_router = primary_router
        self.fallback_router = fallback_router
        
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a model, trying the primary router first.
        
        Args:
            query: The query text
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        # Try the primary router first
        primary_result = self.primary_router.route(query, available_models)
        
        # If the primary router returns a result, use it
        if primary_result is not None:
            return primary_result
            
        # Otherwise, try the fallback router
        return self.fallback_router.route(query, available_models)
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to multiple models, trying the primary router first.
        
        Args:
            query: The query text
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names
        """
        # Try the primary router first
        primary_results = self.primary_router.route_multiple(query, available_models, k)
        
        # If the primary router returns any results, use them
        if primary_results:
            return primary_results
            
        # Otherwise, try the fallback router
        return self.fallback_router.route_multiple(query, available_models, k)
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores from both routers, prioritizing the primary router.
        
        Args:
            query: The query text
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        # Get confidence scores from both routers
        primary_scores = self.primary_router.get_confidence_scores(query, available_models)
        fallback_scores = self.fallback_router.get_confidence_scores(query, available_models)
        
        # Start with the fallback scores as the base
        combined_scores = fallback_scores.copy()
        
        # Override with primary scores (they take precedence)
        for model, score in primary_scores.items():
            combined_scores[model] = score
            
        return combined_scores 