"""
Ensemble router implementation.

This module implements a router that combines results from multiple routers
to make more robust routing decisions.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from .base import Router


class EnsembleRouter(Router):
    """
    A router that combines results from multiple routers.
    
    This router uses a set of underlying routers and an aggregation method
    to determine the final model selection for a given query.
    """
    
    def __init__(self, routers: List[Router], weights: Optional[List[float]] = None):
        """
        Initialize the ensemble router.
        
        Args:
            routers: List of Router instances to ensemble
            weights: Optional weights for each router. Must have the same length as routers.
                     If None, equal weights are assigned to all routers.
        """
        self.routers = routers
        
        # Validate and normalize weights
        if weights is not None:
            if len(weights) != len(routers):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of routers ({len(routers)})")
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights if not provided
            self.weights = [1.0 / len(routers)] * len(routers)
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a model based on ensemble of router decisions.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
        
        # Get confidence scores from all routers
        combined_scores = self._get_combined_scores(query, available_models)
        
        # Return the model with the highest combined score
        if combined_scores:
            return max(combined_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to k models based on ensemble of router decisions.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of k model names with the highest combined confidence scores
        """
        if not available_models:
            return []
        
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        # Get combined confidence scores
        combined_scores = self._get_combined_scores(query, available_models)
        
        # Sort models by score and return top k
        sorted_models = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:k]]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get combined confidence scores from all routers for each available model.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to combined confidence scores
        """
        return self._get_combined_scores(query, available_models)
    
    def _get_combined_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Calculate combined confidence scores from all routers.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to combined confidence scores
        """
        if not available_models:
            return {}
        
        # Initialize scores
        combined_scores = {model: 0.0 for model in available_models}
        
        # Get scores from each router and combine them according to weights
        for i, router in enumerate(self.routers):
            scores = router.get_confidence_scores(query, available_models)
            weight = self.weights[i]
            
            # Add weighted scores to the combined scores
            for model, score in scores.items():
                if model in combined_scores:
                    combined_scores[model] += weight * score
        
        return combined_scores 