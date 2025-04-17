"""
Confidence-based router implementation.

This module implements a router that selects models based on their confidence scores,
choosing the model with the highest confidence for the given query.
"""

from typing import Dict, List, Optional, Tuple, Callable
import heapq
from .base import Router


class ConfidenceRouter(Router):
    """
    A router that selects models based on confidence scores.
    
    This router uses a confidence scoring function to determine which model
    is most suitable for a given query, selecting the one with the highest score.
    """
    
    def __init__(self, confidence_fn: Callable[[str, List[str]], Dict[str, float]]):
        """
        Initialize the confidence router.
        
        Args:
            confidence_fn: Function that takes a query and list of available models
                          and returns a dictionary mapping models to confidence scores
        """
        self.confidence_fn = confidence_fn
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the model with the highest confidence score.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The model name with the highest confidence score, or None if no models are available
        """
        if not available_models:
            return None
            
        # Get confidence scores for each model
        scores = self.get_confidence_scores(query, available_models)
        
        # Return the model with the highest confidence score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to the k models with the highest confidence scores.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of k model names with the highest confidence scores
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        # Get confidence scores for each model
        scores = self.get_confidence_scores(query, available_models)
        
        # Return the k models with the highest confidence scores
        return [model for model, _ in heapq.nlargest(k, scores.items(), key=lambda x: x[1])]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores for each available model for a given query.
        
        Uses the confidence function provided during initialization.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        if not available_models:
            return {}
            
        return self.confidence_fn(query, available_models) 