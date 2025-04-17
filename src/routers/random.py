"""
Random router implementation.

This module implements a router that randomly selects models from the available ones.
"""

import random
from typing import Dict, List, Optional

from src.routers.base import Router


class RandomRouter(Router):
    """
    Router that randomly selects models from the available ones.
    
    All models have an equal probability of being selected.
    """
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a randomly selected model.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
            
        return random.choice(available_models)
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to up to k randomly selected models.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of randomly selected model names
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        # Make a copy to avoid modifying the original list
        models_copy = available_models.copy()
        random.shuffle(models_copy)
        
        return models_copy[:k]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores, assigning random scores to all models.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to random confidence scores
        """
        if not available_models:
            return {}
            
        scores = {}
        
        for model in available_models:
            # Assign random scores between 0.1 and 0.9
            scores[model] = random.uniform(0.1, 0.9)
                
        return scores 