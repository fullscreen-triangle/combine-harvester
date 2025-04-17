"""
Round Robin router implementation.

This module implements a router that cycles through available models in a round-robin fashion.
"""

from typing import Dict, List, Optional

from src.routers.base import Router


class RoundRobinRouter(Router):
    """
    Router that cycles through available models in round-robin fashion.
    
    Keeps track of the last used model index and advances to the next one for each query.
    """
    
    def __init__(self):
        """Initialize the round robin router with a counter."""
        self.current_index = 0
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the next model in the round-robin sequence.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
            
        # Get the model at the current index
        model = available_models[self.current_index % len(available_models)]
        
        # Update the index for the next call
        self.current_index = (self.current_index + 1) % len(available_models)
        
        return model
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to up to k models in round-robin order.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names in round-robin order
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        selected_models = []
        for i in range(k):
            index = (self.current_index + i) % len(available_models)
            selected_models.append(available_models[index])
        
        # Update the index for the next call
        self.current_index = (self.current_index + k) % len(available_models)
        
        return selected_models
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores based on round-robin ordering.
        
        Assigns higher confidence to models that would be selected next in the round-robin order.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        if not available_models:
            return {}
            
        scores = {}
        
        for i, model in enumerate(available_models):
            # Calculate position relative to current index
            position = (i - self.current_index) % len(available_models)
            
            # Assign higher scores to models that come next in the round-robin order
            # Score ranges from 0.9 (next model) down to 0.1 (last model in sequence)
            scores[model] = 0.9 - (position * 0.8 / len(available_models))
                
        return scores 