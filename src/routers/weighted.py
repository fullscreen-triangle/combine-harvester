"""
Weighted router implementation.

This module implements a router that selects models based on predefined weights,
with higher weights increasing the probability of selection.
"""

import random
from typing import Dict, List, Optional
from .base import Router


class WeightedRouter(Router):
    """
    Router that selects models based on predefined weights.
    
    Models with higher weights have a higher probability of being selected.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the weighted router with model weights.
        
        Args:
            weights: Dictionary mapping model names to weights.
                    If None, all models will be assigned equal weights.
        """
        self.weights = weights or {}
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a model selected based on weights.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
            
        # Get weights for available models
        model_weights = self._get_model_weights(available_models)
        
        # Select a model based on weights
        if sum(model_weights.values()) == 0:
            # If all weights are zero, select randomly
            return random.choice(available_models)
        else:
            # Weighted random selection
            models, weights = zip(*model_weights.items())
            return random.choices(models, weights=weights, k=1)[0]
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to multiple models based on weights.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        # Get weights for available models
        model_weights = self._get_model_weights(available_models)
        
        # Weighted random selection without replacement
        if sum(model_weights.values()) == 0:
            # If all weights are zero, select randomly
            return random.sample(available_models, k)
        else:
            # Sample without replacement using weighted probabilities
            selected_models = []
            remaining_models = list(available_models)
            
            for _ in range(k):
                if not remaining_models:
                    break
                    
                # Recalculate weights for remaining models
                current_weights = {model: model_weights[model] for model in remaining_models}
                models, weights = zip(*current_weights.items())
                
                # Select a model
                selected = random.choices(models, weights=weights, k=1)[0]
                selected_models.append(selected)
                remaining_models.remove(selected)
                
            return selected_models
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores based on model weights.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores derived from weights
        """
        if not available_models:
            return {}
            
        # Get weights for available models
        model_weights = self._get_model_weights(available_models)
        
        # Normalize weights to create confidence scores between 0.1 and 0.9
        total_weight = sum(model_weights.values())
        
        if total_weight == 0:
            # If all weights are zero, assign equal confidence
            equal_confidence = 0.5
            return {model: equal_confidence for model in available_models}
        
        # Calculate normalized confidence scores
        confidence_scores = {}
        for model, weight in model_weights.items():
            # Normalize to [0.1, 0.9] range
            normalized_weight = 0.1 + (weight / total_weight) * 0.8
            confidence_scores[model] = normalized_weight
                
        return confidence_scores
    
    def _get_model_weights(self, available_models: List[str]) -> Dict[str, float]:
        """
        Get weights for available models.
        
        If a model doesn't have a predefined weight, assigns it a default weight of 1.0.
        
        Args:
            available_models: List of available model names
            
        Returns:
            Dictionary mapping available model names to their weights
        """
        model_weights = {}
        
        for model in available_models:
            # Use predefined weight if available, otherwise use default weight of 1.0
            model_weights[model] = self.weights.get(model, 1.0)
                
        return model_weights
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update the weights for models.
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        self.weights = weights 