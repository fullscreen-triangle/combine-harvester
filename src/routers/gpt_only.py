"""
GPT-Only router implementation.

This module implements a router that always selects GPT models over other models.
"""

from typing import Dict, List, Optional

from src.routers.base import Router


class GPTOnlyRouter(Router):
    """
    Router that always selects GPT models over other models.
    
    If multiple GPT models are available, selects the newest one
    (highest version number). If no GPT models are available,
    falls back to other models.
    """
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a GPT model if available, otherwise to the first available model.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
            
        # Filter for GPT models
        gpt_models = [model for model in available_models if 'gpt' in model.lower()]
        
        if gpt_models:
            # Try to get the newest GPT model by sorting based on version number
            try:
                # Extract version numbers and sort
                sorted_models = sorted(
                    gpt_models,
                    key=lambda x: float(x.lower().split('gpt')[-1].split('-')[0]),
                    reverse=True
                )
                return sorted_models[0]
            except (ValueError, IndexError):
                # If sorting fails, just return the first GPT model
                return gpt_models[0]
        
        # If no GPT models available, return the first available model
        return available_models[0]
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to up to k models, prioritizing GPT models.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names, prioritizing GPT models
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        # Separate GPT and non-GPT models
        gpt_models = [model for model in available_models if 'gpt' in model.lower()]
        non_gpt_models = [model for model in available_models if 'gpt' not in model.lower()]
        
        # Try to sort GPT models by version
        try:
            gpt_models = sorted(
                gpt_models,
                key=lambda x: float(x.lower().split('gpt')[-1].split('-')[0]),
                reverse=True
            )
        except (ValueError, IndexError):
            # If sorting fails, leave them in original order
            pass
            
        # Combine the lists with GPT models first
        prioritized_models = gpt_models + non_gpt_models
        
        # Return the first k models
        return prioritized_models[:k]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores, assigning high scores to GPT models and low scores to others.
        
        Args:
            query: The query text (unused)
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        if not available_models:
            return {}
            
        scores = {}
        
        for model in available_models:
            if 'gpt' in model.lower():
                # Try to extract version number for GPT models
                try:
                    version = float(model.lower().split('gpt')[-1].split('-')[0])
                    # Scale score based on version, with a minimum of 0.8
                    scores[model] = min(0.8 + (version / 10), 0.99)
                except (ValueError, IndexError):
                    # If version extraction fails, assign a default high score
                    scores[model] = 0.8
            else:
                # Non-GPT models get a lower score
                scores[model] = 0.3
                
        return scores 