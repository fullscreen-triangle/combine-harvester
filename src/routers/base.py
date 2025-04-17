"""
Base router implementation.

This module defines the Router abstract base class that all router implementations must extend.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Router(ABC):
    """
    Abstract base class for model routers.
    
    Routers are responsible for selecting the most appropriate model(s)
    for a given query from a list of available models.
    """
    
    @abstractmethod
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the most appropriate model.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available or suitable
        """
        pass
    
    @abstractmethod
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to the k most appropriate models.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names, in order of confidence (highest first)
        """
        pass
    
    @abstractmethod
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores for each available model for a given query.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        pass
