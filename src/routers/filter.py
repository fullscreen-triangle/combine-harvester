"""
Filter router implementation.

This module implements a router that filters models based on criteria
and then delegates selection to another router.
"""

import re
from typing import Callable, Dict, List, Optional

from src.routers.base import Router


class FilterRouter(Router):
    """
    Router that filters models based on criteria before delegation.
    
    This router applies filtering criteria to determine eligible models,
    then delegates the actual selection to another router.
    """
    
    def __init__(self, delegate_router: Router, filter_func: Optional[Callable[[str, List[str]], List[str]]] = None):
        """
        Initialize the filter router.
        
        Args:
            delegate_router: Router to delegate to after filtering
            filter_func: Optional function that takes (query, available_models) and returns filtered models.
                         If None, keyword-based filtering will be used.
        """
        self.delegate_router = delegate_router
        self.filter_func = filter_func
        self.keyword_filters = {}  # Map of keywords to model patterns
        
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to a model after filtering.
        
        Args:
            query: The query text
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        filtered_models = self._filter_models(query, available_models)
        
        if not filtered_models:
            # If filtering results in no models, fall back to all available models
            filtered_models = available_models
            
        return self.delegate_router.route(query, filtered_models)
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to multiple models after filtering.
        
        Args:
            query: The query text
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of selected model names
        """
        filtered_models = self._filter_models(query, available_models)
        
        if not filtered_models:
            # If filtering results in no models, fall back to all available models
            filtered_models = available_models
            
        return self.delegate_router.route_multiple(query, filtered_models, k)
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores after filtering.
        
        Args:
            query: The query text
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        filtered_models = self._filter_models(query, available_models)
        
        if not filtered_models:
            # If filtering results in no models, fall back to all available models
            filtered_models = available_models
            
        scores = self.delegate_router.get_confidence_scores(query, filtered_models)
        
        # Ensure all available models have a score, assigning 0.0 to filtered-out models
        for model in available_models:
            if model not in scores:
                scores[model] = 0.0
                
        return scores
    
    def _filter_models(self, query: str, available_models: List[str]) -> List[str]:
        """
        Filter available models based on query.
        
        Args:
            query: The query text
            available_models: List of available model names
            
        Returns:
            Filtered list of model names
        """
        if self.filter_func:
            return self.filter_func(query, available_models)
        else:
            return self._apply_keyword_filters(query, available_models)
    
    def _apply_keyword_filters(self, query: str, available_models: List[str]) -> List[str]:
        """
        Apply keyword-based filtering to the available models.
        
        Args:
            query: The query text
            available_models: List of available model names
            
        Returns:
            Filtered list of model names
        """
        if not self.keyword_filters:
            return available_models
            
        # Check for keyword matches in the query
        matched_patterns = set()
        for keyword, patterns in self.keyword_filters.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                for pattern in patterns:
                    matched_patterns.add(pattern)
        
        if not matched_patterns:
            return available_models
            
        # Filter models based on matched patterns
        filtered_models = []
        for model in available_models:
            for pattern in matched_patterns:
                if re.search(pattern, model, re.IGNORECASE):
                    filtered_models.append(model)
                    break
                    
        return filtered_models
    
    def add_keyword_filter(self, keyword: str, model_patterns: List[str]) -> None:
        """
        Add a keyword-based filter.
        
        Args:
            keyword: Keyword to match in the query
            model_patterns: List of regex patterns to match against model names
        """
        self.keyword_filters[keyword.lower()] = model_patterns
    
    def clear_keyword_filters(self) -> None:
        """Clear all keyword filters."""
        self.keyword_filters = {} 