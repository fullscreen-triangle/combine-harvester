"""
Keyword-based router implementation.

This module implements a simple keyword-based router that matches queries
to domains based on keyword presence.
"""

from typing import Dict, List, Optional, Set
from .base import Router


class KeywordRouter(Router):
    """
    A simple router that matches queries to domains based on keyword presence.
    
    This router allows defining keyword sets for each domain and routes queries
    to the domain with the most keyword matches.
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize the keyword router.
        
        Args:
            threshold: Minimum match score required to route a query (0.0 to 1.0)
        """
        self.domains: Dict[str, Set[str]] = {}
        self.threshold = threshold
    
    def add_domain(self, name: str, keywords: List[str]) -> None:
        """
        Add a domain with its associated keywords.
        
        Args:
            name: Domain name (should match a model name)
            keywords: List of keywords associated with the domain
        """
        self.domains[name] = set([kw.lower() for kw in keywords])
    
    def _compute_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Compute match scores for each available model based on keyword matching.
        
        Args:
            query: The query to analyze
            available_models: List of available model names
            
        Returns:
            Dictionary of model names to match scores
        """
        query_words = set(query.lower().split())
        scores = {}
        
        for model in available_models:
            if model not in self.domains:
                scores[model] = 0.0
                continue
                
            domain_keywords = self.domains[model]
            if not domain_keywords:
                scores[model] = 0.0
                continue
                
            # Count how many keywords from the domain are in the query
            matches = query_words.intersection(domain_keywords)
            score = len(matches) / len(domain_keywords) if domain_keywords else 0.0
            scores[model] = score
            
        return scores
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the most appropriate model based on keyword matching.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The name of the most appropriate model, or None if no appropriate model is found
        """
        scores = self._compute_scores(query, available_models)
        
        # Find the model with the highest score
        if not scores:
            return None
            
        best_model = max(scores.items(), key=lambda x: x[1])
        
        if best_model[1] >= self.threshold:
            return best_model[0]
        return None
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to the k most appropriate models based on keyword matching.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of model names, ordered by relevance (most relevant first)
        """
        scores = self._compute_scores(query, available_models)
        
        # Sort models by score in descending order
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter models that meet the threshold
        filtered_models = [model for model, score in sorted_models if score >= self.threshold]
        
        # Return up to k models
        return filtered_models[:k]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores for each available model for a given query.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores (0.0 to 1.0)
        """
        return self._compute_scores(query, available_models)
