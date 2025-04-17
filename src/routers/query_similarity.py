"""
Query similarity router implementation.

This module implements a router that selects models based on the similarity
between the current query and past queries that each model has handled successfully.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from .base import Router


class QuerySimilarityRouter(Router):
    """
    A router that selects models based on query similarity to past successful queries.
    
    This router maintains a history of queries handled by each model and their success,
    and routes new queries to models that have successfully handled similar queries in the past.
    """
    
    def __init__(self, embedding_fn=None):
        """
        Initialize the query similarity router.
        
        Args:
            embedding_fn: Function that converts a string to an embedding vector.
                          If None, a simple fallback method is used.
        """
        self.embedding_fn = embedding_fn or self._default_embedding
        self.query_history = defaultdict(list)  # model -> [(query_embedding, success_score), ...]
        
    def _default_embedding(self, text: str) -> np.ndarray:
        """
        Simple fallback embedding function that creates a basic representation of text.
        This is only used if no embedding function is provided.
        
        Args:
            text: The text to embed
            
        Returns:
            A simple vector representation of the text
        """
        # Very simple embedding based on character frequencies (just a fallback)
        vec = np.zeros(128)
        for char in text:
            idx = ord(char) % 128
            vec[idx] += 1
        if np.sum(vec) > 0:
            vec = vec / np.sum(vec)  # Normalize
        return vec
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between the embeddings
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def add_query_result(self, query: str, model: str, success_score: float = 1.0):
        """
        Add a query and its success score to the history for a model.
        
        Args:
            query: The query that was processed
            model: The model that processed the query
            success_score: How successfully the model handled the query (0.0 to 1.0)
        """
        embedding = self.embedding_fn(query)
        self.query_history[model].append((embedding, success_score))
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the model with the highest similarity score.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The selected model name, or None if no models are available
        """
        if not available_models:
            return None
            
        scores = self.get_confidence_scores(query, available_models)
        if not scores:
            # If no history available, return a random model
            return np.random.choice(available_models)
            
        # Return the model with the highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """
        Route a query to k models with the highest similarity scores.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of k selected model names
        """
        if not available_models:
            return []
            
        # Ensure k is not larger than the number of available models
        k = min(k, len(available_models))
        
        scores = self.get_confidence_scores(query, available_models)
        if not scores:
            # If no history available, return k random models
            return list(np.random.choice(available_models, size=k, replace=False))
            
        # Return k models with the highest scores
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:k]]
    
    def get_confidence_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Get confidence scores for each available model for a given query.
        
        Scores are computed based on similarity to past successful queries.
        
        Args:
            query: The query text to route
            available_models: List of available model names to score
            
        Returns:
            Dictionary mapping model names to confidence scores
        """
        if not available_models:
            return {}
            
        query_embedding = self.embedding_fn(query)
        scores = {}
        
        # Compute similarity scores for each model
        for model in available_models:
            if model not in self.query_history or not self.query_history[model]:
                scores[model] = 0.0
                continue
                
            # Calculate weighted similarity to previous queries
            model_score = 0.0
            total_weight = 0.0
            
            for prev_embedding, success_score in self.query_history[model]:
                similarity = self._compute_similarity(query_embedding, prev_embedding)
                model_score += similarity * success_score
                total_weight += success_score
                
            if total_weight > 0:
                scores[model] = model_score / total_weight
            else:
                scores[model] = 0.0
        
        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            for model in scores:
                scores[model] /= total
                
        return scores 