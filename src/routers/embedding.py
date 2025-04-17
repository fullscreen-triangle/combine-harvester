"""
Embedding-based router implementation.

This module provides a router that uses embeddings to match queries with
the most semantically similar domains.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import Router


class EmbeddingRouter(Router):
    """
    Routes queries based on embedding similarity.
    
    This router computes embeddings for the input query and compares them
    with pre-computed domain embeddings to find the most semantically
    similar domains.
    """
    
    def __init__(self, embedding_model, threshold: float = 0.5):
        """
        Initialize an embedding router.
        
        Args:
            embedding_model: Any model with an encode method that converts text to vectors
            threshold: Minimum similarity threshold (0.0 to 1.0) for a model to be selected
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.domains = {}  # domain_name -> description
        self.domain_examples = {}  # domain_name -> list of example queries
        self.domain_embeddings = {}  # domain_name -> embedding vector
        
    def add_domain(self, name: str, description: str) -> None:
        """
        Add a domain to the router with a description.
        
        Args:
            name: Domain name (should match a model name)
            description: Description of the domain
        """
        self.domains[name] = description
        
    def add_examples(self, domain: str, examples: List[str]) -> None:
        """
        Add example queries for a domain and compute their embeddings.
        
        Args:
            domain: Domain name
            examples: List of example queries that are representative of this domain
        """
        self.domain_examples[domain] = examples
        
        # Compute average embedding from all examples for this domain
        if examples:
            embeddings = self.embedding_model.encode(examples)
            if len(examples) > 1:
                # Average multiple embeddings if multiple examples
                domain_embedding = np.mean(embeddings, axis=0)
            else:
                domain_embedding = embeddings[0]
            
            # Ensure the embedding is normalized
            domain_embedding = domain_embedding / np.linalg.norm(domain_embedding)
            self.domain_embeddings[domain] = domain_embedding
    
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """
        Route a query to the most appropriate model based on embedding similarity.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            
        Returns:
            The name of the most appropriate model, or None if no appropriate model is found
        """
        scores = self._compute_scores(query, available_models)
        
        # Find the model with the highest score above threshold
        best_model = None
        best_score = self.threshold
        
        for model, score in scores.items():
            if score > best_score:
                best_model = model
                best_score = score
                
        return best_model
    
    def route_multiple(self, query: str, available_models: List[str], k: int = 2) -> List[str]:
        """
        Route a query to the k most appropriate models based on embedding similarity.
        
        Args:
            query: The query text to route
            available_models: List of available model names to choose from
            k: Number of models to return
            
        Returns:
            List of model names, ordered by relevance (most relevant first)
        """
        scores = self._compute_scores(query, available_models)
        
        # Sort models by score in descending order
        sorted_models = sorted(
            [(model, score) for model, score in scores.items() if score >= self.threshold],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k models
        return [model for model, _ in sorted_models[:k]]
    
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
    
    def _compute_scores(self, query: str, available_models: List[str]) -> Dict[str, float]:
        """
        Compute similarity scores for each available model.
        
        Args:
            query: The query text
            available_models: List of available model names
            
        Returns:
            Dictionary mapping model names to similarity scores (0.0 to 1.0)
        """
        # Compute query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores = {}
        
        for model in available_models:
            if model not in self.domain_embeddings:
                scores[model] = 0.0
                continue
                
            # Compute cosine similarity between query and domain embeddings
            domain_embedding = self.domain_embeddings[model]
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                domain_embedding.reshape(1, -1)
            )[0][0]
            
            scores[model] = float(similarity)
                
        return scores
