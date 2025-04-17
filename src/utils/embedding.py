"""
Embedding utilities for the DomainFusion framework.

This module provides functions for vector embeddings operations, similarity
calculations, and dimensionality reduction.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        Cosine similarity score (from -1 to 1, where 1 means identical direction)
    """
    if len(vector_a) == 0 or len(vector_b) == 0:
        return 0.0
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vector_a, vector_b) / (norm_a * norm_b)


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(vector_a - vector_b)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector with unit length
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_similarity_matrix(vectors: List[np.ndarray], 
                              method: str = 'cosine') -> np.ndarray:
    """
    Compute similarity matrix between all pairs of vectors.
    
    Args:
        vectors: List of vectors
        method: Similarity method ('cosine' or 'euclidean')
        
    Returns:
        Similarity matrix where cell [i,j] contains similarity between
        vectors[i] and vectors[j]
    """
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if method == 'cosine':
                sim = cosine_similarity(vectors[i], vectors[j])
            elif method == 'euclidean':
                # Convert distance to similarity (1 / (1 + distance))
                dist = euclidean_distance(vectors[i], vectors[j])
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Matrix is symmetric
    
    return similarity_matrix


def find_most_similar(query_vector: np.ndarray, 
                      vector_database: List[np.ndarray],
                      method: str = 'cosine',
                      top_k: int = 1) -> List[Tuple[int, float]]:
    """
    Find the most similar vectors to a query vector.
    
    Args:
        query_vector: Query vector
        vector_database: List of vectors to search
        method: Similarity method ('cosine' or 'euclidean')
        top_k: Number of most similar vectors to return
        
    Returns:
        List of tuples (index, similarity_score) sorted by similarity
    """
    similarities = []
    
    for i, vector in enumerate(vector_database):
        if method == 'cosine':
            sim = cosine_similarity(query_vector, vector)
        elif method == 'euclidean':
            # Convert distance to similarity (1 / (1 + distance))
            dist = euclidean_distance(query_vector, vector)
            sim = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """
    Apply softmax function to a list of scores.
    
    The temperature parameter controls the "sharpness" of the distribution:
    - Higher temperature (>1) makes the distribution more uniform
    - Lower temperature (<1) makes the distribution more concentrated
    
    Args:
        scores: Input scores
        temperature: Temperature parameter
        
    Returns:
        Softmax probabilities that sum to 1
    """
    if not scores:
        return []
    
    # Apply temperature scaling
    scaled_scores = [s / temperature for s in scores]
    
    # Shift for numerical stability (prevent overflow)
    max_score = max(scaled_scores)
    exp_scores = [np.exp(s - max_score) for s in scaled_scores]
    
    # Normalize to get probabilities
    sum_exp = sum(exp_scores)
    if sum_exp == 0:
        # If all scores are extremely negative, return uniform distribution
        return [1.0 / len(scores) for _ in scores]
    
    return [e / sum_exp for e in exp_scores]


def weighted_average_embeddings(embeddings: List[np.ndarray], 
                                weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute weighted average of multiple embeddings.
    
    Args:
        embeddings: List of embedding vectors
        weights: List of weights (defaults to equal weights if None)
        
    Returns:
        Weighted average embedding
    """
    if not embeddings:
        raise ValueError("Cannot compute average of empty embedding list")
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    
    if len(weights) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(weights)} weights but {len(embeddings)} embeddings")
    
    # Normalize weights to sum to 1
    sum_weights = sum(weights)
    if sum_weights == 0:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    else:
        weights = [w / sum_weights for w in weights]
    
    # Compute weighted average
    result = np.zeros_like(embeddings[0])
    for i, emb in enumerate(embeddings):
        result += weights[i] * emb
    
    return result 