"""
Retrieval-Augmented Generation (RAG) with Multiple Knowledge Bases

This module implements the RAG architectural pattern with support for 
domain-specific knowledge bases and retrieval strategies.
"""

from .vector_store import VectorStore
from .retriever import Retriever, HybridRetriever, DomainRetriever
from .augmenter import MultiDomainRAG, DomainRouter

__all__ = [
    'VectorStore', 
    'Retriever', 
    'HybridRetriever', 
    'DomainRetriever',
    'MultiDomainRAG',
    'DomainRouter'
]
