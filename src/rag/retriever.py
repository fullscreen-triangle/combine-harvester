"""
Retriever implementations for domain-specific knowledge retrieval.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
from .vector_store import VectorStore

class Retriever(ABC):
    """
    Abstract base class for retrievers that fetch information from knowledge bases.
    
    Retrievers are responsible for finding relevant information from one or more
    knowledge sources based on a query.
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        pass
    
    @abstractmethod
    def get_relevant_text(self, query: str, top_k: int = 5) -> str:
        """
        Get a combined text string of relevant information for a query.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            Combined text from retrieved documents
        """
        pass


class DomainRetriever(Retriever):
    """
    Retriever that searches a domain-specific vector store.
    
    This retriever is designed to handle a single domain knowledge base
    and can be configured with domain-specific retrieval parameters.
    """
    
    def __init__(self, vector_store: VectorStore, domain: str = None):
        """
        Initialize a domain retriever.
        
        Args:
            vector_store: The vector store to search
            domain: The domain associated with this retriever
        """
        self.vector_store = vector_store
        self.domain = domain
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the domain-specific vector store.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        filter_dict = {"domain": self.domain} if self.domain else None
        results = self.vector_store.query(query, top_k, filter_dict)
        
        # Add domain info to results
        for result in results:
            result["domain"] = self.domain or result["metadata"].get("domain", "unknown")
            
        return results
    
    def get_relevant_text(self, query: str, top_k: int = 5) -> str:
        """
        Get relevant text from the domain-specific knowledge base.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            Combined text from retrieved documents
        """
        results = self.retrieve(query, top_k)
        if not results:
            return ""
            
        # Combine results into a single text string
        text_parts = []
        for result in results:
            text_parts.append(f"[{result['domain']} document, score: {result['score']:.2f}]\n{result['text']}")
            
        return "\n\n".join(text_parts)


class HybridRetriever(Retriever):
    """
    Retriever that combines results from multiple domain-specific retrievers.
    
    This implementation can use different strategies for combining results:
    - Round-robin: Take documents from each domain in turn
    - Weighted: Prioritize domains based on query similarity
    - Top-scoring: Take the highest scoring documents regardless of domain
    """
    
    def __init__(self, retrievers: List[DomainRetriever], embedding_model=None, 
                 strategy: str = "top_scoring"):
        """
        Initialize a hybrid retriever.
        
        Args:
            retrievers: List of domain retrievers
            embedding_model: Model for computing query-domain similarities (for weighted strategy)
            strategy: Retrieval strategy ("round_robin", "weighted", or "top_scoring")
        """
        self.retrievers = retrievers
        self.embedding_model = embedding_model
        self.strategy = strategy
        
        # Create domain embeddings if using weighted strategy and embedding model is available
        self.domain_embeddings = {}
        if strategy == "weighted" and embedding_model is not None:
            for retriever in retrievers:
                if retriever.domain:
                    self.domain_embeddings[retriever.domain] = embedding_model.embed(retriever.domain)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from multiple domain-specific retrievers.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.strategy == "round_robin":
            return self._retrieve_round_robin(query, top_k)
        elif self.strategy == "weighted":
            return self._retrieve_weighted(query, top_k)
        else:  # top_scoring (default)
            return self._retrieve_top_scoring(query, top_k)
    
    def _retrieve_round_robin(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using round-robin strategy.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Calculate how many documents to fetch from each retriever
        n_retrievers = len(self.retrievers)
        if n_retrievers == 0:
            return []
            
        # Get documents per retriever, with extras going to first retrievers
        docs_per_retriever = [top_k // n_retrievers] * n_retrievers
        extras = top_k % n_retrievers
        for i in range(extras):
            docs_per_retriever[i] += 1
            
        # Fetch documents from each retriever
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            if docs_per_retriever[i] > 0:
                results = retriever.retrieve(query, docs_per_retriever[i])
                all_results.extend(results)
                
        # Interleave results
        final_results = []
        for i in range(max(docs_per_retriever)):
            for j, retriever in enumerate(self.retrievers):
                if i < docs_per_retriever[j] and i < len(all_results):
                    idx = sum(docs_per_retriever[:j]) + i
                    if idx < len(all_results):
                        final_results.append(all_results[idx])
                        
        return final_results[:top_k]
    
    def _retrieve_weighted(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using weighted strategy based on domain relevance.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.embedding_model:
            # Fall back to top_scoring if no embedding model
            return self._retrieve_top_scoring(query, top_k)
            
        # Compute query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Compute domain weights based on similarity to query
        domain_weights = {}
        for domain, embedding in self.domain_embeddings.items():
            # Compute cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            domain_weights[domain] = max(0, similarity)  # Ensure non-negative
            
        # Normalize weights
        total_weight = sum(domain_weights.values())
        if total_weight > 0:
            for domain in domain_weights:
                domain_weights[domain] /= total_weight
        else:
            # Equal weights if all similarities are non-positive
            for domain in domain_weights:
                domain_weights[domain] = 1.0 / len(domain_weights)
                
        # Allocate documents per retriever based on weights
        docs_per_retriever = {}
        for retriever in self.retrievers:
            weight = domain_weights.get(retriever.domain, 1.0 / len(self.retrievers))
            docs_per_retriever[retriever] = max(1, int(top_k * weight))
            
        # Adjust to ensure we get exactly top_k documents
        total_docs = sum(docs_per_retriever.values())
        if total_docs < top_k:
            # Add extras to highest weighted retrievers
            sorted_retrievers = sorted(
                self.retrievers, 
                key=lambda r: domain_weights.get(r.domain, 0),
                reverse=True
            )
            for i in range(top_k - total_docs):
                docs_per_retriever[sorted_retrievers[i % len(sorted_retrievers)]] += 1
        elif total_docs > top_k:
            # Remove extras from lowest weighted retrievers
            sorted_retrievers = sorted(
                self.retrievers, 
                key=lambda r: domain_weights.get(r.domain, 0)
            )
            for i in range(total_docs - top_k):
                retriever = sorted_retrievers[i % len(sorted_retrievers)]
                if docs_per_retriever[retriever] > 1:
                    docs_per_retriever[retriever] -= 1
        
        # Fetch documents from each retriever
        all_results = []
        for retriever, doc_count in docs_per_retriever.items():
            results = retriever.retrieve(query, doc_count)
            all_results.extend(results)
            
        # Sort by score and take top_k
        return sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
    
    def _retrieve_top_scoring(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve top scoring documents regardless of domain.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            # Get more results from each retriever to ensure we have enough to choose from
            retriever_top_k = min(top_k * 2, 20)  # Reasonable upper limit
            results = retriever.retrieve(query, retriever_top_k)
            all_results.extend(results)
            
        # Sort by score and take top_k
        return sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
    
    def get_relevant_text(self, query: str, top_k: int = 5) -> str:
        """
        Get relevant text from multiple domain knowledge bases.
        
        Args:
            query: The query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            Combined text from retrieved documents
        """
        results = self.retrieve(query, top_k)
        if not results:
            return ""
            
        # Group results by domain
        domain_results = {}
        for result in results:
            domain = result.get('domain', 'unknown')
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
            
        # Build text with domain sections
        text_parts = []
        for domain, domain_docs in domain_results.items():
            text_parts.append(f"--- {domain.upper()} KNOWLEDGE ---")
            for doc in domain_docs:
                text_parts.append(f"[Score: {doc['score']:.2f}]\n{doc['text']}")
                
        return "\n\n".join(text_parts)
