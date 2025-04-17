"""
Multi-domain RAG implementation for integrating knowledge across domains.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
import numpy as np
import re

from .vector_store import VectorStore
from .retriever import Retriever, DomainRetriever, HybridRetriever

class DomainRouter:
    """
    Router that determines which knowledge bases are relevant to a query.
    
    The domain router analyzes a query and determines which domain-specific
    knowledge bases should be consulted. This enables efficient retrieval
    by focusing only on relevant knowledge sources.
    """
    
    def __init__(self, embedding_model=None, threshold: float = 0.5):
        """
        Initialize a domain router.
        
        Args:
            embedding_model: Model for computing query-domain similarities
            threshold: Minimum similarity threshold for domain relevance
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.domains = {}  # domain_name -> domain_description
        self.domain_embeddings = {}  # domain_name -> embedding
        
    def add_domain(self, name: str, description: str) -> None:
        """
        Add a domain to the router.
        
        Args:
            name: Domain name (e.g., "biomechanics")
            description: Description of the domain for semantic matching
        """
        self.domains[name] = description
        
        # Generate embedding if model is available
        if self.embedding_model:
            self.domain_embeddings[name] = self.embedding_model.embed(description)
            
    def route(self, query: str) -> List[str]:
        """
        Determine which domains are relevant to the query.
        
        Args:
            query: The query string
            
        Returns:
            List of relevant domain names
        """
        if not self.domains:
            return []
            
        # If no embedding model, return all domains
        if not self.embedding_model:
            return list(self.domains.keys())
            
        # Compute query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Compute similarity scores for each domain
        domain_scores = {}
        for domain, embedding in self.domain_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            domain_scores[domain] = similarity
            
        # Filter domains by threshold
        relevant_domains = [
            domain for domain, score in domain_scores.items() 
            if score >= self.threshold
        ]
        
        # If no domains meet the threshold, return the highest scoring domain
        if not relevant_domains and domain_scores:
            top_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            relevant_domains = [top_domain]
            
        return relevant_domains
    
    def route_with_scores(self, query: str) -> Dict[str, float]:
        """
        Determine domain relevance scores for a query.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary mapping domain names to relevance scores
        """
        if not self.domains:
            return {}
            
        # If no embedding model, assign equal scores to all domains
        if not self.embedding_model:
            return {domain: 1.0 for domain in self.domains}
            
        # Compute query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Compute similarity scores for each domain
        domain_scores = {}
        for domain, embedding in self.domain_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            domain_scores[domain] = similarity
            
        return domain_scores
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)


class MultiDomainRAG:
    """
    RAG system that integrates information from multiple domain-specific knowledge bases.
    
    This class implements the Retrieval-Augmented Generation pattern with
    support for multiple domains, enabling cross-domain knowledge integration.
    """
    
    def __init__(self, generation_model, domain_router: Optional[DomainRouter] = None, 
                retrievers: Optional[Dict[str, Retriever]] = None,
                template: str = None):
        """
        Initialize a multi-domain RAG system.
        
        Args:
            generation_model: Model used to generate responses
            domain_router: Router for determining relevant domains
            retrievers: Dictionary mapping domain names to retrievers
            template: Prompt template for generation
        """
        self.generation_model = generation_model
        self.domain_router = domain_router or DomainRouter()
        self.retrievers = retrievers or {}
        
        # Default template if none provided
        self.template = template or """
        Answer the question based on the provided context.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
    
    def add_knowledge_base(self, name: str, vector_store: VectorStore, 
                          description: Optional[str] = None) -> None:
        """
        Add a domain-specific knowledge base to the system.
        
        Args:
            name: Domain name
            vector_store: Vector store for the domain
            description: Domain description for the router
        """
        # Create a retriever for the domain
        retriever = DomainRetriever(vector_store, domain=name)
        self.retrievers[name] = retriever
        
        # Add domain to router
        if description and self.domain_router:
            self.domain_router.add_domain(name, description)
    
    def generate(self, query: str, max_tokens: int = 1000, 
                use_all_retrievers: bool = False) -> str:
        """
        Generate a response to a query using relevant knowledge bases.
        
        Args:
            query: The query string
            max_tokens: Maximum response length
            use_all_retrievers: If True, use all knowledge bases regardless of relevance
            
        Returns:
            Generated response
        """
        # Identify relevant domains
        if use_all_retrievers or not self.domain_router:
            relevant_domains = list(self.retrievers.keys())
        else:
            relevant_domains = self.domain_router.route(query)
            
        # Get relevant retrievers
        relevant_retrievers = [
            self.retrievers[domain] for domain in relevant_domains 
            if domain in self.retrievers
        ]
        
        if not relevant_retrievers:
            # If no retrievers are relevant, generate response without context
            return self.generation_model.generate(f"Question: {query}\n\nAnswer:")
            
        # Create a hybrid retriever for the relevant domains
        hybrid_retriever = HybridRetriever(relevant_retrievers)
        
        # Retrieve context
        context = hybrid_retriever.get_relevant_text(query)
        
        # Generate response
        prompt = self.template.format(context=context, query=query)
        response = self.generation_model.generate(prompt, max_tokens=max_tokens)
        
        return response
    
    def generate_with_sources(self, query: str, max_tokens: int = 1000) -> Tuple[str, List[Dict]]:
        """
        Generate a response with source references.
        
        Args:
            query: The query string
            max_tokens: Maximum response length
            
        Returns:
            Tuple of (generated_response, source_documents)
        """
        # Identify relevant domains
        if not self.domain_router:
            relevant_domains = list(self.retrievers.keys())
        else:
            relevant_domains = self.domain_router.route(query)
            
        # Get relevant retrievers
        relevant_retrievers = [
            self.retrievers[domain] for domain in relevant_domains 
            if domain in self.retrievers
        ]
        
        if not relevant_retrievers:
            # If no retrievers are relevant, generate response without context
            return self.generation_model.generate(f"Question: {query}\n\nAnswer:"), []
            
        # Create a hybrid retriever for the relevant domains
        hybrid_retriever = HybridRetriever(relevant_retrievers)
        
        # Retrieve documents
        documents = hybrid_retriever.retrieve(query)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[{i+1}] {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Add source citation instructions to template
        source_template = self.template + "\nCite sources using [1], [2], etc."
        
        # Generate response
        prompt = source_template.format(context=context, query=query)
        response = self.generation_model.generate(prompt, max_tokens=max_tokens)
        
        return response, documents
    
    def add_domain_specific_templates(self, templates: Dict[str, str]) -> None:
        """
        Add domain-specific prompt templates.
        
        Args:
            templates: Dictionary mapping domain names to prompt templates
        """
        self.domain_templates = templates
        
    def generate_with_domain_templates(self, query: str, max_tokens: int = 1000) -> str:
        """
        Generate a response using domain-specific templates.
        
        Args:
            query: The query string
            max_tokens: Maximum response length
            
        Returns:
            Generated response
        """
        # Identify relevant domains with scores
        if not self.domain_router:
            domain_scores = {domain: 1.0 for domain in self.retrievers.keys()}
        else:
            domain_scores = self.domain_router.route_with_scores(query)
            
        # Sort domains by relevance
        sorted_domains = sorted(
            domain_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Start with most relevant domain's template
        primary_domain = sorted_domains[0][0] if sorted_domains else None
        
        if primary_domain and hasattr(self, 'domain_templates') and primary_domain in self.domain_templates:
            template = self.domain_templates[primary_domain]
        else:
            template = self.template
            
        # Get relevant retrievers
        relevant_retrievers = [
            self.retrievers[domain] for domain, score in sorted_domains 
            if domain in self.retrievers and score > 0
        ]
        
        if not relevant_retrievers:
            # If no retrievers are relevant, generate response without context
            return self.generation_model.generate(f"Question: {query}\n\nAnswer:")
            
        # Create a hybrid retriever for the relevant domains
        hybrid_retriever = HybridRetriever(relevant_retrievers)
        
        # Retrieve context
        context = hybrid_retriever.get_relevant_text(query)
        
        # Generate response
        prompt = template.format(context=context, query=query)
        response = self.generation_model.generate(prompt, max_tokens=max_tokens)
        
        return response
