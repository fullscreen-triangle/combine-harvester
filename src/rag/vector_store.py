"""
Vector storage for domain-specific knowledge bases.
"""

import os
from typing import List, Dict, Any, Optional, Union, Callable
import json
import numpy as np
from abc import ABC, abstractmethod

class VectorStore:
    """
    A vector database for storing and retrieving domain-specific knowledge.
    
    This class provides functionality for:
    - Adding documents to the database with domain tags
    - Embedding documents using a provided model
    - Searching the database for relevant documents based on a query
    - Filtering results by domain or other metadata
    """
    
    def __init__(self, embedding_model=None, name: str = "default", storage_path: Optional[str] = None):
        """
        Initialize a vector store.
        
        Args:
            embedding_model: Model used to create embeddings for documents and queries
            name: Name of the vector store (used for multiple domain stores)
            storage_path: Path to store the vector database on disk (if None, in-memory only)
        """
        self.name = name
        self.embedding_model = embedding_model
        self.storage_path = storage_path
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_document(self, document: Union[str, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a document to the vector store.
        
        Args:
            document: Either a string containing the document text or a dictionary with 'text' key
            metadata: Optional metadata to associate with the document (e.g., domain, source)
            
        Returns:
            The document ID in the vector store
        """
        if isinstance(document, dict):
            text = document.get('text', '')
            if metadata is None:
                metadata = {k: v for k, v in document.items() if k != 'text'}
        else:
            text = document
            
        if metadata is None:
            metadata = {}
            
        # Create embedding if model is available
        embedding = None
        if self.embedding_model is not None:
            embedding = self.embedding_model.embed(text)
        
        # Store the document
        doc_id = len(self.documents)
        self.documents.append(text)
        self.metadata.append(metadata)
        self.embeddings.append(embedding)
        
        return doc_id
    
    def add_documents(self, documents: Union[List[str], List[Dict], str], chunk_size: int = 1000, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of documents or path to directory/file containing documents
            chunk_size: Size of chunks when splitting large documents
            metadata: Optional metadata to associate with all documents
            
        Returns:
            List of document IDs in the vector store
        """
        doc_ids = []
        
        # If documents is a string, it's a path to a directory or file
        if isinstance(documents, str):
            if os.path.isdir(documents):
                # Process all files in the directory
                for filename in os.listdir(documents):
                    filepath = os.path.join(documents, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                            
                        # Split the document into chunks if needed
                        chunks = self._split_text(text, chunk_size)
                        
                        for i, chunk in enumerate(chunks):
                            doc_metadata = metadata.copy() if metadata else {}
                            doc_metadata.update({
                                'source': filepath,
                                'chunk': i,
                                'total_chunks': len(chunks)
                            })
                            doc_ids.append(self.add_document(chunk, doc_metadata))
            else:
                # Process a single file
                with open(documents, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                chunks = self._split_text(text, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy() if metadata else {}
                    doc_metadata.update({
                        'source': documents,
                        'chunk': i,
                        'total_chunks': len(chunks)
                    })
                    doc_ids.append(self.add_document(chunk, doc_metadata))
        else:
            # Process a list of documents
            for doc in documents:
                doc_metadata = metadata.copy() if metadata else {}
                if isinstance(doc, dict) and 'metadata' in doc:
                    doc_metadata.update(doc['metadata'])
                    
                doc_ids.append(self.add_document(doc, doc_metadata))
                
        return doc_ids
    
    def query(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query: The query text
            top_k: Number of results to return
            filter: Dictionary for filtering documents by metadata values
            
        Returns:
            List of dictionaries containing document text, metadata, and similarity scores
        """
        # Check if we have an embedding model
        if self.embedding_model is None:
            raise ValueError("No embedding model provided for query operation")
            
        # Embed the query
        query_embedding = self.embedding_model.embed(query)
        
        # Search by embedding
        return self.query_by_embedding(query_embedding, top_k, filter)
    
    def query_by_embedding(self, embedding: List[float], top_k: int = 5, 
                          filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store using a pre-computed embedding.
        
        Args:
            embedding: The query embedding vector
            top_k: Number of results to return
            filter: Dictionary for filtering documents by metadata values
            
        Returns:
            List of dictionaries containing document text, metadata, and similarity scores
        """
        # Check if we have any documents
        if not self.documents:
            return []
        
        # Filter documents if needed
        indices = list(range(len(self.documents)))
        if filter:
            indices = [i for i in indices if self._match_filter(self.metadata[i], filter)]
            
        if not indices:
            return []
            
        # Calculate similarity scores for all documents
        scores = []
        for i in indices:
            if self.embeddings[i] is None:
                scores.append(-1)  # No embedding available
            else:
                similarity = self._cosine_similarity(embedding, self.embeddings[i])
                scores.append(similarity)
        
        # Sort by score and take top_k
        top_indices = [indices[i] for i in np.argsort(scores)[-top_k:]]
        top_scores = [scores[i] for i in np.argsort(scores)[-top_k:]]
        top_indices.reverse()
        top_scores.reverse()
        
        # Construct result objects
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': score,
                'id': idx
            })
            
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the vector store, or None to use self.storage_path
        """
        if path is None:
            path = self.storage_path
            
        if path is None:
            raise ValueError("No storage path provided for save operation")
            
        os.makedirs(path, exist_ok=True)
        
        # Save documents and metadata as JSON
        with open(os.path.join(path, f"{self.name}_documents.json"), 'w') as f:
            json.dump(self.documents, f)
            
        with open(os.path.join(path, f"{self.name}_metadata.json"), 'w') as f:
            json.dump(self.metadata, f)
            
        # Save embeddings as numpy array
        np.save(os.path.join(path, f"{self.name}_embeddings.npy"), np.array(self.embeddings, dtype=object))
        
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Directory path to load the vector store from, or None to use self.storage_path
        """
        if path is None:
            path = self.storage_path
            
        if path is None:
            raise ValueError("No storage path provided for load operation")
            
        # Load documents and metadata
        with open(os.path.join(path, f"{self.name}_documents.json"), 'r') as f:
            self.documents = json.load(f)
            
        with open(os.path.join(path, f"{self.name}_metadata.json"), 'r') as f:
            self.metadata = json.load(f)
            
        # Load embeddings
        embeddings_path = os.path.join(path, f"{self.name}_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path, allow_pickle=True).tolist()
        else:
            self.embeddings = [None] * len(self.documents)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split a large text into smaller chunks.
        
        Args:
            text: The text to split
            chunk_size: Maximum number of characters per chunk
            
        Returns:
            List of text chunks
        """
        # Simple chunk-by-size implementation
        # More sophisticated chunking (by paragraphs, etc.) could be implemented
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
            
        return chunks
    
    def _match_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria.
        
        Args:
            metadata: Document metadata
            filter: Filter criteria
            
        Returns:
            True if metadata matches the filter, False otherwise
        """
        for key, value in filter.items():
            # Handle special operators (e.g., $in, $gt, etc.)
            if isinstance(value, dict) and any(k.startswith('$') for k in value.keys()):
                for op, op_value in value.items():
                    if op == '$in':
                        if key not in metadata or metadata[key] not in op_value:
                            return False
                    elif op == '$gt':
                        if key not in metadata or metadata[key] <= op_value:
                            return False
                    elif op == '$lt':
                        if key not in metadata or metadata[key] >= op_value:
                            return False
                    elif op == '$eq':
                        if key not in metadata or metadata[key] != op_value:
                            return False
                    elif op == '$ne':
                        if key in metadata and metadata[key] == op_value:
                            return False
            else:
                # Direct value comparison
                if key not in metadata or metadata[key] != value:
                    return False
                    
        return True
    
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
