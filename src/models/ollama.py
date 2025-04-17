from typing import Dict, List, Optional
import os
import json
import requests

from .base import Model


class OllamaModel(Model):
    """
    Implementation of the Model interface for local models running with Ollama.
    
    This class provides methods to generate text and embeddings using models hosted locally via Ollama.
    """
    
    def __init__(
        self, 
        model_name: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize an Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama3", "mistral")
            base_url: Base URL for the Ollama server (defaults to OLLAMA_BASE_URL or "http://localhost:11434")
            temperature: Controls randomness in generation (0 = deterministic, 1 = creative)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to Ollama API
        """
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        # Set up base URL
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Remove trailing slashes from base URL
        self._base_url = self._base_url.rstrip("/")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to the Ollama server."""
        try:
            response = requests.get(f"{self._base_url}/api/version")
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server at {self._base_url}: {str(e)}")
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the given prompt using Ollama's models.
        
        Args:
            prompt: The input prompt to generate a response for
            **kwargs: Additional parameters to override the defaults
            
        Returns:
            The generated text response
        """
        # Merge default kwargs with provided kwargs
        request_kwargs = self._kwargs.copy()
        request_kwargs.update(kwargs)
        
        # Set up request data
        data = {
            "model": self._model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self._temperature),
        }
        
        # Set max_tokens if provided
        if self._max_tokens is not None:
            data["max_tokens"] = self._max_tokens
        
        # Add any additional parameters
        for key, value in request_kwargs.items():
            if key not in data:
                data[key] = value
        
        # Make request
        response = requests.post(
            f"{self._base_url}/api/generate",
            json=data
        )
        response.raise_for_status()
        
        # Parse response
        response_json = response.json()
        return response_json.get("response", "")
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text using Ollama.
        
        Args:
            text: The input text to generate embeddings for
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            A list of floating point numbers representing the text embedding
        """
        # Set up request data
        data = {
            "model": self._model_name,
            "prompt": text,
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            data[key] = value
        
        # Make request
        response = requests.post(
            f"{self._base_url}/api/embeddings",
            json=data
        )
        response.raise_for_status()
        
        # Parse response
        response_json = response.json()
        embeddings = response_json.get("embedding")
        
        if not embeddings:
            raise ValueError(f"No embeddings returned from Ollama server for model {self._model_name}")
        
        return embeddings
