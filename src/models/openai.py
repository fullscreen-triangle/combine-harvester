from typing import Dict, List, Optional
import os

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import Model


class OpenAIModel(Model):
    """
    Implementation of the Model interface for OpenAI's models (GPT-3.5, GPT-4, etc.).
    
    This class provides methods to generate text and embeddings using OpenAI's API.
    """
    
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize an OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            organization: OpenAI organization ID (defaults to OPENAI_ORGANIZATION environment variable)
            temperature: Controls randomness in generation (0 = deterministic, 1 = creative)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to OpenAI API
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. "
                "Please install it with: pip install openai"
            )
        
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        # Set up API key and organization
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided and not found in OPENAI_API_KEY environment variable"
            )
        
        self._organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        
        # Initialize client
        self._client = OpenAI(
            api_key=self._api_key,
            organization=self._organization if self._organization else None
        )
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the given prompt using OpenAI's models.
        
        Args:
            prompt: The input prompt to generate a response for
            **kwargs: Additional parameters to override the defaults
            
        Returns:
            The generated text response
        """
        # Merge default kwargs with provided kwargs
        request_kwargs = self._kwargs.copy()
        request_kwargs.update(kwargs)
        
        # Handle parameters
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        
        # Check if chat or completion model based on name
        if "gpt" in self._model_name.lower() or "turbo" in self._model_name.lower():
            # Use chat completion API
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs
            )
            return response.choices[0].message.content.strip()
        else:
            # Use completions API for non-chat models
            response = self._client.completions.create(
                model=self._model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs
            )
            return response.choices[0].text.strip()
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text using OpenAI's embedding models.
        
        Args:
            text: The input text to generate embeddings for
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            A list of floating point numbers representing the text embedding
        """
        # Default to appropriate embedding model if not specified
        embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        
        response = self._client.embeddings.create(
            input=text,
            model=embedding_model,
            **kwargs
        )
        
        return response.data[0].embedding
