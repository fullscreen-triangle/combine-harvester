from typing import Dict, List, Optional
import os

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import Model


class AnthropicModel(Model):
    """
    Implementation of the Model interface for Anthropic's Claude models.
    
    This class provides methods to generate text and embeddings using Anthropic's API.
    """
    
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize an Anthropic model.
        
        Args:
            model_name: Name of the Anthropic model to use (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            temperature: Controls randomness in generation (0 = deterministic, 1 = creative)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to Anthropic API
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package is not installed. "
                "Please install it with: pip install anthropic"
            )
        
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        # Set up API key
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided and not found in ANTHROPIC_API_KEY environment variable"
            )
        
        # Initialize client
        self._client = Anthropic(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the given prompt using Anthropic's models.
        
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
        
        # Create message
        response = self._client.messages.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **request_kwargs
        )
        
        return response.content[0].text
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text using Anthropic's embedding models.
        
        Note: If Anthropic doesn't have a native embedding API, this will raise an exception.
        Consider using OpenAI's embeddings instead.
        
        Args:
            text: The input text to generate embeddings for
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            A list of floating point numbers representing the text embedding
        """
        try:
            # Check if Anthropic has an embeddings API available
            response = self._client.embeddings.create(
                model=kwargs.get("embedding_model", self._model_name),
                input=text,
                **kwargs
            )
            return response.embedding
        except (AttributeError, NotImplementedError):
            raise NotImplementedError(
                "Embedding functionality is not currently available in Anthropic's API. "
                "Please use an alternative embedding provider like OpenAI."
            )
