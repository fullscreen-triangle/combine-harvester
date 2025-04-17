from abc import ABC, abstractmethod
from typing import List, Optional


class Model(ABC):
    """Abstract base class for all language models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt to generate a response for
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: The input text to generate embeddings for
            **kwargs: Additional model-specific parameters
            
        Returns:
            A list of floating point numbers representing the text embedding
        """
        pass
