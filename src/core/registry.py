from typing import Dict, List, Optional, Type

from .base import DomainExpert


class ModelRegistry:
    """
    Registry for managing domain expert models.
    Provides functionality to register, retrieve, and list available domain experts.
    """

    def __init__(self):
        """Initialize an empty model registry."""
        self._models: Dict[str, DomainExpert] = {}
        self._model_types: Dict[str, Type[DomainExpert]] = {}

    def register(self, domain_name: str, model: DomainExpert) -> None:
        """
        Register a domain expert model.
        
        Args:
            domain_name: The name of the domain
            model: The domain expert model instance
        """
        if domain_name in self._models:
            raise ValueError(f"Domain '{domain_name}' already registered")
        
        self._models[domain_name] = model

    def register_type(self, domain_name: str, model_type: Type[DomainExpert]) -> None:
        """
        Register a domain expert model type for lazy instantiation.
        
        Args:
            domain_name: The name of the domain
            model_type: The domain expert model class
        """
        if domain_name in self._model_types:
            raise ValueError(f"Domain type '{domain_name}' already registered")
        
        self._model_types[domain_name] = model_type

    def get(self, domain_name: str) -> Optional[DomainExpert]:
        """
        Get a domain expert model by name.
        
        Args:
            domain_name: The name of the domain
            
        Returns:
            The domain expert model if registered, None otherwise
        """
        # If model is already instantiated, return it
        if domain_name in self._models:
            return self._models[domain_name]
        
        # If model type is registered, instantiate it
        if domain_name in self._model_types:
            model_type = self._model_types[domain_name]
            model = model_type(domain_name)
            self._models[domain_name] = model
            return model
        
        return None

    def list_domains(self) -> List[str]:
        """
        List all registered domain names.
        
        Returns:
            List of registered domain names
        """
        # Combine keys from both dictionaries
        all_domains = set(list(self._models.keys()) + list(self._model_types.keys()))
        return list(all_domains)

    def remove(self, domain_name: str) -> bool:
        """
        Remove a domain expert model from the registry.
        
        Args:
            domain_name: The name of the domain to remove
            
        Returns:
            True if removed, False if not found
        """
        removed = False
        
        if domain_name in self._models:
            del self._models[domain_name]
            removed = True
            
        if domain_name in self._model_types:
            del self._model_types[domain_name]
            removed = True
            
        return removed

    def clear(self) -> None:
        """Clear all registered models and model types."""
        self._models.clear()
        self._model_types.clear()
