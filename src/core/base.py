from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class DomainExpert(ABC):
    """Base class for domain-specific expert models."""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name

    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query with domain-specific expertise.
        
        Args:
            query: The input query to process
            context: Optional context information
            
        Returns:
            Dictionary containing response and any additional metadata
        """
        pass


class Router(ABC):
    """Base class for routers that direct queries to appropriate domain experts."""

    @abstractmethod
    def route(self, query: str, available_domains: List[str]) -> Union[str, List[str]]:
        """
        Route a query to the most appropriate domain(s).
        
        Args:
            query: The input query to route
            available_domains: List of available domain names
            
        Returns:
            Either a single domain name or a list of domain names
        """
        pass


class Mixer(ABC):
    """Base class for mixers that combine responses from multiple domain experts."""

    @abstractmethod
    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine responses from multiple domain experts.
        
        Args:
            query: The original input query
            domain_responses: Dictionary mapping domain names to their responses
            
        Returns:
            A unified response combining insights from multiple domains
        """
        pass


class ContextManager(ABC):
    """Base class for managing context in sequential chains."""

    @abstractmethod
    def update(self, domain_name: str, response: Dict[str, Any]) -> None:
        """
        Update the context with a new domain response.
        
        Args:
            domain_name: Name of the domain that produced the response
            response: The response from the domain expert
        """
        pass

    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context.
        
        Returns:
            The current context as a dictionary
        """
        pass


class OutputProcessor(ABC):
    """Base class for processing the final output of a chain or ensemble."""

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the final context to generate the output.
        
        Args:
            context: The final context after all processing
            
        Returns:
            The final processed output
        """
        pass
