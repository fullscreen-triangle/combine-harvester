from abc import ABC, abstractmethod
from typing import Any, Dict


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
