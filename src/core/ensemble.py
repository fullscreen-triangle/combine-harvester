from typing import Any, Dict, List, Optional, Union

from .base import DomainExpert, Mixer, Router
from .registry import ModelRegistry


class Ensemble:
    """
    Router-based ensemble that directs queries to appropriate domain experts.
    
    The Ensemble uses a router to determine which domain expert(s) should handle
    a given query, then dispatches the query to the selected expert(s) and returns
    their responses.
    """

    def __init__(
        self,
        router: Router,
        registry: ModelRegistry,
        mixer: Optional[Mixer] = None,
        default_domain: Optional[str] = None
    ):
        """
        Initialize an Ensemble.
        
        Args:
            router: Router to determine which expert(s) to use
            registry: Registry containing domain expert models
            mixer: Optional mixer for combining responses from multiple experts
            default_domain: Optional default domain to fall back to
        """
        self.router = router
        self.registry = registry
        self.mixer = mixer
        self.default_domain = default_domain

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using the appropriate domain expert(s).
        
        Args:
            query: The input query to process
            context: Optional context information
            
        Returns:
            Response from the expert(s)
        """
        available_domains = self.registry.list_domains()
        
        if not available_domains:
            raise ValueError("No domain experts registered")
        
        # Route the query to domain(s)
        domains = self.router.route(query, available_domains)
        
        # Handle single domain routing
        if isinstance(domains, str):
            domains = [domains]
        
        # Handle case where router couldn't select a domain
        if not domains and self.default_domain:
            domains = [self.default_domain]
        elif not domains:
            # If no suitable domain and no default, use all domains
            domains = available_domains
        
        # Collect responses from each selected domain
        domain_responses: Dict[str, Dict[str, Any]] = {}
        
        for domain_name in domains:
            expert = self.registry.get(domain_name)
            
            if not expert:
                continue
                
            try:
                response = expert.process(query, context)
                domain_responses[domain_name] = response
            except Exception as e:
                # Log error but continue with other domains
                print(f"Error from {domain_name} expert: {str(e)}")
        
        # If no successful responses, raise error
        if not domain_responses:
            raise RuntimeError("No domain experts were able to process the query")
        
        # If only one response, return it directly
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "domain": domain_name,
                "response": domain_responses[domain_name],
                "is_mixed": False
            }
        
        # If multiple responses and mixer provided, mix them
        if self.mixer:
            mixed_response = self.mixer.mix(query, domain_responses)
            return {
                "domains": list(domain_responses.keys()),
                "response": mixed_response,
                "is_mixed": True
            }
        
        # If multiple responses but no mixer, return them all
        return {
            "domains": list(domain_responses.keys()),
            "responses": domain_responses,
            "is_mixed": False
        }
