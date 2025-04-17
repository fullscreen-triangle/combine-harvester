from typing import Any, Dict, List, Optional

from .base import Mixer


class WeightedMixer(Mixer):
    """
    Mixer that combines responses with domain-specific weights.
    
    This mixer is useful when some domains are more reliable or relevant
    than others for certain types of responses.
    """

    def __init__(self, domain_weights: Dict[str, float]):
        """
        Initialize a WeightedMixer.
        
        Args:
            domain_weights: Dictionary mapping domain names to their weights
        """
        self.domain_weights = domain_weights
    
    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine responses from multiple domain experts using weights.
        
        Args:
            query: The original input query
            domain_responses: Dictionary mapping domain names to their responses
            
        Returns:
            A weighted combination of responses
        """
        if not domain_responses:
            return {"error": "No domain responses to mix"}
        
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "source_domain": domain_name,
                **domain_responses[domain_name]
            }
        
        # Calculate weights for each domain
        weights = {}
        for domain in domain_responses:
            weights[domain] = self.domain_weights.get(domain, 1.0)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {d: w/total_weight for d, w in weights.items()}
        else:
            # Equal weights if all weights are zero
            normalized_weights = {d: 1.0/len(weights) for d in weights}
        
        # Combine responses with weights
        mixed_response = {
            "source_domains": list(domain_responses.keys()),
            "domain_weights": normalized_weights,
            "weighted_responses": {}
        }
        
        # Include all individual responses with their weights
        for domain, response in domain_responses.items():
            mixed_response["weighted_responses"][domain] = {
                "weight": normalized_weights[domain],
                "response": response
            }
        
        # For the main combined response, use the highest weighted domain
        primary_domain = max(normalized_weights, key=normalized_weights.get)
        mixed_response.update(domain_responses[primary_domain])
        mixed_response["primary_domain"] = primary_domain
        
        return mixed_response
