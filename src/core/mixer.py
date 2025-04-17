from typing import Any, Dict, List, Optional

from .base import Mixer


class DefaultMixer(Mixer):
    """
    Default implementation of a mixer that combines responses from multiple domain experts.
    
    This mixer identifies common fields across all domain responses and integrates them,
    while preserving unique contributions from each domain.
    """

    def __init__(self, primary_domain: Optional[str] = None):
        """
        Initialize a DefaultMixer.
        
        Args:
            primary_domain: Optional name of the domain to prioritize when conflicts arise
        """
        self.primary_domain = primary_domain
    
    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine responses from multiple domain experts.
        
        Args:
            query: The original input query
            domain_responses: Dictionary mapping domain names to their responses
            
        Returns:
            A unified response combining insights from multiple domains
        """
        if not domain_responses:
            return {"error": "No domain responses to mix"}
        
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "source_domain": domain_name,
                **domain_responses[domain_name]
            }
        
        # Start with primary domain if specified, otherwise use first domain
        if self.primary_domain and self.primary_domain in domain_responses:
            base_domain = self.primary_domain
        else:
            base_domain = list(domain_responses.keys())[0]
        
        # Start with the base domain's response
        mixed_response = domain_responses[base_domain].copy()
        mixed_response["source_domains"] = [base_domain]
        
        # Add contributions from other domains
        for domain, response in domain_responses.items():
            if domain == base_domain:
                continue
            
            # Add this domain to the list of sources
            mixed_response["source_domains"].append(domain)
            
            # Merge responses
            for key, value in response.items():
                if key not in mixed_response:
                    # Add unique fields from this domain
                    mixed_response[key] = value
                    mixed_response[f"{key}_source"] = domain
                elif isinstance(value, list) and isinstance(mixed_response[key], list):
                    # Combine lists, avoiding duplicates
                    mixed_response[key] = list(set(mixed_response[key] + value))
                elif isinstance(value, dict) and isinstance(mixed_response[key], dict):
                    # Recursively merge dictionaries
                    mixed_response[key].update(value)
                # For conflicting scalar values, keep the base domain's value by default
        
        return mixed_response


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


class VotingMixer(Mixer):
    """
    Mixer that uses voting mechanisms to select the best response.
    
    This mixer is useful when domains might provide conflicting information,
    and a consensus approach is needed.
    """

    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine responses from multiple domain experts using voting.
        
        Args:
            query: The original input query
            domain_responses: Dictionary mapping domain names to their responses
            
        Returns:
            A response based on consensus among domains
        """
        if not domain_responses:
            return {"error": "No domain responses to mix"}
        
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "source_domain": domain_name,
                **domain_responses[domain_name]
            }
        
        # Extract common fields across all responses
        common_fields = set.intersection(
            *[set(response.keys()) for response in domain_responses.values()]
        )
        
        # Create a voting-based response
        mixed_response = {
            "source_domains": list(domain_responses.keys()),
            "voting_results": {}
        }
        
        # For each common field, tally votes
        for field in common_fields:
            # Count occurrences of each value
            value_counts = {}
            for response in domain_responses.values():
                value = str(response[field])  # Convert to string for hashability
                if value not in value_counts:
                    value_counts[value] = 0
                value_counts[value] += 1
            
            # Find the value with the most votes
            if value_counts:
                winner = max(value_counts, key=value_counts.get)
                
                # Try to convert back to original type
                for response in domain_responses.values():
                    if str(response[field]) == winner:
                        winner_value = response[field]
                        break
                else:
                    winner_value = winner
                
                mixed_response[field] = winner_value
                mixed_response["voting_results"][field] = value_counts
        
        return mixed_response
