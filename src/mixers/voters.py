from typing import Any, Dict, List, Optional, Union
from collections import Counter

from .base import Mixer


class VotingMixer(Mixer):
    """
    Mixer that uses voting mechanisms to select the best response.
    
    This mixer is useful when domains might provide conflicting information,
    and a consensus approach is needed.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize a VotingMixer.
        
        Args:
            threshold: Minimum proportion of votes needed to accept a value (default: 0.5)
        """
        self.threshold = threshold

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
                winner_votes = value_counts[winner]
                total_votes = len(domain_responses)
                
                # Only include consensus values that meet the threshold
                if winner_votes / total_votes >= self.threshold:
                    # Try to convert back to original type
                    for response in domain_responses.values():
                        if str(response[field]) == winner:
                            winner_value = response[field]
                            break
                    else:
                        winner_value = winner
                    
                    mixed_response[field] = winner_value
                    mixed_response["voting_results"][field] = value_counts
        
        # If no fields passed the voting threshold, use the most common domain
        if len(mixed_response) == 2:  # Only source_domains and voting_results
            # Find the domain that has the most fields in common with others
            domain_scores = {d: 0 for d in domain_responses}
            for field in common_fields:
                value_counts = Counter()
                for domain, response in domain_responses.items():
                    value_counts[str(response[field])] += 1
                
                # Give points to domains that provided the most common value
                most_common_value = value_counts.most_common(1)[0][0]
                for domain, response in domain_responses.items():
                    if str(response[field]) == most_common_value:
                        domain_scores[domain] += 1
            
            # Use the domain with the highest score
            best_domain = max(domain_scores, key=domain_scores.get)
            mixed_response.update(domain_responses[best_domain])
            mixed_response["fallback_domain"] = best_domain
            mixed_response["domain_consensus_scores"] = domain_scores
        
        return mixed_response
