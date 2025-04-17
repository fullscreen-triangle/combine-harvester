from typing import Any, Dict, Optional, Protocol

from .base import Mixer


class ModelProtocol(Protocol):
    """Protocol defining the interface for models used in synthesis."""
    
    def generate(self, prompt: str) -> str:
        """Generate a response to the given prompt."""
        ...


class SynthesisMixer(Mixer):
    """
    Mixer that uses an LLM to synthesize responses from multiple domain experts.
    
    This mixer is useful for complex queries where simple aggregation of responses
    isn't sufficient and a more sophisticated integration is needed.
    """

    def __init__(
        self, 
        synthesis_model: ModelProtocol,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize a SynthesisMixer.
        
        Args:
            synthesis_model: Model to use for synthesizing responses
            prompt_template: Optional custom prompt template for synthesis
        """
        self.synthesis_model = synthesis_model
        self.prompt_template = prompt_template or self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Provide the default prompt template for synthesis."""
        return """
        You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.

        Original query: {query}

        Expert responses:
        {domain_responses}

        Create a unified response that integrates insights from all experts, resolving any contradictions and creating a coherent narrative. 
        Preserve the best insights from each domain while eliminating redundancy.
        Focus on providing a comprehensive answer to the original query.
        """
    
    def _format_domain_responses(self, domain_responses: Dict[str, Dict[str, Any]]) -> str:
        """Format domain responses for inclusion in the prompt."""
        formatted = ""
        for domain, response in domain_responses.items():
            formatted += f"\n[{domain.upper()} EXPERT]\n"
            
            # Extract the main response content, assuming it's in a standard format
            if "response" in response:
                content = response["response"]
            elif "content" in response:
                content = response["content"]
            elif "answer" in response:
                content = response["answer"]
            else:
                # Use the whole response if no standard field is found
                content = str(response)
                
            formatted += f"{content}\n"
        
        return formatted
    
    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine responses from multiple domain experts using LLM synthesis.
        
        Args:
            query: The original input query
            domain_responses: Dictionary mapping domain names to their responses
            
        Returns:
            A synthesized response combining insights from multiple domains
        """
        if not domain_responses:
            return {"error": "No domain responses to mix"}
        
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "source_domain": domain_name,
                **domain_responses[domain_name]
            }
        
        # Format the domain responses for the prompt
        formatted_responses = self._format_domain_responses(domain_responses)
        
        # Create the synthesis prompt
        prompt = self.prompt_template.format(
            query=query,
            domain_responses=formatted_responses
        )
        
        # Generate the synthesized response
        synthesized_content = self.synthesis_model.generate(prompt)
        
        # Create the final response
        mixed_response = {
            "source_domains": list(domain_responses.keys()),
            "content": synthesized_content,
            "synthesis_method": "llm",
            "original_responses": domain_responses
        }
        
        return mixed_response
