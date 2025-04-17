from typing import Any, Dict, List, Optional, Callable, Union

from .base import DomainExpert, ContextManager, OutputProcessor
from .registry import ModelRegistry


class DefaultContextManager(ContextManager):
    """
    Default implementation of context manager that maintains a simple dictionary context.
    """
    
    def __init__(self):
        """Initialize an empty context."""
        self.context: Dict[str, Any] = {
            "history": []
        }
    
    def update(self, domain_name: str, response: Dict[str, Any]) -> None:
        """
        Update the context with a new domain response.
        
        Args:
            domain_name: Name of the domain that produced the response
            response: The response from the domain expert
        """
        self.context[domain_name] = response
        self.context["history"].append({
            "domain": domain_name,
            "response": response
        })
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context.
        
        Returns:
            The current context as a dictionary
        """
        return self.context


class DefaultOutputProcessor(OutputProcessor):
    """
    Default implementation of output processor that returns the final response.
    """
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the final context to generate the output.
        
        Args:
            context: The final context after all processing
            
        Returns:
            The final processed output
        """
        # Return the last response from history as final output
        if "history" in context and context["history"]:
            final_step = context["history"][-1]
            return {
                "final_domain": final_step["domain"],
                "response": final_step["response"],
                "full_context": context
            }
        
        return {"response": "No processing completed", "full_context": context}


class Chain:
    """
    Sequential chain that passes queries through multiple domain experts in sequence.
    
    Each expert in the chain builds on the responses of previous experts, allowing for
    progressive refinement and analysis across multiple domains.
    """

    def __init__(
        self,
        domain_sequence: List[str],
        registry: ModelRegistry,
        context_manager: Optional[ContextManager] = None,
        output_processor: Optional[OutputProcessor] = None,
        prompt_templates: Optional[Dict[str, Callable[[str, Dict[str, Any]], str]]] = None
    ):
        """
        Initialize a Chain.
        
        Args:
            domain_sequence: Ordered list of domain names to process the query through
            registry: Registry containing domain expert models
            context_manager: Optional custom context manager (uses DefaultContextManager if None)
            output_processor: Optional custom output processor (uses DefaultOutputProcessor if None)
            prompt_templates: Optional dict mapping domain names to functions that format inputs
        """
        self.domain_sequence = domain_sequence
        self.registry = registry
        self.context_manager = context_manager or DefaultContextManager()
        self.output_processor = output_processor or DefaultOutputProcessor()
        self.prompt_templates = prompt_templates or {}
    
    def _format_query(self, domain: str, query: str, context: Dict[str, Any]) -> str:
        """
        Format a query for a specific domain using its prompt template.
        
        Args:
            domain: The domain name
            query: The original query
            context: The current context
            
        Returns:
            Formatted query for the domain
        """
        if domain in self.prompt_templates:
            return self.prompt_templates[domain](query, context)
        return query
    
    def process(self, query: str, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the chain of domain experts.
        
        Args:
            query: The input query to process
            initial_context: Optional initial context
            
        Returns:
            Final response after processing through the chain
        """
        # Initialize context
        if initial_context:
            for key, value in initial_context.items():
                self.context_manager.context[key] = value
        
        current_query = query
        
        # Process through each domain in sequence
        for domain_name in self.domain_sequence:
            expert = self.registry.get(domain_name)
            
            if not expert:
                raise ValueError(f"Domain expert '{domain_name}' not found in registry")
            
            # Format query using domain-specific template if available
            formatted_query = self._format_query(
                domain_name, 
                current_query, 
                self.context_manager.get_context()
            )
            
            # Process query with the current expert
            response = expert.process(formatted_query, self.context_manager.get_context())
            
            # Update context with expert's response
            self.context_manager.update(domain_name, response)
            
            # Update query for next step if response contains a refined query
            if "refined_query" in response:
                current_query = response["refined_query"]
        
        # Process final output
        return self.output_processor.process(self.context_manager.get_context())
