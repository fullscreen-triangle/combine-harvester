from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core import DomainExpert


class DataGenerator(ABC):
    """
    Abstract base class for generating training data for knowledge distillation.
    """
    
    @abstractmethod
    def generate(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Generate training examples for the specified domains.
        
        Args:
            domains: List of domain names to generate examples for
            
        Returns:
            List of training examples
        """
        pass


class SyntheticDataGenerator(DataGenerator):
    """
    Generates synthetic training data using an LLM to create diverse examples.
    
    This generator creates examples that span multiple domains, testing both
    domain-specific knowledge and cross-domain integration.
    """
    
    def __init__(
        self,
        generation_model: DomainExpert,
        domains: Dict[str, str],
        num_examples: int = 100,
        cross_domain_ratio: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize a SyntheticDataGenerator.
        
        Args:
            generation_model: The model to use for generating examples
            domains: Dictionary mapping domain names to their descriptions
            num_examples: Number of examples to generate
            cross_domain_ratio: Proportion of examples that should span multiple domains
            system_prompt: Optional system prompt to use for generation
        """
        self.generation_model = generation_model
        self.domains = domains
        self.num_examples = num_examples
        self.cross_domain_ratio = cross_domain_ratio
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """
        Create the default system prompt for data generation.
        
        Returns:
            Default system prompt
        """
        return """
        Generate {num_examples} diverse questions that would require expertise in the following domains to answer effectively:
        
        Domains:
        {domain_descriptions}
        
        For each question:
        1. Ensure it requires integration of knowledge from at least {min_domains} domains
        2. Make it specific and detailed enough to test deep domain knowledge
        3. Focus on practical applications rather than purely theoretical concepts
        4. Vary the primary domain focus across the questions
        
        Format each question as a standalone query that a practitioner might ask.
        Return the results as a JSON array where each item has the format:
        {{
            "query": "The question text",
            "domains": ["domain1", "domain2"],  // List of domains the question spans
            "primary_domain": "domain1"  // The primary domain if applicable
        }}
        """
    
    def generate(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Generate synthetic training examples.
        
        Args:
            domains: List of domain names to generate examples for
            
        Returns:
            List of training examples
        """
        # Filter domain descriptions to only include requested domains
        filtered_domains = {k: v for k, v in self.domains.items() if k in domains}
        
        if not filtered_domains:
            raise ValueError(f"None of the requested domains {domains} were found in the configured domains")
        
        # Calculate the number of cross-domain examples
        num_cross_domain = int(self.num_examples * self.cross_domain_ratio)
        num_single_domain = self.num_examples - num_cross_domain
        
        # Create domain descriptions for the prompt
        domain_descriptions = "\n".join([f"{i+1}. {domain}: {desc}" for i, (domain, desc) in enumerate(filtered_domains.items())])
        
        # Format the prompt for cross-domain examples
        cross_domain_prompt = self.system_prompt.format(
            num_examples=num_cross_domain,
            domain_descriptions=domain_descriptions,
            min_domains=min(2, len(filtered_domains))
        )
        
        # Format the prompt for single-domain examples
        single_domain_prompt = self.system_prompt.format(
            num_examples=num_single_domain,
            domain_descriptions=domain_descriptions,
            min_domains=1
        )
        
        # Generate examples
        cross_domain_response = self.generation_model.process(cross_domain_prompt)
        single_domain_response = self.generation_model.process(single_domain_prompt)
        
        # Extract examples from model responses
        # This depends on the actual response format from the model
        # Assume the model returns a JSON array in the 'content' field
        
        # In a real implementation, you would parse the JSON from the responses
        # Here we're using a placeholder implementation
        cross_domain_examples = cross_domain_response.get("examples", [])
        single_domain_examples = single_domain_response.get("examples", [])
        
        return cross_domain_examples + single_domain_examples


class AdversarialDataGenerator(DataGenerator):
    """
    Generates adversarial training data that challenges domain boundaries.
    
    This generator creates examples that appear to belong to one domain but 
    actually require expertise from another domain to answer correctly.
    """
    
    def __init__(
        self,
        generation_model: DomainExpert,
        domains: Dict[str, str],
        num_examples: int = 50,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize an AdversarialDataGenerator.
        
        Args:
            generation_model: The model to use for generating examples
            domains: Dictionary mapping domain names to their descriptions
            num_examples: Number of examples to generate
            system_prompt: Optional system prompt to use for generation
        """
        self.generation_model = generation_model
        self.domains = domains
        self.num_examples = num_examples
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """
        Create the default system prompt for adversarial data generation.
        
        Returns:
            Default system prompt
        """
        return """
        Generate {num_examples} questions that appear to belong to one domain but actually require expertise from another domain to answer correctly.
        
        Domains:
        {domain_descriptions}
        
        For each question:
        1. Make it initially appear to be primarily about the first domain
        2. Ensure that a complete answer actually requires significant knowledge from the second domain
        3. Design the question so that answering from only the first domain's perspective would lead to an incomplete or potentially misleading response
        
        Format each question as a JSON object with the following structure:
        {{
            "query": "The question text",
            "apparent_domain": "domain1",  // The domain it appears to belong to
            "actual_domains": ["domain1", "domain2"],  // All domains needed for a correct answer
            "primary_domain": "domain2"  // The domain that's actually most important
        }}
        """
    
    def generate(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Generate adversarial training examples.
        
        Args:
            domains: List of domain names to generate examples for
            
        Returns:
            List of training examples
        """
        if len(domains) < 2:
            raise ValueError("Adversarial data generation requires at least 2 domains")
        
        # Filter domain descriptions to only include requested domains
        filtered_domains = {k: v for k, v in self.domains.items() if k in domains}
        
        # Create domain pairings for adversarial examples
        domain_pairs = []
        domain_list = list(filtered_domains.keys())
        for i in range(len(domain_list)):
            for j in range(len(domain_list)):
                if i != j:
                    domain_pairs.append((domain_list[i], domain_list[j]))
        
        # Create domain descriptions for the prompt
        domain_descriptions = "\n".join([f"{i+1}. {domain}: {desc}" for i, (domain, desc) in enumerate(filtered_domains.items())])
        
        all_examples = []
        
        # Generate examples for each domain pair
        examples_per_pair = max(1, self.num_examples // len(domain_pairs))
        
        for apparent_domain, actual_domain in domain_pairs:
            # Format the prompt for this domain pair
            prompt = self.system_prompt.format(
                num_examples=examples_per_pair,
                domain_descriptions=f"1. {apparent_domain}: {filtered_domains[apparent_domain]}\n2. {actual_domain}: {filtered_domains[actual_domain]}"
            )
            
            # Generate examples
            response = self.generation_model.process(prompt)
            
            # Extract examples from model response
            # This depends on the actual response format from the model
            examples = response.get("examples", [])
            all_examples.extend(examples)
        
        return all_examples[:self.num_examples]


class CuratedDataGenerator(DataGenerator):
    """
    Uses a pre-defined set of expert-curated examples.
    
    This generator is useful when you have a high-quality dataset of examples
    created or verified by domain experts.
    """
    
    def __init__(self, examples: List[Dict[str, Any]]):
        """
        Initialize a CuratedDataGenerator.
        
        Args:
            examples: List of pre-defined examples
        """
        self.examples = examples
    
    def generate(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Return the pre-defined examples that match the requested domains.
        
        Args:
            domains: List of domain names to filter examples for
            
        Returns:
            List of matching examples
        """
        # Filter examples to only include those relevant to the requested domains
        filtered_examples = []
        
        for example in self.examples:
            # Check if the example spans any of the requested domains
            example_domains = example.get("domains", [])
            if any(domain in domains for domain in example_domains):
                filtered_examples.append(example)
        
        return filtered_examples
