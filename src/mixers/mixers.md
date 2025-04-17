# Mixers: Architecture and Design Documentation

## Overview

Mixers are components that combine responses from multiple domain expert models into a unified, integrated response. They form a critical part of the system's ability to integrate knowledge across different domains.

## Architectural Patterns

The mixer architecture follows several design patterns:

1. **Strategy Pattern**: Different mixing strategies are encapsulated in separate classes with a common interface.
2. **Composition**: Mixers can be composed with other components (Router, ModelRegistry) to create a complete ensemble system.
3. **Dependency Injection**: Mixers are injected into the Ensemble, making the system modular and testable.
4. **Decorator Pattern**: Each mixer enriches the response with metadata about the mixing process.

## Core Interface

All mixers implement the abstract `Mixer` base class:

```python
class Mixer(ABC):
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
```

## Mixer Types

### WeightedMixer

The `WeightedMixer` combines responses with domain-specific weights when some domains are more reliable or relevant than others.

**Key Features**:
- Takes domain weights as input
- Normalizes weights to create a proportional contribution
- Includes weighted individual responses in the output
- Uses the highest weighted domain as the primary source for the combined response

**Use Case**:
When you have prior knowledge about the reliability or relevance of different domains for specific types of queries.

**Example**:
```python
weights = {
    "medicine": 0.7,
    "nutrition": 0.2,
    "fitness": 0.1
}
mixer = WeightedMixer(domain_weights=weights)
```

### VotingMixer

The `VotingMixer` uses a consensus approach to select values when domains might provide conflicting information.

**Key Features**:
- Identifies common fields across all domain responses
- For each field, tallies "votes" from different domains
- Selects the value with the most votes for the final response
- Includes voting results in the output for transparency
- Supports a configurable threshold for consensus

**Use Case**:
When domains might provide conflicting information and you want to rely on consensus.

**Example**:
```python
# Accept values that at least 60% of domains agree on
mixer = VotingMixer(threshold=0.6)
```

### SynthesisMixer

The `SynthesisMixer` uses an LLM to synthesize responses from multiple domains into a coherent narrative.

**Key Features**:
- Uses a synthesis model (typically an LLM) to create integrated responses
- Supports customizable prompt templates for synthesis
- Preserves original domain responses for reference
- Handles complex integration where simple field-by-field combination is insufficient

**Use Case**:
For complex queries requiring sophisticated integration of knowledge across domains.

**Example**:
```python
mixer = SynthesisMixer(
    synthesis_model=llm_model,
    prompt_template="""
    Integrate these expert responses into a coherent answer:
    {domain_responses}
    
    Original query: {query}
    """
)
```

## Integration with Ensemble System

Mixers are designed to be used with the Ensemble system:

```python
ensemble = Ensemble(
    router=router,
    registry=model_registry,
    mixer=mixer,
    default_domain="general"
)

response = ensemble.process("How does nutrition affect recovery from sprint training?")
```

## Implementing Custom Mixers

To implement a new mixing strategy:

1. Create a new class that extends the `Mixer` base class
2. Implement the `mix` method to combine responses according to your strategy
3. Handle edge cases (empty or single responses)
4. Include metadata about the mixing process in the response

Example template:

```python
class CustomMixer(Mixer):
    def __init__(self, custom_param):
        self.custom_param = custom_param
        
    def mix(self, query: str, domain_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        # Handle edge cases
        if not domain_responses:
            return {"error": "No domain responses to mix"}
        
        if len(domain_responses) == 1:
            domain_name = list(domain_responses.keys())[0]
            return {
                "source_domain": domain_name,
                **domain_responses[domain_name]
            }
            
        # Implement your custom mixing logic
        mixed_response = {
            "source_domains": list(domain_responses.keys()),
            # Add your custom mixing result
        }
        
        return mixed_response
```

## Best Practices

1. **Preserve Source Information**: Always include source domains in the mixed response
2. **Handle Edge Cases**: Account for empty responses or single-domain responses
3. **Provide Transparency**: Include metadata about how responses were combined
4. **Maintain Traceability**: Allow access to original responses when needed
5. **Support Multiple Response Formats**: Be flexible with different response structures
