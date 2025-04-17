# DomainFusion Core Module

The core module contains the fundamental components of the DomainFusion framework, providing abstractions for domain experts, routing, chaining, and response mixing.

## Overview

This module implements the following architectural patterns described in the whitepaper:

1. **Router-Based Ensembles** - Direct queries to the most appropriate domain expert models
2. **Sequential Chaining** - Pass queries through multiple domain experts in sequence
3. **Mixers** - Combine responses from multiple domain experts

## Files and Components

### base.py

Contains abstract base classes that define the interfaces for all major components:

- `DomainExpert` - Base class for domain-specific expert models
- `Router` - Base class for routers that direct queries to appropriate domain experts
- `Mixer` - Base class for mixers that combine responses from multiple domain experts
- `ContextManager` - Base class for managing context in sequential chains
- `OutputProcessor` - Base class for processing the final output of a chain or ensemble

### registry.py

Provides the `ModelRegistry` class for managing domain expert models:

- Register domain experts by name
- Lazy instantiation of expert models
- Retrieve models by domain name
- List available domains
- Remove models from the registry

### ensemble.py

Implements the `Ensemble` class for router-based integration:

- Uses a router to determine which domain expert(s) should handle a query
- Dispatches the query to the selected expert(s)
- Optionally combines responses from multiple experts using a mixer
- Provides fallback to a default domain if routing fails

### chain.py

Implements the `Chain` class for sequential chaining:

- Passes queries through multiple domain experts in sequence
- Maintains context between steps in the chain
- Allows for custom context management and output processing
- Supports domain-specific prompt templates

Also provides default implementations:

- `DefaultContextManager` - Simple dictionary-based context management
- `DefaultOutputProcessor` - Basic output processing that returns the final response

### mixer.py

Provides implementations of the `Mixer` interface for combining responses:

- `DefaultMixer` - Combines responses by integrating unique contributions from each domain
- `WeightedMixer` - Applies domain-specific weights to combine responses
- `VotingMixer` - Uses voting mechanisms to select the best response based on consensus

## Usage Examples

### Creating a Router-Based Ensemble

```python
from domainfusion.core import Ensemble, ModelRegistry
from domainfusion.routers import EmbeddingRouter
from domainfusion.models import GPTExpert, LlamaExpert

# Create a registry and register domain experts
registry = ModelRegistry()
registry.register("medicine", GPTExpert("medicine"))
registry.register("law", LlamaExpert("law"))

# Create a router
router = EmbeddingRouter()

# Create an ensemble
ensemble = Ensemble(router, registry)

# Process a query
response = ensemble.process("What are the legal implications of medical malpractice?")
```

### Creating a Sequential Chain

```python
from domainfusion.core import Chain, ModelRegistry
from domainfusion.models import GPTExpert

# Create a registry and register domain experts
registry = ModelRegistry()
registry.register("research", GPTExpert("research"))
registry.register("analysis", GPTExpert("analysis"))
registry.register("summary", GPTExpert("summary"))

# Create a chain
chain = Chain(
    domain_sequence=["research", "analysis", "summary"],
    registry=registry
)

# Process a query
response = chain.process("What are the latest developments in quantum computing?")
```

### Using a Mixer

```python
from domainfusion.core import Ensemble, ModelRegistry, WeightedMixer
from domainfusion.routers import KeywordRouter

# Create a registry with domain experts
registry = ModelRegistry()
registry.register("climate", ClimateExpert())
registry.register("economics", EconomicsExpert())
registry.register("policy", PolicyExpert())

# Create a weighted mixer
mixer = WeightedMixer({
    "climate": 0.5,    # Higher weight for climate domain
    "economics": 0.3,  # Medium weight for economics
    "policy": 0.2      # Lower weight for policy
})

# Create a router that can return multiple domains
router = KeywordRouter(multi_domain=True)

# Create an ensemble with the mixer
ensemble = Ensemble(router, registry, mixer=mixer)

# Process a query
response = ensemble.process("How will carbon taxes affect economic growth?")
```

## Extension Points

The core module is designed to be extended in various ways:

1. Create custom domain experts by implementing the `DomainExpert` interface
2. Create custom routers by implementing the `Router` interface
3. Create custom mixers by implementing the `Mixer` interface
4. Create custom context managers by implementing the `ContextManager` interface
5. Create custom output processors by implementing the `OutputProcessor` interface

These extension points allow for flexibility in implementing the architectural patterns described in the whitepaper.
