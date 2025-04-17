# Models: Architecture and Design Documentation

## Overview

The Models module provides a unified interface for working with various language model providers. It abstracts away the differences between different LLM APIs (OpenAI, Anthropic, HuggingFace, Ollama) and provides a consistent way to generate text and embeddings.

## Architectural Patterns

The models architecture follows several design patterns:

1. **Adapter Pattern**: Each model implementation adapts a specific LLM provider's API to a common interface.
2. **Factory Pattern**: The ModelRegistry serves as a factory for creating model instances.
3. **Strategy Pattern**: Different model implementations represent different strategies for text generation and embedding creation.
4. **Dependency Injection**: Models can be injected into other components (Chains, Mixers, etc.).

## Core Interface

All models implement the abstract `Model` base class:

```python
class Model(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings for the given text."""
        pass
```

## Model Implementations

### OpenAIModel

The `OpenAIModel` integrates with OpenAI's API to provide access to models like GPT-3.5, GPT-4, etc.

**Key Features**:
- Supports both chat-based and completion-based models
- Handles token limits and temperature settings
- Provides access to OpenAI's embedding models
- Gracefully handles API errors

**Use Case**:
When you need high-quality text generation and embeddings with scalable API access.

**Example**:
```python
model = OpenAIModel(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

response = model.generate("Explain quantum computing in simple terms.")
embedding = model.embed("Quantum computing uses quantum bits.")
```

### AnthropicModel

The `AnthropicModel` integrates with Anthropic's API to provide access to Claude models.

**Key Features**:
- Designed for Anthropic's Claude models (Claude 3 Opus, Sonnet, etc.)
- Handles context window limitations
- Optimized for longer, more nuanced conversations
- Falls back to alternative embedding providers if needed

**Use Case**:
When you need models optimized for safety, helpfulness, and harmlessness with long context support.

**Example**:
```python
model = AnthropicModel(
    model_name="claude-3-opus-20240229",
    temperature=0.5,
    max_tokens=2000
)

response = model.generate("Write a detailed analysis of climate change impacts.")
```

### OllamaModel

The `OllamaModel` connects to locally running models via the Ollama server.

**Key Features**:
- Run models locally without API costs
- Access to a wide range of open models (Llama, Mistral, etc.)
- Configurable generation parameters
- Built-in connection testing

**Use Case**:
When you need to run models locally for privacy, cost, or offline access reasons.

**Example**:
```python
model = OllamaModel(
    model_name="llama3",
    base_url="http://localhost:11434",
    temperature=0.8
)

response = model.generate("Suggest three book recommendations for someone interested in astronomy.")
embedding = model.embed("Astronomy is the study of celestial objects.")
```

### HuggingFaceModel

The `HuggingFaceModel` provides access to models from the HuggingFace Hub.

**Key Features**:
- Support for both local model loading and Inference API
- Handles both text generation and embedding creation
- Automatic device management (CPU/GPU)
- Different pooling strategies for embeddings

**Use Case**:
When you need access to specialized models or want to use open-source alternatives.

**Example**:
```python
# Local model usage
model = HuggingFaceModel(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)

# API usage
api_model = HuggingFaceModel(
    model_name="meta-llama/Llama-2-70b-chat-hf",
    use_api=True,
    api_key="hf_..."
)

response = model.generate("Write a poem about artificial intelligence.")
embedding = model.embed("Artificial intelligence is transforming society.")
```

## Integration with Other Components

Models are designed to be used with other components in the system:

### With Chains

```python
from domainfusion import Chain
from domainfusion.models import OpenAIModel, AnthropicModel

chain = Chain(
    models=[
        OpenAIModel(model_name="gpt-4"),
        AnthropicModel(model_name="claude-3-sonnet")
    ],
    prompt_templates={
        "gpt-4": "Analyze this query from a technical perspective: {query}",
        "claude-3-sonnet": "Given this technical analysis: {responses[0]}\nProvide practical applications for: {query}"
    }
)

response = chain.generate("How could quantum computing impact cryptography?")
```

### With Mixers

```python
from domainfusion.mixers import SynthesisMixer
from domainfusion.models import OpenAIModel

synthesis_model = OpenAIModel(model_name="gpt-4")
mixer = SynthesisMixer(synthesis_model=synthesis_model)

responses = {
    "domain1": {"response": "First expert analysis..."},
    "domain2": {"response": "Second expert analysis..."}
}

mixed_response = mixer.mix("Original query", responses)
```

## Best Practices

1. **Graceful Dependency Handling**: Use try-except blocks to handle optional dependencies.
2. **Environment Variable Support**: Allow API keys to be provided via environment variables.
3. **Consistent Parameter Handling**: Maintain consistent parameter names across model implementations.
4. **Proper Error Handling**: Provide informative error messages for common failure modes.
5. **Documentation**: Document model-specific parameters and behaviors.

## Extending with New Model Providers

To add support for a new model provider:

1. Create a new class that extends the `Model` base class
2. Implement the required methods (`name`, `generate`, `embed`)
3. Handle provider-specific authentication and configuration
4. Add proper error handling and dependency management
5. Update the `__init__.py` file to export the new class

Example template:

```python
class NewProviderModel(Model):
    def __init__(self, model_name: str, **kwargs):
        # Initialize model-specific configuration
        pass
        
    @property
    def name(self) -> str:
        # Return model name
        pass
        
    def generate(self, prompt: str, **kwargs) -> str:
        # Generate text using the provider's API
        pass
        
    def embed(self, text: str, **kwargs) -> List[float]:
        # Generate embeddings using the provider's API
        pass
```
