from .base import Model

# Import model implementations with try-except to handle missing dependencies
try:
    from .openai import OpenAIModel
except ImportError:
    pass

try:
    from .anthropic import AnthropicModel
except ImportError:
    pass

try:
    from .ollama import OllamaModel
except ImportError:
    pass

try:
    from .huggingface import HuggingFaceModel
except ImportError:
    pass

__all__ = [
    "Model",
    "OpenAIModel",
    "AnthropicModel",
    "OllamaModel",
    "HuggingFaceModel"
]
