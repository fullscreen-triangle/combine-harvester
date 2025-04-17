from typing import Dict, List, Optional, Union
import os

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from .base import Model


class HuggingFaceModel(Model):
    """
    Implementation of the Model interface for models hosted on HuggingFace.
    
    This class provides methods to generate text and embeddings using HuggingFace's transformers library.
    """
    
    def __init__(
        self, 
        model_name: str,
        tokenizer_name: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize a HuggingFace model.
        
        Args:
            model_name: HuggingFace model name or path
            tokenizer_name: Optional tokenizer name (defaults to model_name)
            use_api: Whether to use the HuggingFace Inference API instead of local models
            api_key: HuggingFace API key for inference API (defaults to HF_API_KEY environment variable)
            device: Device to use for inference (e.g., "cuda", "cpu")
            temperature: Controls randomness in generation (0 = deterministic, 1 = creative)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the model
        """
        if not HUGGINGFACE_AVAILABLE and not use_api:
            raise ImportError(
                "HuggingFace transformers package is not installed. "
                "Please install it with: pip install transformers torch"
            )
        
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name or model_name
        self._use_api = use_api
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        # Determine device
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        # Set up API key if using API
        if use_api:
            try:
                import requests
                self._api_key = api_key or os.environ.get("HF_API_KEY")
                if not self._api_key:
                    raise ValueError(
                        "HuggingFace API key not provided and not found in HF_API_KEY environment variable"
                    )
            except ImportError:
                raise ImportError(
                    "Requests package is not installed. "
                    "Please install it with: pip install requests"
                )
        else:
            # Load model and tokenizer locally
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            
            # Check if the model is an embedding model or a language model
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name, 
                    **kwargs
                ).to(self._device)
                self._is_language_model = True
            except:
                # Try loading as a generic model (for embeddings)
                self._model = AutoModel.from_pretrained(
                    self._model_name,
                    **kwargs
                ).to(self._device)
                self._is_language_model = False
    
    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the given prompt using HuggingFace models.
        
        Args:
            prompt: The input prompt to generate a response for
            **kwargs: Additional parameters to override the defaults
            
        Returns:
            The generated text response
        """
        if self._use_api:
            return self._generate_api(prompt, **kwargs)
        else:
            return self._generate_local(prompt, **kwargs)
    
    def _generate_api(self, prompt: str, **kwargs) -> str:
        """Generate using the HuggingFace Inference API."""
        import requests
        
        headers = {"Authorization": f"Bearer {self._api_key}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self._model_name}"
        
        # Set up payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self._temperature),
                "max_new_tokens": kwargs.get("max_tokens", self._max_tokens),
                "return_full_text": kwargs.get("return_full_text", False)
            }
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens", "return_full_text"]:
                payload["parameters"][key] = value
        
        # Make request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        response_json = response.json()
        
        # Handle different response formats
        if isinstance(response_json, list) and len(response_json) > 0:
            if "generated_text" in response_json[0]:
                return response_json[0]["generated_text"]
            return response_json[0]
        
        return str(response_json)
    
    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate using a local HuggingFace model."""
        if not self._is_language_model:
            raise ValueError(f"Model {self._model_name} is not a language model and cannot generate text")
        
        # Get parameters
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        
        # Generate
        with torch.no_grad():
            generation_args = {
                "max_length": inputs.input_ids.shape[1] + max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            }
            
            outputs = self._model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_args
            )
        
        # Decode output
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output if needed
        if not kwargs.get("return_full_text", False):
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].lstrip()
        
        return generated_text
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text using HuggingFace models.
        
        Args:
            text: The input text to generate embeddings for
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            A list of floating point numbers representing the text embedding
        """
        if self._use_api:
            return self._embed_api(text, **kwargs)
        else:
            return self._embed_local(text, **kwargs)
    
    def _embed_api(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using the HuggingFace Inference API."""
        import requests
        
        headers = {"Authorization": f"Bearer {self._api_key}"}
        API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self._model_name}"
        
        # Make request
        response = requests.post(API_URL, headers=headers, json={"inputs": text, **kwargs})
        response.raise_for_status()
        
        # Parse response
        embeddings = response.json()
        
        # Handle different response formats
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                # Return the first sequence embedding (average if needed)
                return embeddings[0] if len(embeddings[0]) > 1 else embeddings
            return embeddings
        
        return embeddings
    
    def _embed_local(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using a local HuggingFace model."""
        # Tokenize input
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        
        # Extract embeddings - different models output different formats
        if hasattr(outputs, "last_hidden_state"):
            # Get the [CLS] token embedding or average all token embeddings
            embeddings = outputs.last_hidden_state[0]
            if kwargs.get("pooling", "mean") == "cls":
                embedding = embeddings[0].cpu().numpy().tolist()  # CLS token
            else:
                embedding = embeddings.mean(dim=0).cpu().numpy().tolist()  # Mean pooling
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output[0].cpu().numpy().tolist()
        else:
            # Fallback to mean of last hidden state
            embedding = outputs.hidden_states[-1][0].mean(dim=0).cpu().numpy().tolist()
        
        return embedding
