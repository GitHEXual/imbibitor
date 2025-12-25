"""Ollama LLM client."""

from typing import Optional, Dict, Any
from .client import OllamaClient


class OllamaLLM:
    """Client for working with Ollama LLM models."""
    
    def __init__(
        self,
        model: str = "devstral-2:123b-cloud",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model: Model name for LLM
            base_url: Base URL of Ollama API
            timeout: Request timeout in seconds
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream the response
            **kwargs: Additional options for the model
            
        Returns:
            Generated text
        """
        options: Dict[str, Any] = {}
        
        if temperature is not None:
            options["temperature"] = temperature
        elif self.temperature is not None:
            options["temperature"] = self.temperature
        
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        
        # Add any additional kwargs to options
        options.update(kwargs)
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=stream,
            options=options if options else None
        )
        
        return response.get("response", "")

