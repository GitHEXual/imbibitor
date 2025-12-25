"""Base Ollama HTTP client."""

import requests
import json
from typing import Optional, Dict, Any
import time


class OllamaClient:
    """Base HTTP client for Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL of Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _post(
        self,
        endpoint: str,
        json_data: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Make POST request to Ollama API with retry logic.
        
        Args:
            endpoint: API endpoint (without base URL)
            json_data: JSON payload
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            
        Returns:
            Response JSON data
            
        Raises:
            requests.RequestException: If request fails after all retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=json_data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                # Don't retry on client errors (4xx), only on server errors (5xx)
                if e.response.status_code < 500:
                    raise RuntimeError(f"Ollama API request failed: {e}")
                
                # Retry on server errors
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Ollama API request failed after {max_retries} attempts: {e}")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Ollama API request failed after {max_retries} attempts: {e}")
        
        raise RuntimeError(f"Ollama API request failed after {max_retries} attempts")
    
    def check_model(self, model_name: str) -> bool:
        """
        Check if model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name", "").startswith(model_name) for m in models)
        except Exception:
            return False
    
    def generate_embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> list[float]:
        """
        Generate embeddings using Ollama API.
        
        Args:
            model: Model name (e.g., "nomic-embed-text")
            prompt: Text to embed
            options: Additional options for the model
            
        Returns:
            List of embedding values
        """
        payload = {
            "model": model,
            "prompt": prompt
        }
        if options:
            payload["options"] = options
        
        response = self._post("/api/embeddings", payload)
        return response.get("embedding", [])
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama API.
        
        Args:
            model: Model name (e.g., "devstral-2:123b-cloud")
            prompt: Prompt text
            stream: Whether to stream the response
            options: Additional options (temperature, max_tokens, etc.)
            
        Returns:
            Response dictionary with generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        if options:
            payload["options"] = options
        
        if stream:
            # For streaming, we'll return the full response
            url = f"{self.base_url}/api/generate"
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = line.decode('utf-8')
                        data = json.loads(chunk)
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except Exception:
                        continue
            
            return {"response": full_response, "done": True}
        else:
            return self._post("/api/generate", payload)

