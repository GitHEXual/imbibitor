import requests
import subprocess
from typing import Optional
from ollama_setup import OllamaSetup


class OllamaAPI:
    def __init__(self, setup: OllamaSetup):
        self.setup = setup
    
    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.setup.default_model
        response = requests.post(
            f"{self.setup.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def list_models(self) -> list:
        response = requests.get(f"{self.setup.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    
    def set_model(self, model: str) -> bool:
        response = requests.post(
            f"{self.setup.base_url}/api/generate",
            json={"model": model, "prompt": "test"}
        )
        return response.status_code == 200
    
    def auth(self) -> str:
        result = subprocess.run(
            ["ollama", "signin"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    
    def logout(self) -> str:
        result = subprocess.run(
            ["ollama", "signout"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

