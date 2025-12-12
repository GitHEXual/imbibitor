from typing import Optional
from .ollama_setup import OllamaSetup
from .ollama_api import OllamaAPI


class OllamaClient:
    def __init__(self, setup: Optional[OllamaSetup] = None):
        self.setup = setup or OllamaSetup()
        self.api = OllamaAPI(self.setup)

