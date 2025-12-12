class OllamaSetup:
    def __init__(self, host: str = "localhost", port: int = 11434, default_model: str = "llama2"):
        self.host = host
        self.port = port
        self.default_model = default_model
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

