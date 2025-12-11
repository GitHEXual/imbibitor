"""Конфигурация для Ollama LLM."""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")

# Настройки для работы с Ollama
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

