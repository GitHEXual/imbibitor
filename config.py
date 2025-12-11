"""Конфигурация бота - загрузка переменных окружения."""
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# Knowledge Base Path
KNOWLEDGE_BASE_DIR = "knowledge_base"

