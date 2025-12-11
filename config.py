"""Конфигурация бота - загрузка переменных окружения."""
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# Knowledge Base Path
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")

