from typing import Dict
from OllamaClient.ollama_client import OllamaClient

DEFAULT_MENU = "start"
MAIN_MENU = "main"
SETTINGS_MENU = "settings"
SELECT_MODEL_MENU = "select_model"
CHAT_MENU = "chat"

user_menu_state: Dict[int, str] = {}

# Общий экземпляр OllamaClient для всех обработчиков
ollama_client = OllamaClient()
ollama_api = ollama_client.api


def get_menu(chat_id: int) -> str:
    return user_menu_state.get(chat_id, DEFAULT_MENU)


def set_menu(chat_id: int, menu: str) -> None:
    user_menu_state[chat_id] = menu

