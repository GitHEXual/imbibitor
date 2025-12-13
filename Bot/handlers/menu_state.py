from typing import Dict

DEFAULT_MENU = "start"
MAIN_MENU = "main"
SETTINGS_MENU = "settings"
SELECT_MODEL_MENU = "select_model"
CHAT_MENU = "chat"

user_menu_state: Dict[int, str] = {}


def get_menu(chat_id: int) -> str:
    return user_menu_state.get(chat_id, DEFAULT_MENU)


def set_menu(chat_id: int, menu: str) -> None:
    user_menu_state[chat_id] = menu

