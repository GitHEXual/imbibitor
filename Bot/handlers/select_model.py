import requests
from aiogram import Router
from aiogram.types import Message

from Bot.keyboards import settings_keyboard
from Bot.handlers.menu_state import SELECT_MODEL_MENU, SETTINGS_MENU, get_menu, set_menu, ollama_client, ollama_api

router = Router()


@router.message(lambda m: get_menu(m.chat.id) == SELECT_MODEL_MENU and (m.text or "").strip() != "Старт")
async def handle_select_model_choice(message: Message) -> None:
    text = (message.text or "").strip()
    chat_id = message.chat.id
    
    if text == "Назад":
        from Bot.handlers.settings import send_settings_menu
        await send_settings_menu(message)
        return

    try:
        success = ollama_api.set_model(text)
    except requests.RequestException as exc:
        await message.answer(
            f"Ошибка при выборе модели: {exc}",
            reply_markup=settings_keyboard,
        )
        set_menu(chat_id, SETTINGS_MENU)
        return

    if success:
        ollama_client.setup.default_model = text
        await message.answer(
            f"Модель «{text}» активирована.",
            reply_markup=settings_keyboard,
        )
    else:
        await message.answer(
            "Не удалось активировать модель.",
            reply_markup=settings_keyboard,
        )

    set_menu(chat_id, SETTINGS_MENU)

