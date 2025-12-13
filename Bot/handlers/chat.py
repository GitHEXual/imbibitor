import requests
from aiogram import Router
from aiogram.types import Message

from Bot.keyboards import chat_keyboard, main_keyboard
from Bot.handlers.menu_state import CHAT_MENU, MAIN_MENU, get_menu, set_menu
from OllamaClient.ollama_client import OllamaClient

router = Router()
ollama_client = OllamaClient()
ollama_api = ollama_client.api


async def send_chat_menu(message: Message) -> None:
    set_menu(message.chat.id, CHAT_MENU)
    await message.answer(
        "В режиме чата с LLM. Отправьте сообщение или нажмите 'Назад'.",
        reply_markup=chat_keyboard,
    )


async def handle_chat_message(message: Message) -> None:
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустое сообщение не отправляется.", reply_markup=chat_keyboard)
        return

    try:
        response = ollama_api.generate(text)
    except requests.RequestException as exc:
        await message.answer(
            f"Не удалось отправить запрос: {exc}",
            reply_markup=main_keyboard,
        )
    else:
        await message.answer(response, reply_markup=main_keyboard)
    finally:
        set_menu(chat_id, MAIN_MENU)


@router.message(lambda m: get_menu(m.chat.id) == CHAT_MENU)
async def handle_chat_menu(message: Message) -> None:
    text = (message.text or "").strip()
    
    if text == "Назад":
        from Bot.handlers.main import send_main_menu
        await send_main_menu(message)
        return

    await handle_chat_message(message)

