import requests
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import chat_keyboard, main_keyboard
from Bot.handlers.menu_state import ollama_api
from Bot.handlers.states import MenuStates


async def send_chat_menu(message: Message, state: FSMContext) -> None:
    await state.set_state(MenuStates.chat)
    await message.answer(
        "В режиме чата с LLM. Отправьте сообщение или нажмите 'Назад'.",
        reply_markup=chat_keyboard,
    )


async def handle_chat_message(message: Message, state: FSMContext) -> None:
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
        from Bot.handlers.main import send_main_menu
        await send_main_menu(message, state)


def register_handlers(router: Router) -> None:
    """Регистрирует обработчики чата"""
    @router.message(MenuStates.chat, lambda m: (m.text or "").strip() != "Старт")
    async def handle_chat_menu(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        
        if text == "Назад":
            from Bot.handlers.main import send_main_menu
            await send_main_menu(message, state)
            return

        await handle_chat_message(message, state)

