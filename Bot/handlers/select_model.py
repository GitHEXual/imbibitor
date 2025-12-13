import requests
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import settings_keyboard
from Bot.handlers.menu_state import ollama_client, ollama_api
from Bot.handlers.states import MenuStates

router = Router()


@router.message(MenuStates.select_model, lambda m: (m.text or "").strip() != "Старт")
async def handle_select_model_choice(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    
    if text == "Назад":
        from Bot.handlers.settings import send_settings_menu
        await send_settings_menu(message, state)
        return

    try:
        success = ollama_api.set_model(text)
    except requests.RequestException as exc:
        await message.answer(
            f"Ошибка при выборе модели: {exc}",
            reply_markup=settings_keyboard,
        )
        await state.set_state(MenuStates.settings)
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

    await state.set_state(MenuStates.settings)

