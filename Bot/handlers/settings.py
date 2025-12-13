import requests
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import settings_keyboard
from Bot.handlers.menu_state import ollama_client, ollama_api
from Bot.handlers.states import MenuStates

router = Router()


async def send_settings_menu(message: Message, state: FSMContext) -> None:
    await state.set_state(MenuStates.settings)
    current_model = ollama_client.setup.default_model
    await message.answer(
        f"Настройки (текущая модель: {current_model}).",
        reply_markup=settings_keyboard,
    )


@router.message(MenuStates.settings, lambda m: (m.text or "").strip() != "Старт")
async def handle_settings_selection(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    
    if text == "Назад":
        from Bot.handlers.main import send_main_menu
        await send_main_menu(message, state)
        return

    if text == "Выбрать модель":
        try:
            models = ollama_api.list_models()
        except requests.RequestException as exc:
            await message.answer(
                f"Ошибка получения списка моделей: {exc}",
                reply_markup=settings_keyboard,
            )
            return

        if not models:
            await message.answer(
                "Список моделей пуст.",
                reply_markup=settings_keyboard,
            )
            return

        from Bot.keyboards import build_select_model_keyboard
        keyboard = build_select_model_keyboard(models)
        await state.set_state(MenuStates.select_model)
        await message.answer("Выберите модель.", reply_markup=keyboard)
        return

    if text == "Ollama sign in":
        try:
            result = ollama_api.auth()
        except Exception as exc:  # pragma: no cover - subprocess output depends on env
            await message.answer(
                f"Не удалось выполнить вход: {exc}",
                reply_markup=settings_keyboard,
            )
        else:
            await message.answer(
                result or "Вход выполнен.",
                reply_markup=settings_keyboard,
            )
        return

    if text == "Ollama sign out":
        try:
            result = ollama_api.logout()
        except Exception as exc:  # pragma: no cover - subprocess output depends on env
            await message.answer(
                f"Не удалось выйти: {exc}",
                reply_markup=settings_keyboard,
            )
        else:
            await message.answer(
                result or "Выход выполнен.",
                reply_markup=settings_keyboard,
            )
        return

    await message.answer(
        "Нажмите одну из кнопок настроек или 'Назад'.",
        reply_markup=settings_keyboard,
    )

