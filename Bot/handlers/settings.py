import requests
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import settings_keyboard
from Bot.handlers.menu_state import ollama_client, ollama_api
from Bot.handlers.states import MenuStates


async def send_settings_menu(message: Message, state: FSMContext) -> None:
    await state.set_state(MenuStates.settings)
    current_model = ollama_client.setup.default_model
    await message.answer(
        f"Настройки (текущая модель: {current_model}).",
        reply_markup=settings_keyboard,
    )


def register_handlers(router: Router) -> None:
    """Регистрирует обработчики настроек"""
    @router.message(MenuStates.settings, lambda m: m.text == "Назад")
    async def handle_settings_back(message: Message, state: FSMContext) -> None:
        from Bot.handlers.main import send_main_menu
        await send_main_menu(message, state)

    @router.message(MenuStates.settings, lambda m: m.text == "Выбрать модель")
    async def handle_select_model_button(message: Message, state: FSMContext) -> None:
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

        # Сохраняем список моделей в FSM data для проверки
        await state.update_data(available_models=models)
        
        from Bot.keyboards import build_select_model_keyboard
        keyboard = build_select_model_keyboard(models)
        await state.set_state(MenuStates.select_model)
        await message.answer("Выберите модель.", reply_markup=keyboard)

    @router.message(MenuStates.settings, lambda m: m.text == "Ollama sign in")
    async def handle_ollama_signin(message: Message, state: FSMContext) -> None:
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

    @router.message(MenuStates.settings, lambda m: m.text == "Ollama sign out")
    async def handle_ollama_signout(message: Message, state: FSMContext) -> None:
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

