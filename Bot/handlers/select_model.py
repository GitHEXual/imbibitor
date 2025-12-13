import requests
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import settings_keyboard
from Bot.handlers.menu_state import ollama_client, ollama_api
from Bot.handlers.states import MenuStates


def register_handlers(router: Router) -> None:
    """Регистрирует обработчики выбора модели"""
    @router.message(MenuStates.select_model, lambda m: m.text == "Назад")
    async def handle_select_model_back(message: Message, state: FSMContext) -> None:
        from Bot.handlers.settings import send_settings_menu
        await send_settings_menu(message, state)

    @router.message(MenuStates.select_model, lambda m: (m.text or "").strip() != "Старт")
    async def handle_select_model_choice(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        if not text:
            return

        # Получаем список доступных моделей из FSM data
        data = await state.get_data()
        available_models = data.get("available_models", [])
        
        # Проверяем, что выбранная модель есть в списке доступных
        if text not in available_models:
            # Игнорируем неизвестные сообщения - не обрабатываем их как модели
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

