from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import main_keyboard
from Bot.handlers.states import MenuStates


async def send_main_menu(message: Message, state: FSMContext) -> None:
    await state.set_state(MenuStates.main)
    await message.answer("Главное меню. Выберите действие.", reply_markup=main_keyboard)


def register_handlers(router: Router) -> None:
    """Регистрирует обработчики главного меню"""
    @router.message(MenuStates.main, lambda m: m.text == "Написать llm")
    async def handle_chat_enter(message: Message, state: FSMContext) -> None:
        from Bot.handlers.chat import send_chat_menu
        await send_chat_menu(message, state)

    @router.message(MenuStates.main, lambda m: m.text == "Настройки")
    async def handle_settings_enter(message: Message, state: FSMContext) -> None:
        from Bot.handlers.settings import send_settings_menu
        await send_settings_menu(message, state)

