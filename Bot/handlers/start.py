from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.keyboards import start_keyboard
from Bot.handlers.states import MenuStates


async def send_start_menu(message: Message, state: FSMContext) -> None:
    await state.set_state(MenuStates.start)
    await message.answer(
        "Привет! Нажмите кнопку 'Старт', чтобы открыть главное меню.",
        reply_markup=start_keyboard,
    )


def register_handlers(router: Router) -> None:
    """Регистрирует обработчики стартового меню"""
    @router.message(lambda m: m.text == "/start")
    async def handle_start_command(message: Message, state: FSMContext) -> None:
        await send_start_menu(message, state)

