from aiogram import Router
from aiogram.types import Message

from Bot.keyboards import start_keyboard
from Bot.handlers.menu_state import DEFAULT_MENU, set_menu

router = Router()


async def send_start_menu(message: Message) -> None:
    set_menu(message.chat.id, DEFAULT_MENU)
    await message.answer(
        "Привет! Нажмите кнопку 'Старт', чтобы открыть главное меню.",
        reply_markup=start_keyboard,
    )


@router.message(lambda m: m.text == "/start")
async def handle_start_command(message: Message) -> None:
    await send_start_menu(message)


@router.message(lambda m: m.text == "Старт")
async def handle_start_button(message: Message) -> None:
    from Bot.handlers.main import send_main_menu
    await send_main_menu(message)

