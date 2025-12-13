from aiogram import Router
from aiogram.types import Message

from Bot.keyboards import main_keyboard
from Bot.handlers.menu_state import MAIN_MENU, get_menu, set_menu

router = Router()


async def send_main_menu(message: Message) -> None:
    set_menu(message.chat.id, MAIN_MENU)
    await message.answer("Главное меню. Выберите действие.", reply_markup=main_keyboard)


@router.message(lambda m: m.text == "Написать llm" and get_menu(m.chat.id) == MAIN_MENU)
async def handle_chat_enter(message: Message) -> None:
    from Bot.handlers.chat import send_chat_menu
    await send_chat_menu(message)


@router.message(lambda m: m.text == "Настройки" and get_menu(m.chat.id) == MAIN_MENU)
async def handle_settings_enter(message: Message) -> None:
    from Bot.handlers.settings import send_settings_menu
    await send_settings_menu(message)

