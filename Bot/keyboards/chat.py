from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

chat_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Назад")]],
    resize_keyboard=True,
)

