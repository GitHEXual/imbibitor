from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Написать llm")],
        [KeyboardButton(text="Настройки")],
    ],
    resize_keyboard=True,
)

