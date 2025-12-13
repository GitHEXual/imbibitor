from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

settings_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Выбрать модель")],
        [
            KeyboardButton(text="Ollama sign in"),
            KeyboardButton(text="Ollama sign out"),
        ],
        [KeyboardButton(text="Назад")],
    ],
    resize_keyboard=True,
)

