from typing import Iterable

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def build_select_model_keyboard(models: Iterable[str]) -> ReplyKeyboardMarkup:
    rows: list[list[KeyboardButton]] = []
    chunk: list[KeyboardButton] = []

    for model in models:
        chunk.append(KeyboardButton(text=model))
        if len(chunk) == 2:
            rows.append(chunk)
            chunk = []

    if chunk:
        rows.append(chunk)

    rows.append([KeyboardButton(text="Назад")])

    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)

