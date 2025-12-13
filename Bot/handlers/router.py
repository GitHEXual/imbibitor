from __future__ import annotations

from typing import Dict

import requests
from aiogram import Router
from aiogram.types import Message

from Bot.keyboards import (
    build_select_model_keyboard,
    chat_keyboard,
    main_keyboard,
    settings_keyboard,
    start_keyboard,
)
from OllamaClient.ollama_client import OllamaClient

router = Router()
ollama_client = OllamaClient()
ollama_api = ollama_client.api

user_menu_state: Dict[int, str] = {}

DEFAULT_MENU = "start"
MAIN_MENU = "main"
SETTINGS_MENU = "settings"
SELECT_MODEL_MENU = "select_model"
CHAT_MENU = "chat"


def _get_menu(chat_id: int) -> str:
    return user_menu_state.get(chat_id, DEFAULT_MENU)


def _set_menu(chat_id: int, menu: str) -> None:
    user_menu_state[chat_id] = menu


async def _send_start_menu(message: Message) -> None:
    _set_menu(message.chat.id, DEFAULT_MENU)
    await message.answer(
        "Привет! Нажмите кнопку 'Старт', чтобы открыть главное меню.",
        reply_markup=start_keyboard,
    )


async def _send_main_menu(message: Message) -> None:
    _set_menu(message.chat.id, MAIN_MENU)
    await message.answer("Главное меню. Выберите действие.", reply_markup=main_keyboard)


async def _send_settings_menu(message: Message) -> None:
    _set_menu(message.chat.id, SETTINGS_MENU)
    current_model = ollama_client.setup.default_model
    await message.answer(
        f"Настройки (текущая модель: {current_model}).",
        reply_markup=settings_keyboard,
    )


async def _send_chat_menu(message: Message) -> None:
    _set_menu(message.chat.id, CHAT_MENU)
    await message.answer(
        "В режиме чата с LLM. Отправьте сообщение или нажмите 'Назад'.",
        reply_markup=chat_keyboard,
    )


async def _handle_chat_message(message: Message) -> None:
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустое сообщение не отправляется.", reply_markup=chat_keyboard)
        return

    try:
        response = ollama_api.generate(text)
    except requests.RequestException as exc:
        await message.answer(
            f"Не удалось отправить запрос: {exc}",
            reply_markup=main_keyboard,
        )
    else:
        await message.answer(response, reply_markup=main_keyboard)
    finally:
        _set_menu(chat_id, MAIN_MENU)


async def _handle_settings_selection(message: Message, text: str) -> None:
    chat_id = message.chat.id
    if text == "Назад":
        await _send_main_menu(message)
        return

    if text == "Выбрать модель":
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

        keyboard = build_select_model_keyboard(models)
        _set_menu(chat_id, SELECT_MODEL_MENU)
        await message.answer("Выберите модель.", reply_markup=keyboard)
        return

    if text == "Ollama sign in":
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
        return

    if text == "Ollama sign out":
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
        return

    await message.answer(
        "Нажмите одну из кнопок настроек или 'Назад'.",
        reply_markup=settings_keyboard,
    )


async def _handle_select_model_choice(message: Message, text: str) -> None:
    chat_id = message.chat.id
    if text == "Назад":
        await _send_settings_menu(message)
        return

    try:
        success = ollama_api.set_model(text)
    except requests.RequestException as exc:
        await message.answer(
            f"Ошибка при выборе модели: {exc}",
            reply_markup=settings_keyboard,
        )
        _set_menu(chat_id, SETTINGS_MENU)
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

    _set_menu(chat_id, SETTINGS_MENU)


@router.message()
async def handle_message(message: Message) -> None:
    text = (message.text or "").strip()
    chat_id = message.chat.id

    if not text:
        return

    current_menu = _get_menu(chat_id)

    if text == "/start":
        await _send_start_menu(message)
        return

    if text == "Старт":
        await _send_main_menu(message)
        return

    if current_menu == CHAT_MENU:
        if text == "Назад":
            await _send_main_menu(message)
            return

        await _handle_chat_message(message)
        return

    if text == "Написать llm":
        await _send_chat_menu(message)
        return

    if text == "Настройки":
        await _send_settings_menu(message)
        return

    if current_menu == SETTINGS_MENU:
        await _handle_settings_selection(message, text)
        return

    if current_menu == SELECT_MODEL_MENU:
        await _handle_select_model_choice(message, text)
        return

    await _send_start_menu(message)

