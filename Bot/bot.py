import os
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from Bot.handlers.router import router, setup_handlers


def create_bot(token: str) -> Bot:
    """Создает и возвращает экземпляр бота"""
    return Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


def create_dispatcher() -> Dispatcher:
    """Создает и настраивает диспетчер с FSM storage и роутерами"""
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    
    # Настраиваем все обработчики на главном роутере
    setup_handlers()
    
    # Подключаем главный роутер к диспетчеру
    dp.include_router(router)
    return dp


def get_bot_token() -> str:
    """Получает токен бота из переменных окружения"""
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN not found in .env file")
    return token
