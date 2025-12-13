from aiogram import Router
from aiogram.types import Message

from Bot.handlers import chat, main, select_model, settings, start

router = Router()

# Подключаем все роутеры обработчиков
router.include_router(start.router)
router.include_router(main.router)
router.include_router(chat.router)
router.include_router(settings.router)
router.include_router(select_model.router)


# Обработчик по умолчанию для неизвестных сообщений
@router.message()
async def handle_default(message: Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    
    # Если сообщение не было обработано другими обработчиками, возвращаем в стартовое меню
    from Bot.handlers.start import send_start_menu
    await send_start_menu(message)
