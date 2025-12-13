from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from Bot.handlers import chat, main, select_model, settings, start

router = Router()

# Обработчик для кнопки "Старт" - должен работать из любого состояния
# Регистрируем его в главном роутере с высоким приоритетом
@router.message(lambda m: m.text and m.text.strip() == "Старт")
async def handle_start_button_global(message: Message, state: FSMContext) -> None:
    from Bot.handlers.main import send_main_menu
    await send_main_menu(message, state)

# Подключаем все роутеры обработчиков в правильном порядке
# Порядок важен: более специфичные обработчики должны быть подключены первыми
router.include_router(start.router)
router.include_router(main.router)
router.include_router(chat.router)
router.include_router(settings.router)
router.include_router(select_model.router)
