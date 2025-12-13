from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

# Главный роутер для всех обработчиков
router = Router()


def setup_handlers() -> None:
    """Регистрирует все обработчики на главном роутере"""
    # Импортируем обработчики (делаем это здесь, чтобы избежать циклических зависимостей)
    from Bot.handlers import start, main, chat, settings, select_model
    
    # Обработчик для кнопки "Старт" - должен работать из любого состояния
    # Регистрируем его в главном роутере с высоким приоритетом
    @router.message(lambda m: m.text and m.text.strip() == "Старт")
    async def handle_start_button_global(message: Message, state: FSMContext) -> None:
        from Bot.handlers.main import send_main_menu
        await send_main_menu(message, state)
    
    # Регистрируем обработчики из каждого модуля
    start.register_handlers(router)
    main.register_handlers(router)
    chat.register_handlers(router)
    settings.register_handlers(router)
    select_model.register_handlers(router)
