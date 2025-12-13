from aiogram import Router

from Bot.handlers import chat, main, select_model, settings, start

router = Router()

# Подключаем все роутеры обработчиков в правильном порядке
# Порядок важен: более специфичные обработчики должны быть подключены первыми
router.include_router(start.router)
router.include_router(main.router)
router.include_router(chat.router)
router.include_router(settings.router)
router.include_router(select_model.router)
