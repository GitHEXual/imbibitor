from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from Bot.handlers import router


class TelegramBot:
    def __init__(self, token: str):
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.dp = Dispatcher()
        self.dp.include_router(router)

    async def start(self):
        await self.dp.start_polling(self.bot)
