import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from Bot.handlers import router


async def main():
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN not found in .env file")
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

