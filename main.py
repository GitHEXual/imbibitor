import asyncio
from dotenv import load_dotenv
from Bot.bot import create_bot, create_dispatcher, get_bot_token


async def main():
    load_dotenv()
    token = get_bot_token()
    bot = create_bot(token)
    dp = create_dispatcher()
    
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

