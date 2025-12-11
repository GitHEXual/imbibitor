"""Telegram бот для RAG системы."""
import os
import json
import tempfile
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import aiofiles
import config
from parser import parse_telegram_export
from rag_system import RAGSystem


# Состояния для FSM
class ChatStates(StatesGroup):
    waiting_for_question = State()


# Инициализация бота и диспетчера
bot = Bot(token=config.BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Глобальный экземпляр RAG системы
rag_system = RAGSystem()


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start."""
    welcome_text = (
        "Привет! Я бот для работы с базой знаний из экспорта Telegram чатов.\n\n"
        "Отправь мне JSON файл с экспортом чата, и я создам базу знаний.\n"
        "После этого ты сможешь задавать вопросы по содержимому чата.\n\n"
        "Команды:\n"
        "/start - показать это сообщение\n"
        "/load - загрузить существующую базу знаний"
    )
    await message.answer(welcome_text)


@dp.message(Command("load"))
async def cmd_load(message: Message, state: FSMContext):
    """Обработчик команды /load - загрузка существующей базы знаний."""
    if rag_system.load_knowledge_base():
        await message.answer("База знаний успешно загружена! Теперь ты можешь задавать вопросы.")
        await state.set_state(ChatStates.waiting_for_question)
    else:
        await message.answer(
            "База знаний не найдена. Сначала отправь JSON файл с экспортом чата."
        )


@dp.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    """Обработчик загрузки JSON файла."""
    document = message.document
    
    # Проверяем, что это JSON файл
    if not document.file_name.endswith('.json'):
        await message.answer("Пожалуйста, отправь JSON файл с экспортом чата.")
        return
    
    await message.answer("Обрабатываю файл... Это может занять некоторое время.")
    
    try:
        # Скачиваем файл
        file = await bot.get_file(document.file_id)
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp_file:
            tmp_path = tmp_file.name
            await bot.download_file(file.file_path, tmp_path)
        
        # Читаем и парсим JSON
        async with aiofiles.open(tmp_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            json_data = json.loads(content)
        
        # Парсим сообщения
        messages = parse_telegram_export(json_data)
        
        if not messages:
            await message.answer(
                "Не удалось извлечь сообщения из файла. "
                "Убедись, что файл содержит корректный экспорт Telegram чата."
            )
            os.unlink(tmp_path)
            return
        
        await message.answer(
            f"Найдено {len(messages)} сообщений. Создаю базу знаний..."
        )
        
        # Создаем базу знаний
        try:
            rag_system.build_knowledge_base(messages)
        except ConnectionError as e:
            await message.answer(
                f"❌ Ошибка подключения к Ollama:\n\n{str(e)}\n\n"
                "Убедитесь, что:\n"
                "1. Ollama запущен (проверьте командой: ollama serve)\n"
                "2. Модели установлены:\n"
                f"   - ollama pull {config.EMBEDDING_MODEL}\n"
                f"   - ollama pull {config.LLM_MODEL}\n"
                "3. Ollama доступен по адресу из .env файла"
            )
            os.unlink(tmp_path)
            return
        except ValueError as e:
            await message.answer(f"❌ Ошибка: {str(e)}")
            os.unlink(tmp_path)
            return
        
        # Сохраняем базу знаний
        try:
            rag_system.save_knowledge_base()
        except Exception as e:
            await message.answer(
                f"⚠️ База знаний создана, но не удалось сохранить: {str(e)}\n"
                "Попробуйте снова позже."
            )
            os.unlink(tmp_path)
            return
        
        await message.answer(
            f"✅ База знаний успешно создана из {len(messages)} сообщений!\n"
            "Теперь ты можешь задавать вопросы по содержимому чата."
        )
        
        await state.set_state(ChatStates.waiting_for_question)
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
    except json.JSONDecodeError:
        await message.answer("❌ Ошибка: файл не является валидным JSON.")
    except Exception as e:
        await message.answer(
            f"❌ Произошла ошибка при обработке файла:\n{str(e)}\n\n"
            "Проверьте формат файла и попробуйте снова."
        )


@dp.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    """Обработчик текстовых сообщений (вопросы к базе знаний)."""
    current_state = await state.get_state()
    
    # Если база знаний не готова, проверяем команды
    if not rag_system.is_ready():
        if message.text.startswith('/'):
            # Пропускаем команды
            return
        await message.answer(
            "База знаний не загружена. Отправь JSON файл с экспортом чата "
            "или используй команду /load для загрузки существующей базы."
        )
        return
    
    # Обрабатываем вопрос
    question = message.text.strip()
    
    if not question:
        await message.answer("Пожалуйста, задай вопрос.")
        return
    
    await message.answer("Ищу ответ...")
    
    try:
        # Выполняем запрос к RAG системе
        answer = rag_system.query(question)
        await message.answer(answer)
    except ConnectionError as e:
        await message.answer(
            f"❌ Ошибка подключения к Ollama:\n{str(e)}\n\n"
            "Убедитесь, что Ollama запущен и доступен."
        )
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка при обработке вопроса: {str(e)}")


async def main():
    """Главная функция для запуска бота."""
    # Пытаемся загрузить существующую базу знаний при старте
    if rag_system.load_knowledge_base():
        print("База знаний загружена при старте.")
    
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

