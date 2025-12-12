"""Telegram бот для генерации постов на основе контекста чата."""
import os
import json
import tempfile
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import aiofiles
import config
import ollama_config
from parser import parse_telegram_export
from rag_system import RAGSystem


# Состояния для FSM
class ChatStates(StatesGroup):
    waiting_for_post_idea = State()  # Ожидание идеи для поста
    waiting_for_kb_name = State()  # Ожидание имени для новой базы знаний


# Инициализация бота и диспетчера
bot = Bot(token=config.BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Глобальный экземпляр RAG системы (используем реальную модель Ollama)
rag_system = RAGSystem(use_stub_llm=False)

# Текущая выбранная база знаний (хранится в памяти)
current_kb_name = "default"


def get_available_knowledge_bases() -> list[str]:
    """Получает список доступных баз знаний."""
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not os.path.exists(kb_dir):
        return []
    
    # FAISS сохраняет базу знаний как директорию с файлами index.faiss и index.pkl
    # Проверяем наличие этих файлов в поддиректориях
    kb_list = []
    for item in os.listdir(kb_dir):
        item_path = os.path.join(kb_dir, item)
        if os.path.isdir(item_path):
            # Проверяем наличие файлов FAISS
            if os.path.exists(os.path.join(item_path, "index.faiss")) or \
               os.path.exists(os.path.join(item_path, "index.pkl")):
                kb_list.append(item)
    
    return sorted(kb_list)


def get_main_keyboard() -> InlineKeyboardMarkup:
    """Создает основную клавиатуру с кнопками."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📚 Список баз", callback_data="list_kb"),
            InlineKeyboardButton(text="➕ Создать базу", callback_data="create_kb")
        ],
        [
            InlineKeyboardButton(text="🔄 Загрузить базу", callback_data="load_kb"),
            InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help")
        ]
    ])
    return keyboard


def get_kb_list_keyboard(kb_list: list[str]) -> InlineKeyboardMarkup:
    """Создает клавиатуру со списком баз знаний."""
    buttons = []
    for kb_name in kb_list:
        buttons.append([InlineKeyboardButton(
            text=f"{'✅ ' if kb_name == current_kb_name else ''}{kb_name}",
            callback_data=f"select_kb_{kb_name}"
        )])
    buttons.append([InlineKeyboardButton(text="🔙 Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start."""
    global current_kb_name
    
    welcome_text = (
        "Привет! Я бот для генерации постов на основе контекста чата.\n\n"
        "📝 Как это работает:\n"
        "1. Загрузи JSON файл с экспортом Telegram чата\n"
        "2. Напиши идею для поста\n"
        "3. Я сгенерирую пост, используя контекст из чата!\n\n"
        f"Текущая база знаний: {current_kb_name}"
    )
    await message.answer(welcome_text, reply_markup=get_main_keyboard())


@dp.callback_query(F.data == "load_kb")
async def callback_load_kb(callback: CallbackQuery, state: FSMContext):
    """Обработчик кнопки загрузки базы знаний."""
    global current_kb_name
    await callback.answer()
    
    if rag_system.load_knowledge_base(current_kb_name):
        await callback.message.answer(
            f"✅ База знаний '{current_kb_name}' успешно загружена!\n"
            "Теперь напиши идею для поста, и я сгенерирую его на основе контекста чата."
        )
        await state.set_state(ChatStates.waiting_for_post_idea)
    else:
        await callback.message.answer(
            f"❌ База знаний '{current_kb_name}' не найдена.\n"
            "Используй кнопку 'Создать базу' для создания новой базы."
        )


@dp.callback_query(F.data == "list_kb")
async def callback_list_kb(callback: CallbackQuery):
    """Обработчик кнопки списка баз знаний."""
    global current_kb_name
    await callback.answer()
    
    kb_list = get_available_knowledge_bases()
    
    if not kb_list:
        await callback.message.answer(
            "📭 Нет доступных баз знаний.\n"
            "Используй кнопку 'Создать базу' для создания новой."
        )
        return
    
    kb_text = "📚 Доступные базы знаний:\n\n"
    for i, kb_name in enumerate(kb_list, 1):
        marker = "✅" if kb_name == current_kb_name else "  "
        kb_text += f"{marker} {i}. {kb_name}\n"
    
    kb_text += f"\nТекущая база: {current_kb_name}"
    await callback.message.answer(kb_text, reply_markup=get_kb_list_keyboard(kb_list))


@dp.callback_query(F.data.startswith("select_kb_"))
async def callback_select_kb(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора базы знаний."""
    global current_kb_name
    await callback.answer()
    
    kb_name = callback.data.replace("select_kb_", "")
    
    # Загружаем базу знаний
    if rag_system.load_knowledge_base(kb_name):
        current_kb_name = kb_name
        await callback.message.answer(
            f"✅ База знаний '{kb_name}' выбрана и загружена!\n"
            "Теперь напиши идею для поста, и я сгенерирую его на основе контекста чата."
        )
        await state.set_state(ChatStates.waiting_for_post_idea)
    else:
        await callback.message.answer(
            f"❌ Не удалось загрузить базу знаний '{kb_name}'."
        )


@dp.callback_query(F.data == "create_kb")
async def callback_create_kb(callback: CallbackQuery, state: FSMContext):
    """Обработчик кнопки создания новой базы знаний."""
    await callback.answer()
    await callback.message.answer(
        "📝 Для создания новой базы знаний:\n\n"
        "1. Отправь JSON файл с экспортом Telegram чата\n"
        "2. База будет создана автоматически с именем 'default'\n\n"
        "Или используй команду /create <имя> для указания имени."
    )


@dp.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    """Обработчик загрузки JSON файла."""
    global current_kb_name
    document = message.document
    
    # Проверяем, что это JSON файл
    if not document.file_name.endswith('.json'):
        await message.answer("Пожалуйста, отправь JSON файл с экспортом чата.")
        return
    
    # Получаем имя базы знаний из состояния (если создаем новую) или используем текущую
    data = await state.get_data()
    kb_name = data.get('creating_kb_name', current_kb_name)
    
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
            f"Найдено {len(messages)} сообщений. Создаю базу знаний '{kb_name}'..."
        )
        
        # Создаем базу знаний
        try:
            await message.answer("Начинаю создание базы знаний...")
            rag_system.build_knowledge_base(messages)
            await message.answer("База знаний создана в памяти, сохраняю...")
        except ConnectionError as e:
            await message.answer(
                f"❌ Ошибка подключения к Ollama:\n\n{str(e)}\n\n"
                "Убедитесь, что:\n"
                "1. Ollama запущен (проверьте командой: ollama serve)\n"
                "2. Модель для эмбеддингов установлена:\n"
                f"   - ollama pull {ollama_config.EMBEDDING_MODEL}\n"
                "3. Ollama доступен по адресу из .env файла"
            )
            os.unlink(tmp_path)
            return
        except ValueError as e:
            await message.answer(f"❌ Ошибка: {str(e)}")
            os.unlink(tmp_path)
            return
        except Exception as e:
            await message.answer(
                f"❌ Неожиданная ошибка при создании базы знаний:\n{str(e)}\n\n"
                "Проверьте логи для подробностей."
            )
            import traceback
            print(f"Ошибка создания базы знаний: {e}")
            traceback.print_exc()
            os.unlink(tmp_path)
            return
        
        # Сохраняем базу знаний с указанным именем
        try:
            print(f"Сохраняю базу знаний '{kb_name}'...")
            rag_system.save_knowledge_base(kb_name)
            print(f"✓ База знаний '{kb_name}' сохранена")
            current_kb_name = kb_name  # Обновляем текущую базу
            await state.update_data(creating_kb_name=None)  # Очищаем состояние
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            await message.answer(
                f"⚠️ База знаний создана в памяти, но не удалось сохранить на диск:\n{error_msg}\n\n"
                "Попробуйте снова позже или проверьте права доступа к директории."
            )
            print(f"Ошибка сохранения базы знаний: {e}")
            os.unlink(tmp_path)
            return
        
        await message.answer(
            f"✅ База знаний '{kb_name}' успешно создана из {len(messages)} сообщений!\n"
            "Теперь напиши идею для поста, и я сгенерирую его на основе контекста чата!",
            reply_markup=get_main_keyboard()
        )
        
        await state.set_state(ChatStates.waiting_for_post_idea)
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
    except json.JSONDecodeError:
        await message.answer("❌ Ошибка: файл не является валидным JSON.")
    except Exception as e:
        await message.answer(
            f"❌ Произошла ошибка при обработке файла:\n{str(e)}\n\n"
            "Проверьте формат файла и попробуйте снова."
        )


@dp.callback_query(F.data == "main_menu")
async def callback_main_menu(callback: CallbackQuery):
    """Обработчик кнопки возврата в главное меню."""
    await callback.answer()
    await callback.message.answer("Главное меню:", reply_markup=get_main_keyboard())


@dp.callback_query(F.data == "help")
async def callback_help(callback: CallbackQuery):
    """Обработчик кнопки помощи."""
    await callback.answer()
    help_text = (
        "📖 Помощь:\n\n"
        "1. Загрузи JSON файл с экспортом Telegram чата\n"
        "2. Нажми 'Загрузить базу' для активации\n"
        "3. Напиши идею для поста\n"
        "4. Бот сгенерирует пост на основе контекста чата!\n\n"
        "💡 Совет: Чем больше сообщений в чате, тем лучше контекст для генерации."
    )
    await callback.message.answer(help_text, reply_markup=get_main_keyboard())


@dp.message(Command("create"))
async def cmd_create(message: Message, state: FSMContext):
    """Обработчик команды /create - создание новой базы знаний с именем."""
    global current_kb_name
    
    command_parts = message.text.split(maxsplit=1)
    if len(command_parts) < 2:
        await message.answer(
            "❌ Укажи имя для новой базы знаний.\n"
            "Пример: /create my_new_kb\n\n"
            "После этого отправь JSON файл с экспортом чата."
        )
        return
    
    kb_name = command_parts[1].strip()
    
    if not all(c.isalnum() or c in ('_', '-') for c in kb_name):
        await message.answer(
            "❌ Имя базы знаний может содержать только буквы, цифры, "
            "подчеркивания и дефисы."
        )
        return
    
    kb_list = get_available_knowledge_bases()
    if kb_name in kb_list:
        await message.answer(
            f"⚠️ База знаний '{kb_name}' уже существует."
        )
        return
    
    current_kb_name = kb_name
    await state.update_data(creating_kb_name=kb_name)
    await message.answer(
        f"✅ Имя базы знаний установлено: '{kb_name}'\n\n"
        "Теперь отправь JSON файл с экспортом чата для создания базы знаний."
    )


@dp.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    """Обработчик текстовых сообщений - генерация постов."""
    # Пропускаем команды
    if message.text.startswith('/'):
        return
    
    # Если база знаний не готова
    if not rag_system.is_ready():
        await message.answer(
            "База знаний не загружена. Отправь JSON файл с экспортом чата "
            "или используй кнопку 'Загрузить базу'.",
            reply_markup=get_main_keyboard()
        )
        return
    
    # Обрабатываем идею для поста
    idea = message.text.strip()
    
    if not idea:
        await message.answer("Пожалуйста, напиши идею для поста.")
        return
    
    await message.answer("Генерирую пост на основе контекста чата...")
    
    try:
        # Генерируем пост через RAG систему
        post = rag_system.generate_post(idea)
        await message.answer(post)
    except ConnectionError as e:
        await message.answer(
            f"❌ Ошибка подключения к Ollama:\n{str(e)}\n\n"
            "Убедитесь, что Ollama запущен и доступен."
        )
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка при генерации поста: {str(e)}")


async def main():
    """Главная функция для запуска бота."""
    global current_kb_name
    # Пытаемся загрузить существующую базу знаний при старте
    if rag_system.load_knowledge_base(current_kb_name):
        print(f"База знаний '{current_kb_name}' загружена при старте.")
    else:
        print(f"База знаний '{current_kb_name}' не найдена при старте.")
    
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

