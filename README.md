# Telegram RAG Bot

Telegram бот для создания базы знаний из экспорта чатов и ответов на вопросы с использованием RAG системы.

## Требования

- Python 3.8+
- Ollama (локально запущенный)
- Модели Ollama:
  - `nomic-embed` (для эмбеддингов)
  - `llama3` (для генерации ответов)

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Настройте `.env` файл:
```
BOT_TOKEN=your_bot_token_here
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed
LLM_MODEL=llama3
```

3. Установите модели Ollama:
```bash
ollama pull nomic-embed
ollama pull llama3
```

4. Запустите Ollama (если еще не запущен):
```bash
ollama serve
```

## Запуск

```bash
python bot.py
```

## Использование

1. Отправьте боту команду `/start`
2. Отправьте JSON файл с экспортом Telegram чата
3. Дождитесь создания базы знаний
4. Задавайте вопросы по содержимому чата

## Команды

- `/start` - показать приветственное сообщение
- `/load` - загрузить существующую базу знаний
