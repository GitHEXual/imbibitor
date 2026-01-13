# RAG Web Application

Веб-приложение для работы с RAG системой.

## Структура

```
rag_web/
├── web_app/          # Flask приложение
├── run.py            # Точка входа
├── requirements.txt  # Зависимости
└── README.md         # Этот файл
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что Ollama запущен:
```bash
ollama serve
```

## Запуск

Из директории `rag_web/`:

```bash
python run.py
```

Или:

```bash
cd /home/maximally/Univer/sRAG_system/rag_web
python run.py
```

Приложение будет доступно по адресу: http://localhost:5000

## Зависимости от imbibitor

Веб-приложение использует модули из `../imbibitor/`:
- `ollama_client` - для работы с Ollama API
- `rag_system` - для работы с RAG системой

Путь к `imbibitor` добавляется автоматически в `run.py` и `web_app/utils/ollama_helper.py`.

## Функционал

- Авторизация и регистрация пользователей
- Главное меню
- Настройки Ollama (URL, выбор моделей)
- Изоляция данных по пользователям (отдельные коллекции ChromaDB)
