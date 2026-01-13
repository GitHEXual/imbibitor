# Flask Web App для RAG системы

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

```bash
python run.py
```

Приложение будет доступно по адресу: http://localhost:5000

## Функционал

### Авторизация и регистрация
- Регистрация новых пользователей (`/auth/register`)
- Вход в систему (`/auth/login`)
- Выход из системы (`/auth/logout`)

### Главное меню
- Главная страница с навигацией (`/`)
- Кнопка выхода

### Настройки Ollama
- Настройка URL Ollama (`/settings/ollama`)
- Выбор модели для embeddings
- Выбор модели для генерации текста
- Автоматическое получение списка доступных моделей из Ollama

## Структура

- `web_app/` - Flask приложение
  - `models.py` - SQLAlchemy модели (User, UserSettings)
  - `forms.py` - WTForms формы
  - `routes/` - Роуты (auth, main, settings)
  - `templates/` - HTML шаблоны
  - `utils/` - Вспомогательные функции
- `run.py` - Точка входа для запуска приложения

## База данных

SQLite база данных создается автоматически в `web_app/instance/database.db`

ChromaDB использует коллекции по пользователям: `user_{user_id}_messages`

## Переменные окружения

- `SECRET_KEY` - секретный ключ для Flask (по умолчанию: dev-secret-key)
- `DATABASE_URL` - URL базы данных (по умолчанию: SQLite)
