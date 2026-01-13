# Инструкция по запуску

## 1. Установка зависимостей

```bash
cd /home/maximally/Univer/sRAG_system/rag_web
pip install -r requirements.txt
```

## 2. Запуск Ollama (если еще не запущен)

```bash
ollama serve
```

## 3. Запуск веб-приложения

```bash
cd /home/maximally/Univer/sRAG_system/rag_web
python run.py
```

Приложение будет доступно по адресу: **http://localhost:5000**

## 4. Проверка работы

1. Откройте браузер и перейдите на http://localhost:5000
2. Зарегистрируйтесь (или войдите, если уже есть аккаунт)
3. Перейдите в "Настройки Ollama"
4. Проверьте подключение к Ollama и выберите модели

## Структура проекта

```
sRAG_system/
├── imbibitor/          # Основной код RAG системы
│   ├── ollama_client/
│   ├── rag_system/
│   └── ...
└── rag_web/            # Веб-приложение
    ├── web_app/
    ├── run.py
    └── requirements.txt
```

Веб-приложение автоматически добавляет путь к `imbibitor` для импорта модулей.
