"""Парсер JSON экспорта Telegram чата."""
import json
from typing import List, Dict, Any


def parse_telegram_export(json_data: Dict[str, Any]) -> List[str]:
    """
    Парсит JSON экспорт Telegram чата и извлекает текстовые сообщения.
    
    Args:
        json_data: Словарь с данными экспорта чата
        
    Returns:
        Список текстовых сообщений из чата
    """
    messages = []
    
    # Проверяем структуру экспорта Telegram
    # Обычно сообщения находятся в messages или chats[0].messages
    if "messages" in json_data:
        chat_messages = json_data["messages"]
    elif "chats" in json_data and len(json_data["chats"]) > 0:
        if "messages" in json_data["chats"][0]:
            chat_messages = json_data["chats"][0]["messages"]
        else:
            chat_messages = []
    else:
        return messages
    
    for msg in chat_messages:
        # Извлекаем текст сообщения
        text = None
        
        # Обычно текст находится в поле "text" или "message"
        if isinstance(msg.get("text"), str):
            text = msg["text"]
        elif isinstance(msg.get("message"), str):
            text = msg["message"]
        elif isinstance(msg.get("text"), list):
            # Иногда текст может быть списком объектов с полем "text"
            text_parts = []
            for part in msg["text"]:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            text = " ".join(text_parts) if text_parts else None
        
        # Если есть медиа с подписью, используем подпись
        if not text and "caption" in msg:
            text = msg["caption"]
        
        # Добавляем сообщение, если есть текст
        if text and text.strip():
            # Добавляем метаданные для контекста (опционально)
            sender = msg.get("from", msg.get("from_id", "Unknown"))
            date = msg.get("date", "")
            
            # Формируем сообщение с контекстом
            formatted_msg = f"[{date}] {sender}: {text}"
            messages.append(formatted_msg)
    
    return messages


def parse_json_file(file_path: str) -> List[str]:
    """
    Читает JSON файл и парсит его.
    
    Args:
        file_path: Путь к JSON файлу
        
    Returns:
        Список текстовых сообщений
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return parse_telegram_export(json_data)


async def parse_json_file_async(file_path: str) -> List[str]:
    """
    Асинхронно читает JSON файл и парсит его.
    
    Args:
        file_path: Путь к JSON файлу
        
    Returns:
        Список текстовых сообщений
    """
    import aiofiles
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        json_data = json.loads(content)
    return parse_telegram_export(json_data)

