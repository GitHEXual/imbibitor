"""Data loader for Telegram messages."""

import json
from typing import List, Dict, Any


def load_messages(json_path: str) -> List[Dict[str, Any]]:
    """
    Load and filter text messages from Telegram export JSON.
    
    Args:
        json_path: Path to result.json file
        
    Returns:
        List of message dictionaries with only text messages
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get("messages", [])
    
    # Filter only text messages
    text_messages = []
    for msg in messages:
        # Check if it's a message type and has text field
        if msg.get("type") == "message" and msg.get("text"):
            text = msg.get("text", "")
            
            # Handle case when text is a list (from text_entities)
            if isinstance(text, list):
                # Extract text from list of entities
                text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in text
                )
            else:
                text = str(text)
            
            # Skip empty text
            if text and text.strip():
                text_messages.append({
                    "id": msg.get("id"),
                    "date": msg.get("date"),
                    "date_unixtime": msg.get("date_unixtime"),
                    "from": msg.get("from"),
                    "from_id": msg.get("from_id"),
                    "text": text.strip()
                })
    
    return text_messages

