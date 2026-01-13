"""Helper functions for working with Ollama."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path to import from imbibitor
# Path structure: rag_web/web_app/utils/ollama_helper.py -> rag_web/ -> sRAG_system/ -> imbibitor/
imbibitor_path = Path(__file__).parent.parent.parent.parent / 'imbibitor'
if str(imbibitor_path) not in sys.path:
    sys.path.insert(0, str(imbibitor_path))

from ollama_client import OllamaClient


def get_available_models(ollama_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """
    Get list of available models from Ollama.
    
    Args:
        ollama_url: Ollama API base URL
        
    Returns:
        List of model dictionaries
    """
    try:
        client = OllamaClient(base_url=ollama_url)
        return client.list_models()
    except Exception:
        return []


def categorize_models(models: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Categorize models into embeddings and LLM models.
    
    Args:
        models: List of model dictionaries from Ollama
        
    Returns:
        Tuple of (embedding_models, llm_models) lists
    """
    embedding_models = []
    llm_models = []
    
    for model in models:
        model_name = model.get('name', '')
        # Simple heuristic: if name contains "embed", it's likely an embedding model
        if 'embed' in model_name.lower():
            embedding_models.append(model_name)
        else:
            llm_models.append(model_name)
    
    # If no embedding models found by name, try to test via API
    # For now, we'll use the simple heuristic
    return sorted(embedding_models), sorted(llm_models)


def test_ollama_connection(ollama_url: str) -> Tuple[bool, str]:
    """
    Test connection to Ollama.
    
    Args:
        ollama_url: Ollama API base URL
        
    Returns:
        Tuple of (success, message)
    """
    try:
        client = OllamaClient(base_url=ollama_url)
        models = client.list_models()
        return True, f"Подключение успешно. Найдено моделей: {len(models)}"
    except Exception as e:
        return False, f"Ошибка подключения: {str(e)}"
