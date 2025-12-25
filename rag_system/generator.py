"""Post generator using LLM and retrieved messages."""

from typing import List, Dict, Any, Optional
from ollama_client import OllamaLLM


class PostGenerator:
    """Generator for creating posts based on topic and similar messages."""
    
    def __init__(self, llm_client: OllamaLLM):
        """
        Initialize post generator.
        
        Args:
            llm_client: OllamaLLM instance
        """
        self.llm_client = llm_client
    
    def _format_similar_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format similar messages for the prompt.
        
        Args:
            messages: List of similar message dictionaries
            
        Returns:
            Formatted string with messages
        """
        formatted = []
        for i, msg in enumerate(messages, 1):
            text = msg.get("text", "")
            metadata = msg.get("metadata", {})
            from_user = metadata.get("from", "Unknown")
            date = metadata.get("date", "")
            
            formatted.append(
                f"{i}. [{from_user}, {date}]: {text}"
            )
        
        return "\n".join(formatted)
    
    def _create_prompt(
        self,
        topic: str,
        similar_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Create prompt for post generation.
        
        Args:
            topic: Topic for the post
            similar_messages: List of similar messages as context
            
        Returns:
            Formatted prompt string
        """
        messages_text = self._format_similar_messages(similar_messages)
        
        prompt = f"""Тема поста: {topic}

Похожие сообщения из базы данных:
{messages_text}

На основе этих сообщений сгенерируй пост на тему "{topic}". 
Пост должен быть информативным, интересным и соответствовать стилю и содержанию найденных сообщений.
Учти, что это сообщения от RAG системы, и некоторые могут быть мусорными.
"""
        return prompt
    
    def generate_post(
        self,
        topic: str,
        similar_messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a post based on topic and similar messages.
        
        Args:
            topic: Topic for the post
            similar_messages: List of similar messages as context
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated post text
        """
        prompt = self._create_prompt(topic, similar_messages)
        
        generated_text = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return generated_text.strip()

