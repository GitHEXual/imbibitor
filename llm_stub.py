"""Заглушка для LLM модели."""
from typing import List, Optional, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult


class LLMStub(LLM):
    """Заглушка LLM для тестирования RAG системы без реальной модели."""
    
    model_name: str = "stub"
    
    @property
    def _llm_type(self) -> str:
        """Тип LLM."""
        return "stub"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Генерирует ответ на основе промпта.
        
        Args:
            prompt: Промпт с контекстом и вопросом
            stop: Список стоп-слов
            run_manager: Менеджер колбэков
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        return self._generate_stub_response(prompt)
    
    def _generate_stub_response(self, prompt: str) -> str:
        """
        Генерирует заглушку ответа на основе промпта.
        
        Args:
            prompt: Промпт с контекстом и вопросом
            
        Returns:
            Заглушка ответа
        """
        # Извлекаем вопрос и контекст из промпта
        question = ""
        context = ""
        
        if "Вопрос:" in prompt:
            parts = prompt.split("Вопрос:")
            if len(parts) >= 2:
                context = parts[0].replace("Контекст:", "").strip()
                question = parts[1].replace("Ответ:", "").strip()
        
        # Формируем ответ на основе контекста
        if context and question:
            # Пытаемся найти ответ в контексте
            context_lower = context.lower()
            question_lower = question.lower()
            
            # Простой поиск ключевых слов из вопроса в контексте
            question_words = [w for w in question_lower.split() if len(w) > 3]
            found_sentences = []
            for sentence in context.split("\n\n"):
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in question_words):
                    found_sentences.append(sentence[:200])  # Ограничиваем длину
            
            if found_sentences:
                return f"[Заглушка LLM] На основе контекста: {' '.join(found_sentences[:2])}..."
            
            return f"[Заглушка LLM] На основе предоставленного контекста, ответ на вопрос '{question[:100]}...' будет найден в базе знаний. В реальной системе здесь будет ответ от LLM модели."
        
        return "[Заглушка LLM] Ответ на основе контекста из базы знаний. В реальной системе здесь будет ответ от LLM модели."

