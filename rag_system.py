"""RAG система на основе Langchain."""
import os
import requests
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel

import config
import ollama_config
from llm_stub import LLMStub


class RAGSystem:
    """Класс для управления RAG системой через Langchain."""
    
    def __init__(self, use_stub_llm: bool = False):
        """
        Инициализация RAG системы.
        
        Args:
            use_stub_llm: Использовать заглушку LLM вместо реальной модели (по умолчанию False - используется реальная модель)
        """
        self.use_stub_llm = use_stub_llm
        self.embeddings = None
        self.llm: Optional[BaseLanguageModel] = None
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[Any] = None
        
        # Создаем директорию для базы знаний, если её нет
        os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)
        
        # Инициализируем LLM (заглушку или реальную модель)
        self._init_llm()
    
    def _check_ollama_available(self) -> bool:
        """Проверяет доступность Ollama сервера."""
        try:
            response = requests.get(
                f"{ollama_config.OLLAMA_URL}/api/tags",
                timeout=ollama_config.OLLAMA_TIMEOUT
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _init_llm(self):
        """Инициализирует LLM (заглушку или реальную модель)."""
        if self.use_stub_llm:
            # Используем заглушку
            self.llm = LLMStub(model_name="stub")
            return
        
        # Используем реальную модель Ollama
        if not self._check_ollama_available():
            print(f"⚠️ Ollama недоступен, используем заглушку LLM")
            self.llm = LLMStub(model_name="stub")
            return
        
        try:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama
            
            self.llm = ChatOllama(
                model=ollama_config.LLM_MODEL,
                base_url=ollama_config.OLLAMA_URL,
                temperature=0.2,  # Низкая температура для более детерминированных ответов
                num_predict=100,  # Короткие ответы
                repeat_penalty=1.3,  # Штраф за повторения
                top_p=0.9
            )
        except Exception as e:
            print(f"⚠️ Ошибка инициализации LLM: {e}, используем заглушку")
            self.llm = LLMStub(model_name="stub")
    
    def _init_embeddings(self):
        """Инициализирует модель для эмбеддингов."""
        if self.embeddings is not None:
            return
        
        # Проверяем доступность Ollama
        if not self._check_ollama_available():
            raise ConnectionError(
                f"Ollama недоступен на {ollama_config.OLLAMA_URL}. "
                "Пожалуйста, убедитесь, что Ollama запущен и доступен."
            )
        
        try:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                from langchain_community.embeddings import OllamaEmbeddings
            
            self.embeddings = OllamaEmbeddings(
                model=ollama_config.EMBEDDING_MODEL,
                base_url=ollama_config.OLLAMA_URL
            )
        except Exception as e:
            raise ConnectionError(
                f"Не удалось инициализировать эмбеддинги. "
                f"Проверьте, что модель {ollama_config.EMBEDDING_MODEL} установлена в Ollama. "
                f"Ошибка: {str(e)}"
            )
    
    def build_knowledge_base(self, messages: List[str]) -> None:
        """
        Создает базу знаний из списка сообщений.
        
        Args:
            messages: Список текстовых сообщений из чата
            
        Raises:
            ConnectionError: Если Ollama недоступен для эмбеддингов
            ValueError: Если сообщения пусты
        """
        if not messages:
            raise ValueError("Список сообщений пуст.")
        
        # Инициализируем эмбеддинги
        self._init_embeddings()
        
        # Создаем документы Langchain из сообщений
        documents = [
            Document(page_content=msg, metadata={"index": i, "source": "telegram_chat"})
            for i, msg in enumerate(messages)
        ]
        
        # Создаем векторное хранилище FAISS через Langchain
        # Используем батчинг для избежания перегрузки Ollama
        try:
            import time
            # Создаем эмбеддинги батчами по 10 документов
            batch_size = 10
            all_embeddings = []
            
            print(f"Создаю эмбеддинги для {len(documents)} документов (батчами по {batch_size})...")
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_texts = [doc.page_content for doc in batch]
                
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    print(f"  Обработано {min(i+batch_size, len(documents))}/{len(documents)} документов")
                    
                    # Небольшая задержка между батчами
                    if i + batch_size < len(documents):
                        time.sleep(0.1)
                except Exception as batch_error:
                    print(f"⚠️ Ошибка в батче {i//batch_size + 1}: {batch_error}")
                    # Пропускаем проблемный батч и продолжаем
                    all_embeddings.extend([None] * len(batch))
            
            # Фильтруем None значения
            valid_docs = [doc for doc, emb in zip(documents, all_embeddings) if emb is not None]
            valid_embeddings = [emb for emb in all_embeddings if emb is not None]
            
            if not valid_docs:
                raise ConnectionError("Не удалось создать ни одного эмбеддинга")
            
            print(f"✓ Создано {len(valid_embeddings)} эмбеддингов из {len(documents)} документов")
            
            # Создаем FAISS индекс из эмбеддингов
            # Используем from_embeddings с парами (текст, эмбеддинг)
            text_embeddings = [
                (doc.page_content, emb) 
                for doc, emb in zip(valid_docs, valid_embeddings)
            ]
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embeddings
            )
            
        except Exception as e:
            raise ConnectionError(
                f"Ошибка при создании эмбеддингов. "
                f"Проверьте, что модель {ollama_config.EMBEDDING_MODEL} установлена в Ollama. "
                f"Ошибка: {str(e)}"
            )
        
        # Инициализируем RAG цепочку
        self._init_qa_chain()
    
    def _init_qa_chain(self) -> None:
        """Инициализирует RAG цепочку для вопросов-ответов."""
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано. Сначала создайте базу знаний.")
        
        if self.llm is None:
            raise ValueError("LLM не инициализирован.")
        
        # Создаем retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Форматируем контекст из документов
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Создаем промпт с явным указанием формата ответа
        prompt_template = """Ты помощник. Отвечай на вопросы на основе контекста из чата Telegram.

Контекст:
{context}

Вопрос: {question}

ВАЖНО: Отвечай ТОЛЬКО на русском языке. Будь кратким (1-2 предложения).

Ответ:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Создаем RAG цепочку через новый API Langchain
        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def save_knowledge_base(self, name: str = "default") -> None:
        """
        Сохраняет базу знаний в файл.
        
        Args:
            name: Имя базы знаний
            
        Raises:
            ValueError: Если векторное хранилище не инициализировано
        """
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано.")
        
        save_path = os.path.join(config.KNOWLEDGE_BASE_DIR, name)
        self.vectorstore.save_local(save_path)
        print(f"✓ База знаний сохранена в {save_path}")
    
    def load_knowledge_base(self, name: str = "default") -> bool:
        """
        Загружает базу знаний из файла.
        
        Args:
            name: Имя базы знаний
            
        Returns:
            True если загрузка успешна, False если файл не найден
        """
        load_path = os.path.join(config.KNOWLEDGE_BASE_DIR, name)
        
        if not os.path.exists(load_path):
            return False
        
        # Инициализируем эмбеддинги для загрузки
        try:
            self._init_embeddings()
        except ConnectionError as e:
            print(f"⚠️ {e}")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._init_qa_chain()
            print(f"✓ База знаний загружена из {load_path}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при загрузке базы знаний: {e}")
            return False
    
    def query(self, question: str) -> str:
        """
        Выполняет запрос к RAG системе.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Ответ на основе базы знаний
            
        Raises:
            ValueError: Если RAG цепочка не инициализирована
        """
        if self.qa_chain is None:
            raise ValueError("RAG цепочка не инициализирована. Сначала создайте или загрузите базу знаний.")
        
        try:
            answer = self.qa_chain.invoke(question)
            
            if not answer:
                return "Не удалось получить ответ."
            
            # Постобработка ответа - убираем артефакты промпта
            answer_clean = answer.strip()
            
            # Убираем строки с инструкциями
            lines = answer_clean.split('\n')
            filtered_lines = []
            skip_phrases = [
                'используй следующие', 'use the following',
                'контекст из чата:', 'context from',
                'правила:', 'rules:', 'инструкции:',
                'ответ (на русском', 'ответ:', 'answer:'
            ]
            
            for line in lines:
                line_lower = line.lower().strip()
                # Пропускаем строки с инструкциями или заголовками промпта
                if any(phrase in line_lower for phrase in skip_phrases):
                    # Но оставляем строку, если она содержит реальный контент после заголовка
                    if ':' in line and len(line.split(':', 1)[1].strip()) > 10:
                        filtered_lines.append(line.split(':', 1)[1].strip())
                    continue
                if line.strip():
                    filtered_lines.append(line)
            
            answer_clean = '\n'.join(filtered_lines).strip()
            
            # Если после фильтрации ничего не осталось, возвращаем оригинал
            if not answer_clean:
                answer_clean = answer.strip()
            
            # Обрезаем слишком длинные ответы (максимум 200 символов)
            if len(answer_clean) > 200:
                # Пытаемся обрезать по последнему предложению
                sentences = answer_clean.split('.')
                result = []
                total_len = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if total_len + len(sent) > 200:
                        break
                    result.append(sent)
                    total_len += len(sent) + 2
                answer_clean = '. '.join(result).strip()
                if answer_clean and not answer_clean.endswith(('.', '!', '?')):
                    answer_clean += '.'
                if len(answer_clean) > 200:
                    answer_clean = answer_clean[:197] + "..."
            
            return answer_clean
            
        except Exception as e:
            return f"Ошибка при обработке запроса: {str(e)}"
    
    def is_ready(self) -> bool:
        """
        Проверяет, готова ли RAG система к работе.
        
        Returns:
            True если система готова, False иначе
        """
        return self.qa_chain is not None and self.vectorstore is not None
    
    def get_stats(self) -> dict:
        """
        Возвращает статистику о системе.
        
        Returns:
            Словарь со статистикой
        """
        return {
            "is_ready": self.is_ready(),
            "has_vectorstore": self.vectorstore is not None,
            "has_qa_chain": self.qa_chain is not None,
            "has_embeddings": self.embeddings is not None,
            "has_llm": self.llm is not None,
            "llm_type": type(self.llm).__name__ if self.llm else None,
            "use_stub_llm": self.use_stub_llm,
            "ollama_available": self._check_ollama_available() if not self.use_stub_llm else None
        }
