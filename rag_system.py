"""RAG система на основе Langchain."""
import os
import requests
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except ImportError:
    # Fallback для старых версий
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config


class RAGSystem:
    """Класс для управления RAG системой через Langchain."""
    
    def __init__(self):
        """Инициализация RAG системы."""
        self.embeddings = None
        self.llm = None
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[Any] = None
        
        # Создаем директорию для базы знаний, если её нет
        os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)
        
        # Инициализируем эмбеддинги и LLM только при необходимости
        self._init_models()
    
    def _check_ollama_available(self) -> bool:
        """Проверяет доступность Ollama сервера."""
        try:
            response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _init_models(self):
        """Инициализирует модели Ollama."""
        if not self._check_ollama_available():
            # Модели будут инициализированы позже при необходимости
            return
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_URL
            )
            self.llm = OllamaLLM(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_URL
            )
        except Exception as e:
            raise ConnectionError(
                f"Не удалось подключиться к Ollama на {config.OLLAMA_URL}. "
                f"Убедитесь, что Ollama запущен. Ошибка: {str(e)}"
            )
    
    def build_knowledge_base(self, messages: List[str]) -> None:
        """
        Создает базу знаний из списка сообщений.
        
        Args:
            messages: Список текстовых сообщений из чата
            
        Raises:
            ConnectionError: Если Ollama недоступен
            ValueError: Если сообщения пусты
        """
        if not messages:
            raise ValueError("Список сообщений пуст.")
        
        # Проверяем доступность Ollama перед началом работы
        if not self._check_ollama_available():
            raise ConnectionError(
                f"Ollama недоступен на {config.OLLAMA_URL}. "
                "Пожалуйста, убедитесь, что Ollama запущен и доступен."
            )
        
        # Инициализируем модели, если еще не инициализированы
        if self.embeddings is None or self.llm is None:
            self._init_models()
        
        # Создаем документы Langchain из сообщений
        documents = [
            Document(page_content=msg, metadata={"index": i})
            for i, msg in enumerate(messages)
        ]
        
        # Создаем векторное хранилище FAISS через Langchain
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        except Exception as e:
            raise ConnectionError(
                f"Ошибка при создании эмбеддингов. "
                f"Проверьте, что модель {config.EMBEDDING_MODEL} установлена в Ollama. "
                f"Ошибка: {str(e)}"
            )
        
        # Инициализируем RAG цепочку
        self._init_qa_chain()
    
    def _init_qa_chain(self) -> None:
        """Инициализирует RAG цепочку для вопросов-ответов."""
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано. Сначала создайте базу знаний.")
        
        # Создаем retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Создаем промпт для RAG
        prompt_template = """Используй следующие фрагменты контекста из чата для ответа на вопрос.
Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.

Контекст: {context}

Вопрос: {question}

Ответ:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Форматируем контекст из документов
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        
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
        """
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано.")
        
        save_path = os.path.join(config.KNOWLEDGE_BASE_DIR, name)
        self.vectorstore.save_local(save_path)
    
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
        
        # Проверяем доступность Ollama перед загрузкой
        if not self._check_ollama_available():
            print(f"Ollama недоступен на {config.OLLAMA_URL}. Не могу загрузить базу знаний.")
            return False
        
        # Инициализируем модели, если еще не инициализированы
        if self.embeddings is None or self.llm is None:
            self._init_models()
        
        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._init_qa_chain()
            return True
        except Exception as e:
            print(f"Ошибка при загрузке базы знаний: {e}")
            return False
    
    def query(self, question: str) -> str:
        """
        Выполняет запрос к RAG системе.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Ответ на основе базы знаний
        """
        if self.qa_chain is None:
            raise ValueError("RAG цепочка не инициализирована. Сначала создайте или загрузите базу знаний.")
        
        try:
            answer = self.qa_chain.invoke(question)
            return answer if answer else "Не удалось получить ответ."
        except Exception as e:
            return f"Ошибка при обработке запроса: {str(e)}"
    
    def is_ready(self) -> bool:
        """
        Проверяет, готова ли RAG система к работе.
        
        Returns:
            True если система готова, False иначе
        """
        return self.qa_chain is not None and self.vectorstore is not None

