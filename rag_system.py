"""RAG система на основе Langchain для генерации постов."""
import os
import re
import requests
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
    
    def _is_mostly_english(self, text: str) -> bool:
        """Проверяет, написан ли текст в основном на английском."""
        if not text:
            return False
        
        # Подсчитываем кириллические и латинские символы
        cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', text))
        latin_count = len(re.findall(r'[A-Za-z]', text))
        
        # Если латинских символов больше чем кириллических в 2 раза - вероятно английский
        return latin_count > cyrillic_count * 2 and latin_count > 20
    
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
            
            # Проверяем, что используется правильная модель
            model_name = ollama_config.LLM_MODEL
            if "qwen" not in model_name.lower():
                print(f"⚠️ Внимание: используется модель {model_name}, ожидается qwen3:4b")
            
            self.llm = ChatOllama(
                model=model_name,
                base_url=ollama_config.OLLAMA_URL,
                temperature=0.7,  # Температура для креативной генерации
                num_predict=300,  # Достаточно для поста
                repeat_penalty=1.2,
                top_p=0.9
            )
            print(f"✓ LLM инициализирован: {model_name}")
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
        
        # Создаем retriever с ограниченным количеством контекста (кратко)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Форматируем контекст из документов, убирая метаданные и ограничивая длину
        def format_docs(docs: List[Document]) -> str:
            """Форматирует документы, убирая метаданные и ограничивая длину."""
            texts = []
            max_doc_length = 200  # Максимальная длина одного документа
            max_total_length = 500  # Максимальная общая длина контекста
            
            for doc in docs:
                text = doc.page_content
                # Убираем временные метки типа [2025-10-28T14:37:09]
                text = re.sub(r'\[\d{4}-\d{2}-\d{2}T[\d:]+\]', '', text)
                # Убираем имена в начале строки типа "Имя: "
                text = re.sub(r'^[А-Яа-яA-Za-z\s]+:\s*', '', text, flags=re.MULTILINE)
                text = text.strip()
                
                # Ограничиваем длину каждого документа
                if len(text) > max_doc_length:
                    text = text[:max_doc_length] + "..."
                
                if text:
                    texts.append(text)
            
            # Объединяем и ограничиваем общую длину
            result = "\n".join(texts)
            if len(result) > max_total_length:
                # Берем первые документы до лимита
                result = result[:max_total_length].rsplit('\n', 1)[0] + "..."
            
            return result
        
        # Четкий промпт для генерации постов на русском языке
        prompt_template = """Ты креативный копирайтер. Напиши оригинальный пост для социальной сети на русском языке.

Контекст из чата (для понимания темы):
{context}

Тема для поста: {question}

ТРЕБОВАНИЯ:
1. Напиши НОВЫЙ оригинальный пост на русском языке
2. НЕ копируй текст из контекста дословно
3. Используй контекст только для понимания темы
4. Пост должен быть интересным и актуальным
5. Длина: 2-5 предложений
6. Пиши ТОЛЬКО на русском языке

Напиши пост:"""
        
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
    
    def _is_mostly_english(self, text: str) -> bool:
        """Проверяет, написан ли текст в основном на английском."""
        if not text:
            return False
        
        # Подсчитываем кириллические и латинские символы
        cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', text))
        latin_count = len(re.findall(r'[A-Za-z]', text))
        
        # Если латинских символов больше чем кириллических в 2 раза - вероятно английский
        return latin_count > cyrillic_count * 2 and latin_count > 20
    
    def generate_post(self, idea: str) -> str:
        """
        Генерирует пост на основе идеи и контекста из базы знаний.
        
        Args:
            idea: Идея для поста
            
        Returns:
            Сгенерированный пост
            
        Raises:
            ValueError: Если RAG цепочка не инициализирована
        """
        if self.qa_chain is None:
            raise ValueError("RAG цепочка не инициализирована. Сначала создайте или загрузите базу знаний.")
        
        try:
            post = self.qa_chain.invoke(idea)
            
            if not post:
                return "Не удалось сгенерировать пост."
            
            # Проверяем, что ответ на русском языке
            post_clean = post.strip()
            
            # Если ответ на английском, пытаемся исправить
            if self._is_mostly_english(post_clean):
                print("⚠️ Обнаружен ответ на английском, повторяю запрос с усиленным промптом")
                # Повторяем с более жестким промптом
                enhanced_prompt = f"Напиши пост на русском языке на тему: {idea}. Контекст: {post_clean[:200]}"
                post = self.qa_chain.invoke(enhanced_prompt)
                post_clean = post.strip()
            
            # Очистка ответа от артефактов промпта
            
            # Убираем строки с инструкциями и метаданными
            lines = post_clean.split('\n')
            filtered_lines = []
            skip_phrases = [
                'контекст из чата:', 'context from',
                'тема/идея для поста:', 'идея для поста:',
                'важно:', 'important:',
                'пост:', 'post:'
            ]
            
            for line in lines:
                line_lower = line.lower().strip()
                # Пропускаем строки с инструкциями
                if any(phrase in line_lower for phrase in skip_phrases):
                    # Но оставляем контент после двоеточия, если он есть
                    if ':' in line:
                        content_after_colon = line.split(':', 1)[1].strip()
                        if len(content_after_colon) > 20:  # Достаточно длинный контент
                            filtered_lines.append(content_after_colon)
                    continue
                
                # Убираем временные метки
                line = re.sub(r'\[\d{4}-\d{2}-\d{2}T[\d:]+\]', '', line)
                line = line.strip()
                
                if line and len(line) > 5:  # Минимальная длина строки
                    filtered_lines.append(line)
            
            post_clean = '\n'.join(filtered_lines).strip()
            
            # Если после фильтрации ничего не осталось, возвращаем оригинал
            if not post_clean:
                post_clean = post.strip()
            
            # Убираем дубликаты строк
            unique_lines = []
            seen = set()
            for line in post_clean.split('\n'):
                line_stripped = line.strip()
                if line_stripped and line_stripped.lower() not in seen:
                    seen.add(line_stripped.lower())
                    unique_lines.append(line)
            
            post_clean = '\n'.join(unique_lines).strip()
            
            return post_clean
            
        except Exception as e:
            return f"Ошибка при генерации поста: {str(e)}"
    
    def query(self, question: str) -> str:
        """
        Выполняет запрос к RAG системе (алиас для generate_post для обратной совместимости).
        
        Args:
            question: Идея для поста
            
        Returns:
            Сгенерированный пост
        """
        return self.generate_post(question)
    
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
