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
    
    def _is_mostly_russian(self, text: str) -> bool:
        """Проверяет, написан ли текст в основном на русском."""
        if not text:
            return False
        
        # Подсчитываем кириллические и латинские символы
        cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', text))
        latin_count = len(re.findall(r'[A-Za-z]', text))
        
        # Если кириллических символов больше чем латинских - вероятно русский
        # Или если кириллических символов достаточно много
        return cyrillic_count > latin_count or cyrillic_count > 50
    
    def _is_mostly_english(self, text: str) -> bool:
        """Проверяет, написан ли текст в основном на английском или другом языке."""
        if not text:
            return False
        
        # Подсчитываем кириллические и латинские символы
        cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', text))
        latin_count = len(re.findall(r'[A-Za-z]', text))
        
        # Если латинских символов больше чем кириллических в 2 раза - вероятно не русский
        return latin_count > cyrillic_count * 2 and latin_count > 30
    
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
            
            # Используем модель из конфигурации
            model_name = ollama_config.LLM_MODEL
            
            self.llm = ChatOllama(
                model=model_name,
                base_url=ollama_config.OLLAMA_URL,
                temperature=0.8,  # Температура для креативной генерации
                num_predict=-1,  # Без ограничения длины
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
        
        # Создаем retriever с большим количеством контекста для лучшего поиска смежных слов
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        
        # Форматируем контекст из документов, убирая метаданные
        def format_docs(docs: List[Document]) -> str:
            """Форматирует документы, убирая метаданные для использования в промпте."""
            texts = []
            
            for doc in docs:
                text = doc.page_content
                # Убираем временные метки типа [2025-10-28T14:37:09]
                text = re.sub(r'\[\d{4}-\d{2}-\d{2}T[\d:]+\]', '', text)
                # Убираем имена в начале строки типа "Имя: "
                text = re.sub(r'^[А-Яа-яA-Za-z\s]+:\s*', '', text, flags=re.MULTILINE)
                text = text.strip()
                
                if text and len(text) > 10:  # Минимальная длина
                    texts.append(text)
            
            # Объединяем все релевантные фрагменты
            return "\n".join(texts)
        
        # Очень жесткий промпт с требованием ТОЛЬКО русского языка
        prompt_template = """Напиши пост для социальной сети.

Контекст из чата:
{context}

Тема: {question}

Напиши оригинальный пост на русском языке. Используй контекст для вдохновения. НЕ копируй текст из контекста дословно.

КРИТИЧЕСКИ ВАЖНО: Пиши ТОЛЬКО на русском языке. Запрещено использовать английский, немецкий или другие языки. Только русский язык."""
        
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
    
    def _extract_russian_text(self, text: str) -> str:
        """Извлекает русский текст из многоязычного ответа."""
        if not text:
            return text
        
        lines = text.split('\n')
        russian_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Подсчитываем кириллические символы
            cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', line_stripped))
            latin_count = len(re.findall(r'[A-Za-z]', line_stripped))
            
            # Если в строке больше кириллических символов - это русский текст
            if cyrillic_count > latin_count and cyrillic_count > 10:
                russian_lines.append(line_stripped)
        
        result = '\n'.join(russian_lines).strip()
        return result if result else text.strip()
    
    def _clean_post_response(self, text: str) -> str:
        """Очищает ответ от промптов, инструкций и артефактов."""
        if not text:
            return text
        
        lines = text.split('\n')
        filtered_lines = []
        
        # Фразы, которые указывают на промпт или инструкции
        skip_phrases = [
            'контекст из чата:', 'context from', 'релевантный контекст',
            'тема для поста:', 'тема поста:', 'идея для поста:',
            'требования:', 'важно:', 'important:', 'requirements:',
            'напиши пост:', 'пост:', 'post:', 'write a post',
            'напиши оригинальный пост', 'пиши только текст поста',
            'insert background music', 'insert pictures', '[insert',
            'требо', 'треб', 'напиши н', 'пиши т', 'используй контекст',
            'длина:', 'length:', 'должен быть'
        ]
        
        # Проверяем, не является ли весь текст промптом
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['требования:', 'requirements:', 'важно:', 'important:']):
            # Если есть структурированные требования, это скорее всего промпт
            # Ищем начало реального поста после инструкций
            found_start = False
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                line_lower = line_stripped.lower()
                
                # Пропускаем строки с инструкциями
                if any(phrase in line_lower for phrase in skip_phrases):
                    # Но проверяем, есть ли контент после двоеточия
                    if ':' in line_stripped:
                        parts = line_stripped.split(':', 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                            # Если после двоеточия достаточно длинный текст - это может быть пост
                            if len(content) > 30 and not any(phrase in content.lower() for phrase in skip_phrases):
                                filtered_lines.append(content)
                                found_start = True
                    continue
                
                # Убираем временные метки
                line_clean = re.sub(r'\[\d{4}-\d{2}-\d{2}T[\d:]+\]', '', line_stripped)
                line_clean = re.sub(r'\[.*?\]', '', line_clean)  # Убираем все квадратные скобки
                line_clean = line_clean.strip()
                
                # Пропускаем строки не на русском языке
                cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', line_clean))
                latin_count = len(re.findall(r'[A-Za-z]', line_clean))
                if latin_count > cyrillic_count * 2 and latin_count > 20:
                    continue  # Пропускаем не-русские строки
                
                # Пропускаем очень короткие строки или строки с только пунктами списка
                if len(line_clean) < 10:
                    continue
                
                # Пропускаем строки, которые выглядят как пункты списка (начинаются с цифры или дефиса)
                if re.match(r'^[\d\-•]\s*', line_clean):
                    continue
                
                if line_clean:
                    filtered_lines.append(line_clean)
                    found_start = True
            
            if not found_start:
                # Если не нашли начало поста, возвращаем оригинал без первых строк с инструкциями
                filtered_lines = []
                skip_count = 0
                for line in lines:
                    line_lower = line.lower().strip()
                    if any(phrase in line_lower for phrase in skip_phrases):
                        skip_count += 1
                        continue
                    if skip_count > 0:  # Пропустили инструкции, теперь берем контент
                        line_clean = re.sub(r'\[.*?\]', '', line.strip())
                        if line_clean and len(line_clean) > 10:
                            filtered_lines.append(line_clean)
        else:
            # Обычная очистка без структурированных инструкций
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                line_lower = line_stripped.lower()
                
                # Пропускаем строки с инструкциями
                if any(phrase in line_lower for phrase in skip_phrases):
                    continue
                
                # Убираем временные метки и квадратные скобки
                line_clean = re.sub(r'\[\d{4}-\d{2}-\d{2}T[\d:]+\]', '', line_stripped)
                line_clean = re.sub(r'\[.*?\]', '', line_clean)
                line_clean = line_clean.strip()
                
                # Пропускаем строки не на русском языке
                cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', line_clean))
                latin_count = len(re.findall(r'[A-Za-z]', line_clean))
                if latin_count > cyrillic_count * 2 and latin_count > 20:
                    continue  # Пропускаем не-русские строки
                
                if line_clean and len(line_clean) > 5:
                    filtered_lines.append(line_clean)
        
        result = '\n'.join(filtered_lines).strip()
        
        # Если результат слишком короткий или похож на промпт, возвращаем оригинал
        if len(result) < 20:
            # Пытаемся найти хоть что-то полезное в оригинале
            original_lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in original_lines:
                if len(line) > 30 and not any(phrase in line.lower() for phrase in skip_phrases):
                    result = line
                    break
        
        return result if result else text.strip()
    
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
            
            # Если ответ не на русском, повторяем генерацию с более жестким промптом
            max_retries = 2
            retry_count = 0
            while not self._is_mostly_russian(post_clean) and retry_count < max_retries:
                retry_count += 1
                print(f"⚠️ Обнаружен ответ не на русском языке (попытка {retry_count}/{max_retries}), повторяю запрос")
                # Повторяем с очень жестким промптом на русском
                enhanced_prompt = f"Напиши пост ТОЛЬКО на русском языке. Тема: {idea}. Пиши ТОЛЬКО на русском языке, никаких других языков."
                post = self.qa_chain.invoke(enhanced_prompt)
                post_clean = post.strip()
            
            # Если все еще не на русском, пытаемся извлечь русскую часть
            if not self._is_mostly_russian(post_clean):
                print("⚠️ Ответ все еще не на русском, пытаюсь извлечь русскую часть")
                post_clean = self._extract_russian_text(post_clean)
            
            # Агрессивная очистка ответа от промптов и инструкций
            post_clean = self._clean_post_response(post_clean)
            
            # Финальная проверка - если все еще не на русском, извлекаем только русский текст
            if not self._is_mostly_russian(post_clean):
                print("⚠️ После очистки текст все еще не на русском, извлекаю русскую часть")
                post_clean = self._extract_russian_text(post_clean)
            
            # Если после фильтрации ничего не осталось, возвращаем оригинал
            if not post_clean or len(post_clean.strip()) < 10:
                # Пытаемся найти хоть что-то на русском в оригинале
                russian_part = self._extract_russian_text(post.strip())
                if russian_part and len(russian_part) > 10:
                    post_clean = russian_part
                else:
                    post_clean = post.strip()
            
            # Убираем дубликаты строк
            unique_lines = []
            seen = set()
            for line in post_clean.split('\n'):
                line_stripped = line.strip()
                # Пропускаем не-русские строки
                if line_stripped:
                    cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', line_stripped))
                    latin_count = len(re.findall(r'[A-Za-z]', line_stripped))
                    if latin_count > cyrillic_count * 2 and latin_count > 15:
                        continue  # Пропускаем не-русские строки
                
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
