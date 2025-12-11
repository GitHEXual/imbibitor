"""Тест RAG системы на реальном файле result.json."""
import sys
import json
from parser import parse_telegram_export
from rag_system import RAGSystem


def test_parse_result_json():
    """Тест парсинга result.json."""
    print("=" * 60)
    print("ТЕСТ 1: Парсинг result.json")
    print("=" * 60)
    
    try:
        with open('result.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        messages = parse_telegram_export(json_data)
        print(f"✓ Успешно распарсено сообщений: {len(messages)}")
        
        if messages:
            print(f"\nПримеры сообщений:")
            for i, msg in enumerate(messages[:5], 1):
                print(f"  {i}. {msg[:80]}...")
        
        return messages
    except Exception as e:
        print(f"✗ Ошибка при парсинге: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_build_knowledge_base(messages):
    """Тест создания базы знаний."""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Создание базы знаний")
    print("=" * 60)
    
    if not messages:
        print("✗ Нет сообщений для создания базы знаний")
        return None
    
    rag = RAGSystem(use_stub_llm=True)
    print(f"✓ RAG система создана (заглушка LLM)")
    
    try:
        print(f"Создаю базу знаний из {len(messages)} сообщений...")
        rag.build_knowledge_base(messages)
        
        stats = rag.get_stats()
        print(f"✓ База знаний создана успешно!")
        print(f"  - Векторное хранилище: {stats['has_vectorstore']}")
        print(f"  - RAG цепочка: {stats['has_qa_chain']}")
        print(f"  - Система готова: {stats['is_ready']}")
        
        return rag
    except ConnectionError as e:
        print(f"⚠️ Ollama недоступен для создания эмбеддингов:")
        print(f"   {e}")
        print("\nДля создания базы знаний нужно:")
        print("1. Запустить Ollama: ollama serve")
        print("2. Установить модель эмбеддингов: ollama pull nomic-embed-text")
        return None
    except Exception as e:
        print(f"✗ Ошибка при создании базы знаний: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_queries(rag):
    """Тест запросов к RAG системе."""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Запросы к RAG системе")
    print("=" * 60)
    
    if not rag or not rag.is_ready():
        print("⚠️ Система не готова, пропускаем тест запросов")
        return
    
    # Тестовые вопросы
    test_questions = [
        "О чем этот чат?",
        "Какие основные темы обсуждались?",
        "Что было важного?",
    ]
    
    for question in test_questions:
        print(f"\nВопрос: {question}")
        print("-" * 60)
        try:
            answer = rag.query(question)
            print(f"Ответ: {answer[:300]}...")
        except Exception as e:
            print(f"✗ Ошибка: {e}")


def test_save_load(rag):
    """Тест сохранения и загрузки."""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Сохранение и загрузка базы знаний")
    print("=" * 60)
    
    if not rag or not rag.is_ready():
        print("⚠️ Система не готова, пропускаем тест сохранения")
        return
    
    try:
        rag.save_knowledge_base("result_test")
        print("✓ База знаний сохранена")
        
        # Создаем новую систему и загружаем
        rag2 = RAGSystem(use_stub_llm=True)
        if rag2.load_knowledge_base("result_test"):
            print("✓ База знаний загружена")
            print(f"  - Система готова: {rag2.is_ready()}")
        else:
            print("⚠️ Не удалось загрузить базу знаний")
    except Exception as e:
        print(f"✗ Ошибка: {e}")


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ RAG СИСТЕМЫ НА result.json")
    print("=" * 60 + "\n")
    
    # Тест 1: Парсинг
    messages = test_parse_result_json()
    
    if not messages:
        print("\n✗ Не удалось распарсить файл. Тестирование остановлено.")
        return
    
    # Тест 2: Создание базы знаний
    rag = test_build_knowledge_base(messages)
    
    # Тест 3: Запросы (только если база знаний создана)
    if rag and rag.is_ready():
        test_queries(rag)
        test_save_load(rag)
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

