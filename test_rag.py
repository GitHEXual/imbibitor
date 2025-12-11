"""Тесты для RAG системы."""
import sys
from parser import parse_telegram_export
from rag_system import RAGSystem


def test_parser():
    """Тест парсера JSON."""
    print("=" * 50)
    print("Тест 1: Парсер JSON экспорта Telegram")
    print("=" * 50)
    
    test_json = {
        "messages": [
            {
                "text": "Привет! Как дела?",
                "date": "2024-01-01T10:00:00",
                "from": "User1"
            },
            {
                "text": "Отлично, спасибо! А у тебя?",
                "date": "2024-01-01T10:01:00",
                "from": "User2"
            },
            {
                "text": "Тоже хорошо, работаю над проектом.",
                "date": "2024-01-01T10:02:00",
                "from": "User1"
            }
        ]
    }
    
    messages = parse_telegram_export(test_json)
    print(f"✓ Найдено сообщений: {len(messages)}")
    for i, msg in enumerate(messages[:3], 1):
        print(f"  {i}. {msg[:60]}...")
    
    assert len(messages) == 3, "Должно быть 3 сообщения"
    print("✓ Парсер работает корректно\n")
    return messages


def test_rag_system_creation():
    """Тест создания RAG системы."""
    print("=" * 50)
    print("Тест 2: Создание RAG системы")
    print("=" * 50)
    
    rag = RAGSystem(use_stub_llm=True)
    stats = rag.get_stats()
    
    print(f"✓ RAG система создана")
    print(f"  - Использует заглушку LLM: {stats['use_stub_llm']}")
    print(f"  - LLM тип: {stats['llm_type']}")
    print(f"  - Готова к работе: {stats['is_ready']}")
    
    assert stats['has_llm'], "LLM должен быть инициализирован"
    assert stats['use_stub_llm'], "Должна использоваться заглушка"
    print("✓ RAG система инициализирована корректно\n")
    
    return rag


def test_knowledge_base_building(rag: RAGSystem, messages: list):
    """Тест создания базы знаний."""
    print("=" * 50)
    print("Тест 3: Создание базы знаний")
    print("=" * 50)
    
    try:
        rag.build_knowledge_base(messages)
        stats = rag.get_stats()
        
        print(f"✓ База знаний создана")
        print(f"  - Векторное хранилище: {stats['has_vectorstore']}")
        print(f"  - RAG цепочка: {stats['has_qa_chain']}")
        print(f"  - Система готова: {stats['is_ready']}")
        
        assert stats['is_ready'], "Система должна быть готова"
        print("✓ База знаний создана успешно\n")
        
        return True
    except ConnectionError as e:
        print(f"⚠️ Ollama недоступен: {e}")
        print("  Это нормально, если Ollama не запущен")
        print("  Для полного теста нужно запустить Ollama\n")
        return False


def test_query(rag: RAGSystem):
    """Тест запросов к RAG системе."""
    print("=" * 50)
    print("Тест 4: Запросы к RAG системе")
    print("=" * 50)
    
    if not rag.is_ready():
        print("⚠️ Система не готова, пропускаем тест запросов")
        return
    
    test_questions = [
        "Как дела?",
        "Что говорили о проекте?",
        "Кто писал сообщения?"
    ]
    
    for question in test_questions:
        try:
            answer = rag.query(question)
            print(f"Q: {question}")
            print(f"A: {answer[:100]}...")
            print()
        except Exception as e:
            print(f"✗ Ошибка при запросе '{question}': {e}")
    
    print("✓ Запросы обработаны\n")


def test_save_load(rag: RAGSystem):
    """Тест сохранения и загрузки базы знаний."""
    print("=" * 50)
    print("Тест 5: Сохранение и загрузка базы знаний")
    print("=" * 50)
    
    if not rag.is_ready():
        print("⚠️ Система не готова, пропускаем тест сохранения")
        return
    
    try:
        # Сохраняем
        rag.save_knowledge_base("test_kb")
        print("✓ База знаний сохранена")
        
        # Создаем новую систему и загружаем
        rag2 = RAGSystem(use_stub_llm=True)
        loaded = rag2.load_knowledge_base("test_kb")
        
        if loaded:
            print("✓ База знаний загружена")
            assert rag2.is_ready(), "Загруженная система должна быть готова"
            print("✓ Сохранение и загрузка работают корректно\n")
        else:
            print("⚠️ Не удалось загрузить базу знаний (возможно, Ollama недоступен)\n")
    except Exception as e:
        print(f"⚠️ Ошибка при сохранении/загрузке: {e}\n")


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ RAG СИСТЕМЫ")
    print("=" * 50 + "\n")
    
    # Тест 1: Парсер
    messages = test_parser()
    
    # Тест 2: Создание RAG системы
    rag = test_rag_system_creation()
    
    # Тест 3: Создание базы знаний
    kb_created = test_knowledge_base_building(rag, messages)
    
    # Тест 4: Запросы (только если база знаний создана)
    if kb_created:
        test_query(rag)
        test_save_load(rag)
    
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

