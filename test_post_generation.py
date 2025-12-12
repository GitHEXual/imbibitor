#!/usr/bin/env python3
"""Тест генерации постов."""
import sys
from rag_system import RAGSystem

def test_post_generation():
    """Тестирует генерацию поста."""
    print("Инициализация RAG системы...")
    rag = RAGSystem(use_stub_llm=False)
    
    # Проверяем, что модель загружена
    if not rag.llm:
        print("❌ LLM не инициализирован")
        return False
    
    print(f"✓ LLM инициализирован: {type(rag.llm).__name__}")
    
    # Проверяем, что база знаний загружена
    if not rag.is_ready():
        print("⚠️ База знаний не загружена. Загрузите базу знаний сначала.")
        print("Используйте: rag.load_knowledge_base('default')")
        return False
    
    print("✓ База знаний загружена")
    
    # Тестируем генерацию поста
    test_idea = "Хакатон"
    print(f"\nТестирую генерацию поста на тему: '{test_idea}'")
    print("-" * 50)
    
    try:
        post = rag.generate_post(test_idea)
        print(f"\nСгенерированный пост:\n{post}")
        print("-" * 50)
        
        # Проверяем, что пост на русском
        cyrillic_count = len([c for c in post if 'А' <= c <= 'я' or c == 'Ё' or c == 'ё'])
        latin_count = len([c for c in post if 'A' <= c <= 'z'])
        
        print(f"\nСтатистика:")
        print(f"  Кириллических символов: {cyrillic_count}")
        print(f"  Латинских символов: {latin_count}")
        
        if cyrillic_count > latin_count:
            print("✓ Пост в основном на русском языке")
        else:
            print("⚠️ Пост содержит много не-русского текста")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_post_generation()
    sys.exit(0 if success else 1)
