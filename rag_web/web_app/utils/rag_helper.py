"""Helper functions for working with RAG system."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to import from imbibitor
imbibitor_path = Path(__file__).parent.parent.parent.parent / 'imbibitor'
if str(imbibitor_path) not in sys.path:
    sys.path.insert(0, str(imbibitor_path))

from ollama_client import OllamaEmbeddings, OllamaLLM
from rag_system import MessageVectorStore, MessageRetriever, HybridRetriever, PostGenerator


def get_rag_components(
    collection_name: str,
    ollama_url: str,
    embedding_model: str,
    db_path: str = None
):
    """
    Initialize RAG components for a specific collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        ollama_url: Ollama API base URL
        embedding_model: Embedding model name
        db_path: Path to ChromaDB database directory
    
    Returns:
        Tuple of (vector_store, embeddings_client) or None if error
    """
    try:
        if db_path is None:
            db_path = str(imbibitor_path / 'chroma_db')
        
        # Initialize embeddings client
        embeddings_client = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
        
        # Initialize vector store with specific collection
        vector_store = MessageVectorStore(
            collection_name=collection_name,
            persist_directory=db_path
        )
        
        return vector_store, embeddings_client
    except Exception as e:
        return None, None


def perform_search(
    collection_name: str,
    query: str,
    top_k: int,
    use_hybrid: bool,
    semantic_weight: float,
    keyword_weight: float,
    ollama_url: str,
    embedding_model: str,
    db_path: str = None
) -> List[Dict[str, Any]]:
    """
    Perform RAG search on a collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        query: Search query
        top_k: Number of results to return
        use_hybrid: Whether to use hybrid search
        semantic_weight: Weight for semantic search
        keyword_weight: Weight for keyword search
        ollama_url: Ollama API base URL
        embedding_model: Embedding model name
        db_path: Path to ChromaDB database directory
    
    Returns:
        List of search results with metadata
    """
    vector_store, embeddings_client = get_rag_components(
        collection_name, ollama_url, embedding_model, db_path
    )
    
    if vector_store is None or embeddings_client is None:
        raise Exception("Не удалось инициализировать компоненты RAG")
    
    # Initialize retriever
    if use_hybrid:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embeddings_client=embeddings_client,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        results = retriever.retrieve(query, top_k=top_k, use_hybrid=True)
    else:
        retriever = MessageRetriever(vector_store, embeddings_client)
        results = retriever.retrieve(query, top_k=top_k)
    
    return results


def generate_post(
    collection_name: str,
    topic: str,
    query: str,
    top_k: int,
    use_hybrid: bool,
    semantic_weight: float,
    keyword_weight: float,
    ollama_url: str,
    embedding_model: str,
    llm_model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    db_path: str = None
) -> Dict[str, Any]:
    """
    Generate a post based on search results.
    
    Args:
        collection_name: Name of the ChromaDB collection
        topic: Topic for the post
        query: Search query
        top_k: Number of results to use
        use_hybrid: Whether to use hybrid search
        semantic_weight: Weight for semantic search
        keyword_weight: Weight for keyword search
        ollama_url: Ollama API base URL
        embedding_model: Embedding model name
        llm_model: LLM model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        db_path: Path to ChromaDB database directory
    
    Returns:
        Dictionary with 'post' (generated text) and 'similar_messages' (search results)
    """
    # First, perform search
    similar_messages = perform_search(
        collection_name=collection_name,
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        ollama_url=ollama_url,
        embedding_model=embedding_model,
        db_path=db_path
    )
    
    if not similar_messages:
        raise Exception("Не найдено похожих сообщений для генерации поста")
    
    # Initialize LLM client
    llm_client = OllamaLLM(
        model=llm_model,
        base_url=ollama_url,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Generate post
    generator = PostGenerator(llm_client)
    post = generator.generate_post(
        topic=topic,
        similar_messages=similar_messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return {
        'post': post,
        'similar_messages': similar_messages
    }
