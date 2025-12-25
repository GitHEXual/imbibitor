"""Message retriever for finding similar messages."""

from typing import List, Dict, Any
from .vector_store import MessageVectorStore
from ollama_client import OllamaEmbeddings


class MessageRetriever:
    """Retriever for finding similar messages using vector search."""
    
    def __init__(
        self,
        vector_store: MessageVectorStore,
        embeddings_client: OllamaEmbeddings
    ):
        """
        Initialize message retriever.
        
        Args:
            vector_store: MessageVectorStore instance
            embeddings_client: OllamaEmbeddings instance
        """
        self.vector_store = vector_store
        self.embeddings_client = embeddings_client
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar messages for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar messages with metadata
        """
        return self.vector_store.search_similar(
            query=query,
            top_k=top_k,
            embeddings_client=self.embeddings_client
        )

