"""Vector store for message embeddings using ChromaDB."""

import chromadb
from typing import List, Dict, Any, Optional
from chromadb.config import Settings


class MessageVectorStore:
    """Vector store for storing and searching message embeddings."""
    
    def __init__(
        self,
        collection_name: str = "messages",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_messages(
        self,
        messages: List[Dict[str, Any]],
        embeddings_client,
        batch_size: int = 100,
        skip_errors: bool = True,
        progress_callback=None
    ):
        """
        Add messages to the vector store with embeddings.
        
        Args:
            messages: List of message dictionaries
            embeddings_client: OllamaEmbeddings instance
            batch_size: Batch size for processing embeddings
            skip_errors: If True, skip messages that fail to embed
            progress_callback: Optional callback function(current, total) for progress updates
        """
        texts = [msg["text"] for msg in messages]
        ids = [str(msg["id"]) for msg in messages]
        
        # Generate embeddings with error handling
        embeddings = embeddings_client.embed_documents(
            texts,
            skip_errors=skip_errors,
            max_retries=3
        )
        
        # Filter out messages with empty embeddings if skip_errors is True
        valid_indices = []
        for i, emb in enumerate(embeddings):
            if emb:  # Non-empty embedding
                valid_indices.append(i)
        
        if progress_callback:
            progress_callback(len(valid_indices), len(messages))
        
        # Prepare metadata only for valid messages
        valid_messages = [messages[i] for i in valid_indices]
        valid_ids = [ids[i] for i in valid_indices]
        valid_embeddings = [embeddings[i] for i in valid_indices]
        valid_texts = [texts[i] for i in valid_indices]
        
        metadatas = []
        for msg in valid_messages:
            metadata = {
                "date": msg.get("date", ""),
                "from": msg.get("from", ""),
                "from_id": msg.get("from_id", ""),
                "text": msg["text"][:500]  # Limit metadata text length
            }
            metadatas.append(metadata)
        
        # Add to collection in batches
        for i in range(0, len(valid_ids), batch_size):
            batch_ids = valid_ids[i:i + batch_size]
            batch_embeddings = valid_embeddings[i:i + batch_size]
            batch_texts = valid_texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
    
    def search_similar(
        self,
        query: str,
        top_k: int,
        embeddings_client
    ) -> List[Dict[str, Any]]:
        """
        Search for similar messages.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            embeddings_client: OllamaEmbeddings instance
            
        Returns:
            List of similar messages with metadata and scores
        """
        # Generate query embedding
        query_embedding = embeddings_client.embed_query(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        similar_messages = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                msg_dict = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                }
                similar_messages.append(msg_dict)
        
        return similar_messages
    
    def count(self) -> int:
        """
        Get total number of messages in the store.
        
        Returns:
            Number of messages
        """
        return self.collection.count()
    
    def clear(self):
        """Clear all messages from the store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

