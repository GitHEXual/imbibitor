"""Hybrid retriever combining semantic and keyword search."""

from typing import List, Dict, Any, Optional, Tuple
from .vector_store import MessageVectorStore
from .keyword_search import BM25KeywordSearch
from ollama_client import OllamaEmbeddings


class HybridRetriever:
    """Hybrid retriever combining semantic (vector) and keyword (BM25) search."""
    
    def __init__(
        self,
        vector_store: MessageVectorStore,
        embeddings_client: OllamaEmbeddings,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: MessageVectorStore instance
            embeddings_client: OllamaEmbeddings instance
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            rrf_k: RRF constant (higher = more weight to top results)
        """
        self.vector_store = vector_store
        self.embeddings_client = embeddings_client
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        # Initialize BM25 search
        self.bm25 = None
        self._messages_cache = None
        self._bm25_initialized = False
    
    def _initialize_bm25(self):
        """Initialize BM25 index from vector store."""
        if self._bm25_initialized:
            return
        
        # Get all documents from vector store
        collection = self.vector_store.collection
        count = collection.count()
        
        if count == 0:
            self._bm25_initialized = True
            return
        
        # Get all documents (ChromaDB get() without arguments returns all)
        try:
            all_data = collection.get(limit=count)
        except Exception:
            # Fallback: try without limit
            all_data = collection.get()
        
        if not all_data or not all_data.get("documents"):
            self._bm25_initialized = True
            return
        
        documents = all_data["documents"]
        self._messages_cache = {}
        
        # Build message cache
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        
        for i, doc_id in enumerate(ids):
            self._messages_cache[doc_id] = {
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "index": i
            }
        
        # Initialize BM25
        self.bm25 = BM25KeywordSearch()
        self.bm25.fit(documents)
        self._bm25_initialized = True
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Tuple[int, float]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search (list of (index, score))
            top_k: Number of final results
            
        Returns:
            Combined and ranked results
        """
        # Create score dictionaries
        semantic_scores = {}
        keyword_scores = {}
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            msg_id = result["id"]
            # RRF score: 1 / (k + rank)
            semantic_scores[msg_id] = self.semantic_weight / (self.rrf_k + rank)
        
        # Process keyword results
        for rank, (index, score) in enumerate(keyword_results, 1):
            # Find message ID by index
            if self._messages_cache:
                for msg_id, msg_data in self._messages_cache.items():
                    if msg_data["index"] == index:
                        # RRF score: 1 / (k + rank)
                        keyword_scores[msg_id] = self.keyword_weight / (self.rrf_k + rank)
                        break
        
        # Combine scores
        combined_scores = {}
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for msg_id in all_ids:
            combined_scores[msg_id] = (
                semantic_scores.get(msg_id, 0.0) +
                keyword_scores.get(msg_id, 0.0)
            )
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build result list
        results = []
        for msg_id, score in sorted_ids:
            # Try to get from semantic results first (has more metadata)
            found = False
            for sem_result in semantic_results:
                if sem_result["id"] == msg_id:
                    result = sem_result.copy()
                    result["hybrid_score"] = score
                    result["semantic_score"] = semantic_scores.get(msg_id, 0.0)
                    result["keyword_score"] = keyword_scores.get(msg_id, 0.0)
                    results.append(result)
                    found = True
                    break
            
            # If not found in semantic results, get from cache
            if not found and self._messages_cache and msg_id in self._messages_cache:
                msg_data = self._messages_cache[msg_id]
                results.append({
                    "id": msg_id,
                    "text": msg_data["text"],
                    "metadata": msg_data["metadata"],
                    "hybrid_score": score,
                    "semantic_score": semantic_scores.get(msg_id, 0.0),
                    "keyword_score": keyword_scores.get(msg_id, 0.0),
                    "distance": None
                })
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar messages using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: If True, use hybrid search; if False, use only semantic
            
        Returns:
            List of similar messages with metadata and scores
        """
        # Always get semantic results
        semantic_results = self.vector_store.search_similar(
            query=query,
            top_k=top_k * 2,  # Get more for better fusion
            embeddings_client=self.embeddings_client
        )
        
        if not use_hybrid:
            return semantic_results[:top_k]
        
        # Initialize BM25 if needed
        self._initialize_bm25()
        
        if not self.bm25:
            # Fallback to semantic only if BM25 not available
            return semantic_results[:top_k]
        
        # Get keyword search results
        keyword_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Combine using RRF
        hybrid_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=top_k
        )
        
        return hybrid_results

