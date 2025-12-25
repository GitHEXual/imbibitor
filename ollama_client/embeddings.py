"""Ollama embeddings client."""

import time
from typing import List, Optional, Dict, Any
from .client import OllamaClient


class OllamaEmbeddings:
    """Client for working with Ollama embeddings models."""
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_text_length: int = 8192
    ):
        """
        Initialize Ollama embeddings client.
        
        Args:
            model: Model name for embeddings
            base_url: Base URL of Ollama API
            timeout: Request timeout in seconds
            max_text_length: Maximum text length for embeddings (truncate if longer)
        """
        self.model = model
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.max_text_length = max_text_length
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate text if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return self.client.generate_embeddings(self.model, text)
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 10,
        skip_errors: bool = False,
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in parallel (currently sequential)
            skip_errors: If True, skip documents that fail to embed instead of raising
            max_retries: Maximum retry attempts for each document
            
        Returns:
            List of embedding lists (empty lists for skipped documents if skip_errors=True)
        """
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Use empty embedding for empty texts
                embeddings.append([])
                continue
            
            # Truncate text if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            # Try to embed with retries
            success = False
            for attempt in range(max_retries):
                try:
                    embedding = self.embed_query(text)
                    embeddings.append(embedding)
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Wait before retry (exponential backoff)
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    else:
                        # All retries failed
                        if skip_errors:
                            embeddings.append([])
                            failed_indices.append(i)
                            break
                        else:
                            raise RuntimeError(f"Failed to embed document {i} after {max_retries} attempts: {e}")
            
            if not success and not skip_errors:
                raise RuntimeError(f"Failed to embed document {i}")
        
        if failed_indices and skip_errors:
            import warnings
            warnings.warn(f"Skipped {len(failed_indices)} documents that failed to embed: {failed_indices[:10]}...")
        
        return embeddings

