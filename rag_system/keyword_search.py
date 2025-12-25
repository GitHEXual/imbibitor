"""Keyword search using BM25 algorithm."""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class BM25KeywordSearch:
    """BM25 keyword search implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0.0
        self.doc_lengths = []
        self._is_fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words (Russian and English).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract words
        text = text.lower()
        # Match words (Russian and English letters, numbers)
        tokens = re.findall(r'\b[а-яёa-z0-9]+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 model on documents.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        self.doc_lengths = [len(self._tokenize(doc)) for doc in documents]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0
        
        # Calculate document frequencies
        self.doc_freqs = []
        df = Counter()
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            self.doc_freqs.append(Counter(self._tokenize(doc)))
            df.update(tokens)
        
        # Calculate IDF
        n_docs = len(documents)
        for term, freq in df.items():
            self.idf[term] = self._calculate_idf(freq, n_docs)
        
        self._is_fitted = True
    
    def _calculate_idf(self, doc_freq: int, n_docs: int) -> float:
        """
        Calculate IDF for a term.
        
        Args:
            doc_freq: Number of documents containing the term
            n_docs: Total number of documents
            
        Returns:
            IDF value
        """
        if doc_freq == 0:
            return 0.0
        return max(0.0, (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Search query
            
        Returns:
            List of BM25 scores
        """
        if not self._is_fitted:
            raise ValueError("BM25 model must be fitted before scoring")
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[i]
            
            for term in query_tokens:
                if term in self.doc_freqs[i]:
                    tf = self.doc_freqs[i][term]
                    idf = self.idf.get(term, 0.0)
                    
                    # BM25 formula
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_len / self.avg_doc_len)
                    )
                    score += numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search documents and return top-k results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        scores = self.get_scores(query)
        
        # Get top-k indices
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]

