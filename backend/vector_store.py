"""
Vector Store
Lightweight document retrieval using TF-IDF + BM25-style scoring.
No external vector DB required. Falls back to keyword search gracefully.

For production, swap with ChromaDB, Pinecone, or Qdrant.
"""

import os
import re
import json
import math
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent keyword-based vector store using TF-IDF scoring.
    Stores chunks per user in pickle files.
    Optionally uses sentence-transformers for semantic search if installed.
    """

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.pkl"
        self._load_index()
        self._try_load_embedder()

    def _try_load_embedder(self):
        """Try to load sentence-transformers for better semantic search."""
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.np = np
            logger.info("Semantic search enabled via sentence-transformers")
        except ImportError:
            logger.info("sentence-transformers not available; using TF-IDF search")

    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data.get("chunks", [])         # List of chunk dicts
            self.embeddings = data.get("embeddings", []) # Optional numpy arrays
        else:
            self.chunks = []
            self.embeddings = []

    def _save_index(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings
            }, f)

    def add_document(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """Add chunks for a document, removing any previous version."""
        self.remove_document(doc_id)

        new_chunks = []
        new_embeddings = []

        for chunk in chunks:
            entry = {**chunk, "doc_id": doc_id}
            new_chunks.append(entry)

        if self.embedder:
            texts = [c["text"] for c in new_chunks]
            try:
                vecs = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)
                new_embeddings = list(vecs)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
                new_embeddings = [None] * len(new_chunks)
        else:
            new_embeddings = [None] * len(new_chunks)

        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_embeddings)
        self._save_index()

    def remove_document(self, doc_id: str):
        """Remove all chunks for a document."""
        indices_to_keep = [i for i, c in enumerate(self.chunks) if c.get("doc_id") != doc_id]
        self.chunks = [self.chunks[i] for i in indices_to_keep]
        self.embeddings = [self.embeddings[i] for i in indices_to_keep] if self.embeddings else []
        self._save_index()

    def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Return top-k most relevant chunks for query.
        Optionally filter by doc_ids.
        """
        if not self.chunks:
            return []

        # Filter by doc_ids if provided
        if doc_ids:
            candidates = [(i, c) for i, c in enumerate(self.chunks) if c.get("doc_id") in doc_ids]
        else:
            candidates = list(enumerate(self.chunks))

        if not candidates:
            return []

        if self.embedder and any(self.embeddings):
            return self._semantic_search(query, candidates, top_k)
        else:
            return self._tfidf_search(query, candidates, top_k)

    def _semantic_search(self, query: str, candidates, top_k: int) -> List[Dict]:
        query_vec = self.embedder.encode([query])[0]
        scored = []
        for i, chunk in candidates:
            emb = self.embeddings[i] if i < len(self.embeddings) else None
            if emb is not None:
                score = float(self.np.dot(query_vec, emb) / (
                    self.np.linalg.norm(query_vec) * self.np.linalg.norm(emb) + 1e-9
                ))
            else:
                score = self._tfidf_score(query, chunk["text"])
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _tfidf_search(self, query: str, candidates, top_k: int) -> List[Dict]:
        query_terms = self._tokenize(query)
        scored = []
        for _, chunk in candidates:
            score = self._tfidf_score_from_terms(query_terms, chunk["text"])
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _tfidf_score(self, query: str, text: str) -> float:
        return self._tfidf_score_from_terms(self._tokenize(query), text)

    def _tfidf_score_from_terms(self, query_terms: List[str], text: str) -> float:
        """Simple BM25-style scoring."""
        text_terms = self._tokenize(text)
        if not text_terms:
            return 0.0
        term_freq = Counter(text_terms)
        N = len(text_terms)
        score = 0.0
        k1, b = 1.5, 0.75
        avg_len = 500  # approximate

        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf > 0:
                tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * N / avg_len))
                # Simple IDF approximation
                idf = math.log(1 + 1 / (tf + 0.5))
                score += tf_score * idf

        return score

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-z\u00c0-\u024f]{2,}\b", text)
        stopwords = {"the","a","an","is","in","on","at","to","of","for","and","or","but","it","as","by"}
        return [t for t in tokens if t not in stopwords]