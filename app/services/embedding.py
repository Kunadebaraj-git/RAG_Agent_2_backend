"""Embedding service using fastembed (all-MiniLM-L6-v2, 384 dim) with fallback."""

import logging
import asyncio
import hashlib
import numpy as np
from typing import Optional

from app.config import get_settings

logger = logging.getLogger("docmind")


class EmbeddingService:
    """Embeds text using fastembed all-MiniLM-L6-v2 (384-dim)."""

    def __init__(self):
        settings = get_settings()
        self.dim = settings.HF_EMBEDDING_DIM or 384
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        self.batch_size = 32
        self._init_model()

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info("Fastembed embedding model loaded successfully.")
        except Exception as e:
            logger.warning(f"Fastembed init failed: {e}. Will use semantic-hash fallback.")
            self._model = None

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously."""
        if not texts:
            return []

        if self._model is not None:
            try:
                # Fastembed run in thread pool
                loop = asyncio.get_running_loop()
                embeddings = await loop.run_in_executor(
                    None, lambda: [e.tolist() for e in self._model.embed(texts, batch_size=self.batch_size)]
                )
                return embeddings
            except Exception as e:
                logger.warning(f"Fastembed batch embedding error: {e}. Falling back...")

        # Fallback deterministic embedding
        return [self._hash_embedding(t) for t in texts]

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    def _hash_embedding(self, text: str) -> list[float]:
        """Deterministic 384-dim normalized vector fallback."""
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, w in enumerate(words):
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()


# Singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

