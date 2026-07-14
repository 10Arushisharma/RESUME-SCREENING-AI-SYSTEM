from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service responsible for generating and caching text embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self._model: Any | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            logger.warning("sentence-transformers is unavailable: %s", _IMPORT_ERROR)
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model %s", self.model_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Falling back to lightweight embeddings because model loading failed: %s", exc)
            self._model = None
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            logger.warning("Embedding requested for empty text; returning zero vector")
            return np.zeros(self.settings.embedding_dimensions, dtype=np.float32)
        if text in self._cache:
            logger.debug("Using cached embedding for text length=%d", len(text))
            return self._cache[text]

        model = self._load_model()
        if model is not None:
            logger.debug("Encoding text with SentenceTransformer model length=%d", len(text))
            vector = np.asarray(model.encode(text), dtype=np.float32)
        else:
            logger.debug("Encoding text with fallback embedding length=%d", len(text))
            vector = self._fallback_embedding(text)
        vector = self._normalize_vector(vector)
        self._cache[text] = vector
        logger.debug("Generated embedding norm=%s shape=%s", np.linalg.norm(vector), vector.shape)
        return vector

    def _fallback_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.settings.embedding_dimensions, dtype=np.float32)
        tokens = [token for token in re.split(r"\W+", text.lower()) if token]
        for token in tokens:
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.settings.embedding_dimensions
            vector[index] += 1.0
        return vector

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros(self.settings.embedding_dimensions, dtype=np.float32)
        return vector / norm
