"""Local embedding service using sentence-transformers with GPU acceleration."""

import logging
import os
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Generates text embeddings locally using sentence-transformers.

    The model is loaded once and kept in memory for fast inference.
    Supports GPU (CUDA), Apple Silicon (MPS), and CPU fallback.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
        )
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.device = device or self._resolve_device()
        self.model = self._load_model()

    def _resolve_device(self) -> str:
        env_device = os.getenv("EMBEDDING_DEVICE")
        if env_device:
            return env_device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> SentenceTransformer:
        logger.info(
            "Loading embedding model %s on %s", self.model_name, self.device
        )
        model = SentenceTransformer(self.model_name, device=self.device)
        return model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string. Returns a list of floats."""
        embedding = self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in a batch. Returns list of float lists."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()
