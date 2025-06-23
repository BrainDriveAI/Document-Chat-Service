from abc import ABC
from typing import List, Dict, Any
from ...core.ports.embedding_service import EmbeddingService
from ...core.domain.value_objects.embedding import EmbeddingVector


class BaseEmbeddingService(EmbeddingService, ABC):
    """Base class for embedding services with common functionality"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model_info = None

    def _validate_text(self, text: str) -> str:
        """Validate and clean text for embedding"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Clean the text
        cleaned = text.strip()

        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())

        # Truncate if too long (most embedding models have limits)
        max_length = 8000  # Conservative limit
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]

        return cleaned

    def _validate_batch(self, texts: List[str]) -> List[str]:
        """Validate and clean a batch of texts"""
        if not texts:
            raise ValueError("Text batch cannot be empty")

        if len(texts) > 100:  # Reasonable batch size limit
            raise ValueError("Batch size too large")

        return [self._validate_text(text) for text in texts]
