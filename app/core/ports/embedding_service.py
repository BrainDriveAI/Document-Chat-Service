from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..domain.value_objects.embedding import EmbeddingVector


class EmbeddingService(ABC):
    """Port for embedding generation services"""

    @abstractmethod
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingVector containing the embedding and metadata
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingVector objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the embedding model"""
        pass
