from abc import ABC, abstractmethod
from typing import List
from ..domain.entities.document_chunk import DocumentChunk

class ClusteringService(ABC): # Or Protocol
    """Port for mathematical services like clustering or sampling."""

    @abstractmethod
    async def get_diverse_representatives(
        self,
        chunks_with_embeddings: List[DocumentChunk],
        k: int
    ) -> List[DocumentChunk]:
        """Performs clustering and selects representatives (e.g., centroids)."""
        raise NotImplementedError
