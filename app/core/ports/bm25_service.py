from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..domain.entities.document_chunk import DocumentChunk
from ..domain.value_objects.embedding import SearchQuery


class BM25Service(ABC):
    """Port for BM25 keyword search functionality"""

    @abstractmethod
    async def index_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Index chunks for BM25 search"""
        pass

    @abstractmethod
    async def search(
            self,
            query_text: str,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """Search using BM25 algorithm"""
        pass

    @abstractmethod
    async def remove_chunks(self, chunk_ids: List[str]) -> bool:
        """Remove chunks from BM25 index"""
        pass
