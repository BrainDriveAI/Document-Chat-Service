from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ...core.domain.entities.document import Document
from ...core.domain.entities.document_chunk import DocumentChunk


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""

    @abstractmethod
    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create chunks from structured content"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        pass

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the created chunks"""
        if not chunks:
            return {}

        sizes = [len(chunk.content) for chunk in chunks]
        word_counts = [len(chunk.content.split()) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
            "strategy": self.get_strategy_name()
        }
