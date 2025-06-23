from typing import List, Dict, Any
from ....core.domain.entities.document import Document
from ....core.domain.entities.document_chunk import DocumentChunk
from ....core.ports.chunking_strategy import ChunkingStrategy
from .hierarchical import HierarchicalChunkingStrategy


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Chunks documents based on semantic similarity and structure.
    Useful for maintaining topical coherence within chunks.
    """

    def __init__(
            self,
            target_chunk_size: int = 512,
            similarity_threshold: float = 0.7,
            min_chunk_size: int = 100,
            max_chunk_size: int = 1024
    ):
        self.target_chunk_size = target_chunk_size
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        return "semantic"

    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create semantically coherent chunks"""

        # For now, fall back to a simpler structure-aware approach
        # In a full implementation, you'd use embeddings to compute similarity
        hierarchical_strategy = HierarchicalChunkingStrategy(
            small_chunk_size=self.target_chunk_size,
            large_chunk_size=self.max_chunk_size
        )

        return await hierarchical_strategy.create_chunks(document, structured_content)
