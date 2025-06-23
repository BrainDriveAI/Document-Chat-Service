import logging
from typing import List, Dict, Any, Optional
from ....core.domain.entities.document import Document
from ....core.domain.entities.document_chunk import DocumentChunk
from ....core.domain.entities.structured_element import StructuredElement
from ....core.ports.chunking_strategy import ChunkingStrategy


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Simple fixed-size chunking strategy.
    Useful for consistent chunk sizes across all documents.
    """

    def __init__(
            self,
            chunk_size: int = 512,
            overlap: int = 50,
            respect_boundaries: bool = True
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries

    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        return "fixed_size"

    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create fixed-size chunks"""

        # Extract text content
        if 'text' in structured_content:
            text = structured_content['text']
        elif 'elements' in structured_content:
            text = '\n\n'.join(
                element.get('text', '')
                for element in structured_content['elements']
                if element.get('text', '').strip()
            )
        else:
            text = str(structured_content)

        # Split into chunks
        chunks = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Respect boundaries if requested
            if self.respect_boundaries and end < len(text):
                # Look for sentence boundary
                boundary_pos = self._find_boundary(text, start + self.chunk_size - 100, end)
                if boundary_pos > start:
                    end = boundary_pos

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk = DocumentChunk.create(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    chunk_type='fixed',
                    metadata={
                        'chunk_size': 'fixed',
                        'word_count': len(chunk_content.split()),
                        'char_count': len(chunk_content),
                        'collection_id': document.collection_id,
                        'document_type': str(document.document_type.value),
                        'chunking_strategy': 'fixed_size',
                        **document.metadata
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next position with overlap
            start = max(start + self.chunk_size - self.overlap, end)

        return chunks

    def _find_boundary(self, text: str, search_start: int, search_end: int) -> int:
        """Find the best boundary within the search range"""
        boundaries = ['. ', '! ', '? ', '\n\n', '\n', '; ']

        best_pos = -1
        for boundary in boundaries:
            pos = text.rfind(boundary, search_start, search_end)
            if pos > best_pos:
                best_pos = pos + len(boundary)

        return best_pos if best_pos > search_start else search_end
