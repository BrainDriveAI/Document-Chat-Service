import logging
from typing import List, Dict, Any
from ....core.domain.entities.document import Document
from ....core.domain.entities.document_chunk import DocumentChunk
from ....core.domain.entities.structured_element import StructuredElement
from ....core.ports.chunking_strategy import ChunkingStrategy

logger = logging.getLogger(__name__)


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """
    Creates both small chunks (for precision) and large chunks (for context).
    Small chunks point to their parent large chunks.

    This strategy is particularly effective for small models as it provides:
    1. Small chunks for precise retrieval
    2. Large chunks for comprehensive context
    3. Hierarchical relationships for better understanding
    """

    def __init__(
            self,
            small_chunk_size: int = 600,
            large_chunk_size: int = 2000,
            overlap: int = 75,
            respect_boundaries: bool = True
    ):
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries  # Respect sentence/paragraph boundaries

    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        return "hierarchical"

    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create hierarchical chunks from structured content"""

        try:
            # Extract structured elements
            elements = self._extract_structured_elements(structured_content)

            # Create large chunks first (parent chunks)
            large_chunks = await self._create_large_chunks(document, elements)

            # Create small chunks that reference their parent large chunks
            small_chunks = await self._create_small_chunks(document, large_chunks)

            # Combine all chunks
            all_chunks = large_chunks + small_chunks

            # Log statistics for monitoring
            stats = self.get_chunk_statistics(all_chunks)
            logger.info(f"Created chunks for document {document.id}: {stats}")

            return all_chunks

        except Exception as e:
            logger.error(f"Error creating chunks for document {document.id}: {str(e)}")
            raise

    def _extract_structured_elements(self, structured_content: Dict[str, Any]) -> List[StructuredElement]:
        """Extract structured elements from spaCy layout processing results"""
        elements = []

        # Handle different structured content formats
        if 'elements' in structured_content:
            # Standard spaCy layout format
            for element in structured_content['elements']:
                elements.append(StructuredElement(
                    content=element.get('text', '').strip(),
                    element_type=element.get('type', 'paragraph'),
                    level=element.get('level'),
                    page_number=element.get('page', 1),
                    bbox=element.get('bbox')
                ))
        elif 'text' in structured_content:
            # Fallback: plain text with basic structure detection
            elements = self._detect_structure_from_text(structured_content['text'])
        else:
            # Last resort: treat as plain text
            text_content = str(structured_content)
            elements = [StructuredElement(
                content=text_content,
                element_type='paragraph',
                page_number=1
            )]

        # Filter out empty elements
        elements = [e for e in elements if e.content and len(e.content.strip()) > 10]

        return elements

    def _detect_structure_from_text(self, text: str) -> List[StructuredElement]:
        """Basic structure detection from plain text"""
        elements = []
        paragraphs = text.split('\n\n')

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Simple heading detection
            if len(para) < 100 and (para.isupper() or para.count('\n') == 0):
                element_type = 'heading'
            elif para.startswith(('•', '-', '*', '1.', '2.', '3.')):
                element_type = 'list'
            else:
                element_type = 'paragraph'

            elements.append(StructuredElement(
                content=para,
                element_type=element_type,
                page_number=1
            ))

        return elements

    async def _create_large_chunks(
            self,
            document: Document,
            elements: List[StructuredElement]
    ) -> List[DocumentChunk]:
        """Create large chunks (parent chunks) from structured elements"""
        large_chunks = []
        current_chunk_content = []
        current_chunk_length = 0
        chunk_index = 0

        current_context = {
            'headings': [],  # Stack of current headings
            'page_number': 1,
            'element_types': []
        }

        for element in elements:
            # Update context
            if element.element_type == 'heading':
                # Manage heading hierarchy
                level = element.level or 1
                current_context['headings'] = current_context['headings'][:level - 1]
                current_context['headings'].append(element.content)

            if element.page_number:
                current_context['page_number'] = element.page_number

            element_length = len(element.content)

            # Check if we need to start a new chunk
            if (current_chunk_length + element_length > self.large_chunk_size and
                    current_chunk_content):

                # Create chunk from current content
                chunk = await self._create_chunk_from_content(
                    document=document,
                    content_parts=current_chunk_content,
                    chunk_index=chunk_index,
                    chunk_type='large',
                    context=current_context.copy()
                )
                large_chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_content = self._get_overlap_content(current_chunk_content, self.overlap)
                    current_chunk_content = overlap_content
                    current_chunk_length = sum(len(part['content']) for part in overlap_content)
                else:
                    current_chunk_content = []
                    current_chunk_length = 0

                chunk_index += 1

            # Add current element to chunk
            current_chunk_content.append({
                'content': element.content,
                'type': element.element_type,
                'page': element.page_number
            })
            current_chunk_length += element_length
            current_context['element_types'].append(element.element_type)

        # Create final chunk if there's remaining content
        if current_chunk_content:
            chunk = await self._create_chunk_from_content(
                document=document,
                content_parts=current_chunk_content,
                chunk_index=chunk_index,
                chunk_type='large',
                context=current_context.copy()
            )
            large_chunks.append(chunk)

        return large_chunks

    def _get_overlap_content(
            self,
            content_parts: List[Dict[str, Any]],
            overlap_chars: int
    ) -> List[Dict[str, Any]]:
        """Get overlap content from the end of current chunk"""
        if not content_parts:
            return []

        # Start from the last part and work backwards
        overlap_parts = []
        current_length = 0

        for part in reversed(content_parts):
            part_length = len(part['content'])
            if current_length + part_length <= overlap_chars:
                overlap_parts.insert(0, part)
                current_length += part_length
            else:
                # Take partial content from this part
                remaining_chars = overlap_chars - current_length
                if remaining_chars > 50:  # Only if we can get meaningful content
                    partial_content = part['content'][-remaining_chars:]
                    overlap_parts.insert(0, {
                        'content': partial_content,
                        'type': part['type'],
                        'page': part.get('page', 1)
                    })
                break

        return overlap_parts

    async def _create_small_chunks(
            self,
            document: Document,
            large_chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Create small chunks from large chunks"""
        small_chunks = []
        small_chunk_index = 0

        for large_chunk in large_chunks:
            # Split large chunk content into smaller pieces
            small_chunk_contents = self._split_text_into_small_chunks(
                large_chunk.content,
                self.small_chunk_size,
                self.overlap
            )

            for i, small_content in enumerate(small_chunk_contents):
                # Create metadata that references the parent chunk
                small_chunk_metadata = large_chunk.metadata.copy()
                small_chunk_metadata.update({
                    'parent_chunk_id': large_chunk.id,
                    'chunk_size': 'small',
                    'sub_chunk_index': i,
                    'total_sub_chunks': len(small_chunk_contents)
                })

                small_chunk = DocumentChunk.create(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    content=small_content,
                    chunk_index=small_chunk_index,
                    chunk_type='small',
                    parent_chunk_id=large_chunk.id,
                    metadata=small_chunk_metadata
                )

                small_chunks.append(small_chunk)
                small_chunk_index += 1

        return small_chunks

    async def _create_chunk_from_content(
            self,
            document: Document,
            content_parts: List[Dict[str, Any]],
            chunk_index: int,
            chunk_type: str,
            context: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a chunk from content parts with rich metadata"""

        # Combine content
        content = '\n\n'.join(part['content'] for part in content_parts)

        # Create rich metadata that includes collection filtering info
        metadata = {
            'chunk_size': chunk_type,
            'element_types': list(set(part['type'] for part in content_parts)),
            'page_numbers': list(set(part.get('page', 1) for part in content_parts)),
            'headings_context': context.get('headings', []),
            'word_count': len(content.split()),
            'char_count': len(content),
            'has_headings': any(part['type'] == 'heading' for part in content_parts),
            'has_lists': any(part['type'] == 'list' for part in content_parts),
            'has_tables': any(part['type'] == 'table' for part in content_parts),
            'collection_id': document.collection_id,  # Important for filtering
            'document_type': str(document.document_type.value),
            'chunking_strategy': 'hierarchical'
        }

        # Add document metadata
        metadata.update(document.metadata)

        return DocumentChunk.create(
            document_id=document.id,
            collection_id=document.collection_id,
            content=content,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            metadata=metadata
        )

    def _split_text_into_small_chunks(
            self,
            text: str,
            chunk_size: int,
            overlap: int
    ) -> List[str]:
        """Split text into smaller chunks with overlap"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary if respecting boundaries
            if self.respect_boundaries and end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position considering overlap
            start = max(start + chunk_size - overlap, end)

            if start >= len(text):
                break

        return chunks

    def _find_sentence_boundary(self, text: str, search_start: int, search_end: int) -> int:
        """Find the best sentence boundary within the search range"""
        sentence_endings = ['.', '!', '?', '\n\n']

        best_pos = -1
        for ending in sentence_endings:
            pos = text.rfind(ending, search_start, search_end)
            if pos > best_pos:
                best_pos = pos + len(ending)

        return best_pos if best_pos > search_start else search_end
