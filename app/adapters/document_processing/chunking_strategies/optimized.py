import logging
from typing import List, Dict, Any, Optional
from ....core.domain.entities.document import Document
from ....core.domain.entities.document_chunk import DocumentChunk
from ....core.domain.entities.structured_element import StructuredElement
from ....core.ports.chunking_strategy import ChunkingStrategy
from ....core.ports.token_service import TokenService

logger = logging.getLogger(__name__)


class OptimizedHierarchicalChunkingStrategy(ChunkingStrategy):
    """
    Token-aware hierarchical chunking optimized for 8B parameter models.

    This strategy creates:
    1. Small chunks (128-192 tokens) for precise retrieval
    2. Large chunks (384-512 tokens) for comprehensive context
    3. Respects document structure and semantic boundaries
    4. Optimized for mxbai-embed-large (max 512 tokens)
    """

    def __init__(
            self,
            token_service: TokenService,
            small_chunk_tokens: int = 160,  # Sweet spot for precision
            large_chunk_tokens: int = 448,  # Fits well in 8B model context
            overlap_tokens: int = 40,  # Semantic overlap
            max_embedding_tokens: int = 480,  # Safe limit for mxbai-embed-large
            respect_boundaries: bool = True,
            min_chunk_tokens: int = 32  # Minimum viable chunk size
    ):
        self.token_service = token_service
        self.small_chunk_tokens = small_chunk_tokens
        self.large_chunk_tokens = large_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.max_embedding_tokens = max_embedding_tokens
        self.respect_boundaries = respect_boundaries
        self.min_chunk_tokens = min_chunk_tokens

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate chunking configuration"""
        if self.small_chunk_tokens >= self.large_chunk_tokens:
            raise ValueError("Small chunk tokens must be less than large chunk tokens")

        if self.overlap_tokens >= self.small_chunk_tokens:
            raise ValueError("Overlap tokens must be less than small chunk tokens")

        if self.large_chunk_tokens > self.max_embedding_tokens:
            logger.warning(
                f"Large chunk tokens ({self.large_chunk_tokens}) exceeds "
                f"max embedding tokens ({self.max_embedding_tokens}). "
                "This may cause embedding truncation."
            )

    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        return "optimized_hierarchical"

    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create optimized hierarchical chunks from structured content"""
        try:
            # Extract structured elements
            elements = self._extract_structured_elements(structured_content)

            # Create large chunks first (parent chunks)
            large_chunks = await self._create_large_chunks(document, elements)

            # Create small chunks that reference their parent large chunks
            small_chunks = await self._create_small_chunks(document, large_chunks)

            # Validate chunk token counts
            self._validate_chunk_tokens(large_chunks + small_chunks)

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

        # Handle spaCy layout sections
        if 'sections' in structured_content:
            for section in structured_content['sections']:
                content = section.get('text', '').strip()
                if content and self.token_service.count_tokens(content) >= self.min_chunk_tokens:
                    elements.append(StructuredElement(
                        content=content,
                        element_type=self._determine_element_type(section),
                        level=self._extract_heading_level(section),
                        page_number=section.get('layout_info', {}).get('page', 1),
                        bbox=section.get('layout_info', {}).get('bbox')
                    ))

        # Handle tables separately
        if 'tables' in structured_content:
            for table in structured_content['tables']:
                content = table.get('text', '').strip()
                if content and self.token_service.count_tokens(content) >= self.min_chunk_tokens:
                    elements.append(StructuredElement(
                        content=content,
                        element_type='table',
                        page_number=table.get('layout_info', {}).get('page', 1),
                        bbox=table.get('layout_info', {}).get('bbox')
                    ))

        # Fallback to full text if no sections found
        if not elements and 'full_text' in structured_content:
            text_content = structured_content['full_text'].strip()
            if text_content:
                elements = self._split_text_into_elements(text_content)

        # Filter elements by minimum token count
        filtered_elements = [
            e for e in elements
            if self.token_service.count_tokens(e.content) >= self.min_chunk_tokens
        ]

        logger.info(f"Extracted {len(filtered_elements)} structured elements")
        return filtered_elements

    def _determine_element_type(self, section: Dict[str, Any]) -> str:
        """Determine element type from section information"""
        label = section.get('label', '').lower()
        text = section.get('text', '')

        # Check for heading patterns
        if any(heading_word in label for heading_word in ['title', 'heading', 'header']):
            return 'heading'

        # Check for list patterns
        if any(list_word in label for list_word in ['list', 'item', 'bullet']):
            return 'list'

        # Simple heuristics based on text
        if len(text) < 100 and (text.isupper() or text.count('\n') == 0):
            return 'heading'
        elif text.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
            return 'list'

        return 'paragraph'

    def _extract_heading_level(self, section: Dict[str, Any]) -> Optional[int]:
        """Extract heading level from section"""
        label = section.get('label', '').lower()
        heading = section.get('heading', '')

        # Try to extract level from label
        for i in range(1, 7):
            if f'h{i}' in label or f'heading{i}' in label:
                return i

        # Try to infer from heading text patterns
        if heading and isinstance(heading, str):
            # Count leading # symbols (markdown style)
            if heading.startswith('#'):
                return min(heading.count('#'), 6)

        return None

    def _split_text_into_elements(self, text: str) -> List[StructuredElement]:
        """Split plain text into structured elements"""
        elements = []
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para or self.token_service.count_tokens(para) < self.min_chunk_tokens:
                continue

            # Simple structure detection
            element_type = 'paragraph'
            if len(para) < 200 and (para.isupper() or '\n' not in para):
                element_type = 'heading'
            elif para.startswith(('•', '-', '*', '1.', '2.', '3.')):
                element_type = 'list'

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
        current_chunk_elements = []
        current_token_count = 0
        chunk_index = 0

        # Track document context
        context = {
            'headings': [],  # Hierarchical heading stack
            'page_number': 1,
            'element_types': [],
            'tables_count': 0,
            'lists_count': 0
        }

        for element in elements:
            # Update context
            self._update_context(context, element)

            element_tokens = self.token_service.count_tokens(element.content)

            # Check if adding this element would exceed token limit
            if (current_token_count + element_tokens > self.large_chunk_tokens
                    and current_chunk_elements):
                # Create chunk from current elements
                chunk = await self._create_chunk_from_elements(
                    document=document,
                    elements=current_chunk_elements,
                    chunk_index=chunk_index,
                    chunk_type='large',
                    context=context.copy()
                )
                large_chunks.append(chunk)

                # Start new chunk with overlap
                overlap_elements = self._get_overlap_elements(
                    current_chunk_elements,
                    self.overlap_tokens
                )

                current_chunk_elements = overlap_elements
                current_token_count = sum(
                    self.token_service.count_tokens(elem.content)
                    for elem in overlap_elements
                )
                chunk_index += 1

            # Add current element
            current_chunk_elements.append(element)
            current_token_count += element_tokens

        # Create final chunk
        if current_chunk_elements:
            chunk = await self._create_chunk_from_elements(
                document=document,
                elements=current_chunk_elements,
                chunk_index=chunk_index,
                chunk_type='large',
                context=context.copy()
            )
            large_chunks.append(chunk)

        return large_chunks

    def _update_context(self, context: Dict[str, Any], element: StructuredElement):
        """Update document context with current element"""
        if element.element_type == 'heading':
            level = element.level or 1
            # Maintain heading hierarchy
            context['headings'] = context['headings'][:level - 1]
            context['headings'].append(element.content[:100])  # Truncate long headings

        if element.page_number:
            context['page_number'] = element.page_number

        context['element_types'].append(element.element_type)

        if element.element_type == 'table':
            context['tables_count'] += 1
        elif element.element_type == 'list':
            context['lists_count'] += 1

    def _get_overlap_elements(
            self,
            elements: List[StructuredElement],
            max_overlap_tokens: int
    ) -> List[StructuredElement]:
        """Get elements from the end for overlap"""
        if not elements or max_overlap_tokens <= 0:
            return []

        overlap_elements = []
        current_tokens = 0

        # Start from the end and work backwards
        for element in reversed(elements):
            element_tokens = self.token_service.count_tokens(element.content)
            if current_tokens + element_tokens <= max_overlap_tokens:
                overlap_elements.insert(0, element)
                current_tokens += element_tokens
            else:
                break

        return overlap_elements

    async def _create_chunk_from_elements(
            self,
            document: Document,
            elements: List[StructuredElement],
            chunk_index: int,
            chunk_type: str,
            context: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a chunk from structured elements with rich metadata"""
        # Combine element content intelligently
        content_parts = []
        for element in elements:
            if element.element_type == 'heading':
                content_parts.append(f"\n## {element.content}\n")
            elif element.element_type == 'table':
                content_parts.append(f"\n[TABLE]\n{element.content}\n")
            else:
                content_parts.append(element.content)

        content = "\n".join(content_parts).strip()

        # Create comprehensive metadata
        metadata = {
            'chunk_size': chunk_type,
            'token_count': self.token_service.count_tokens(content),
            'element_types': list(set(e.element_type for e in elements)),
            'page_numbers': list(set(e.page_number for e in elements if e.page_number)),
            'headings_context': context.get('headings', [])[-3:],  # Last 3 headings
            'has_headings': any(e.element_type == 'heading' for e in elements),
            'has_lists': any(e.element_type == 'list' for e in elements),
            'has_tables': any(e.element_type == 'table' for e in elements),
            'element_count': len(elements),
            'collection_id': document.collection_id,
            'document_type': str(document.document_type.value),
            'chunking_strategy': 'optimized_hierarchical',
            'overlap_tokens': self.overlap_tokens,
            'structure_preserved': True
        }

        # Add document metadata
        metadata.update(document.metadata or {})

        return DocumentChunk.create(
            document_id=document.id,
            collection_id=document.collection_id,
            content=content,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            metadata=metadata
        )

    async def _create_small_chunks(
            self,
            document: Document,
            large_chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Create small chunks from large chunks using token-aware splitting"""
        small_chunks = []
        small_chunk_index = 0

        for large_chunk in large_chunks:
            # Split large chunk content into smaller token-based pieces
            small_contents = self.token_service.split_text_by_tokens(
                text=large_chunk.content,
                max_tokens=self.small_chunk_tokens,
                overlap_tokens=self.overlap_tokens
            )

            for i, small_content in enumerate(small_contents):
                # Ensure minimum token count
                if self.token_service.count_tokens(small_content) < self.min_chunk_tokens:
                    continue

                # Create metadata referencing parent chunk
                small_metadata = large_chunk.metadata.copy()
                small_metadata.update({
                    'parent_chunk_id': large_chunk.id,
                    'chunk_size': 'small',
                    'sub_chunk_index': i,
                    'total_sub_chunks': len(small_contents),
                    'token_count': self.token_service.count_tokens(small_content),
                    'derived_from_large': True
                })

                small_chunk = DocumentChunk.create(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    content=small_content,
                    chunk_index=small_chunk_index,
                    chunk_type='small',
                    parent_chunk_id=large_chunk.id,
                    metadata=small_metadata
                )

                small_chunks.append(small_chunk)
                small_chunk_index += 1

        return small_chunks

    def _validate_chunk_tokens(self, chunks: List[DocumentChunk]):
        """Validate that chunks don't exceed token limits"""
        for chunk in chunks:
            token_count = self.token_service.count_tokens(chunk.content)

            if token_count > self.max_embedding_tokens:
                logger.warning(
                    f"Chunk {chunk.id} has {token_count} tokens, "
                    f"exceeding max embedding tokens ({self.max_embedding_tokens})"
                )

            # Update metadata with actual token count
            if chunk.metadata:
                chunk.metadata['validated_token_count'] = token_count

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get comprehensive statistics about the created chunks"""
        if not chunks:
            return {}

        # Separate small and large chunks
        small_chunks = [c for c in chunks if c.chunk_type == 'small']
        large_chunks = [c for c in chunks if c.chunk_type == 'large']

        # Calculate token statistics
        all_tokens = [self.token_service.count_tokens(c.content) for c in chunks]
        small_tokens = [self.token_service.count_tokens(c.content) for c in small_chunks]
        large_tokens = [self.token_service.count_tokens(c.content) for c in large_chunks]

        stats = {
            "total_chunks": len(chunks),
            "small_chunks": len(small_chunks),
            "large_chunks": len(large_chunks),
            "strategy": self.get_strategy_name(),

            # Token statistics
            "avg_tokens": sum(all_tokens) / len(all_tokens) if all_tokens else 0,
            "min_tokens": min(all_tokens) if all_tokens else 0,
            "max_tokens": max(all_tokens) if all_tokens else 0,
            "total_tokens": sum(all_tokens),

            # Small chunk statistics
            "small_avg_tokens": sum(small_tokens) / len(small_tokens) if small_tokens else 0,
            "small_min_tokens": min(small_tokens) if small_tokens else 0,
            "small_max_tokens": max(small_tokens) if small_tokens else 0,

            # Large chunk statistics
            "large_avg_tokens": sum(large_tokens) / len(large_tokens) if large_tokens else 0,
            "large_min_tokens": min(large_tokens) if large_tokens else 0,
            "large_max_tokens": max(large_tokens) if large_tokens else 0,

            # Content analysis
            "chunk_types": list(set(c.chunk_type for c in chunks)),
            "element_types": list(set(
                elem_type for c in chunks
                for elem_type in c.metadata.get('element_types', [])
            )),
            "has_hierarchical_structure": len(large_chunks) > 0 and len(small_chunks) > 0,
            "avg_elements_per_chunk": sum(
                c.metadata.get('element_count', 1) for c in chunks
            ) / len(chunks) if chunks else 0,

            # Validation flags
            "chunks_exceeding_embedding_limit": sum(
                1 for tokens in all_tokens if tokens > self.max_embedding_tokens
            ),
            "chunks_below_minimum": sum(
                1 for tokens in all_tokens if tokens < self.min_chunk_tokens
            )
        }

        return stats
