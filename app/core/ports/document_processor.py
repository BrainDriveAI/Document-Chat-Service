from abc import ABC, abstractmethod
from typing import List, Tuple
from ..domain.entities.document import Document
from ..domain.entities.document_chunk import DocumentChunk


class DocumentProcessor(ABC):
    """Port for document processing services"""

    @abstractmethod
    async def process_document(self, document: Document) -> Tuple[List[DocumentChunk], str]:
        """
        Process a document and return structured chunks.

        Args:
            document: Document entity to process

        Returns:
            List of document chunks with structure and metadata,
            and complete text
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return list of supported document types"""
        pass
