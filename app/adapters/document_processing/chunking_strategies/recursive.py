from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Dict, Any
from ....core.domain.entities.document import Document
from ....core.domain.entities.document_chunk import DocumentChunk
from ....core.ports.chunking_strategy import ChunkingStrategy


class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    It tries to split on them in order until the chunks are small enough.
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
        return "recursive"

    async def create_chunks(
            self,
            document: Document,
            structured_content: Dict[str, Any],
            complete_text: str,
    ) -> List[DocumentChunk]:
        """Create chunks"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.target_chunk_size,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([complete_text])

        chunks = []

        for doc in docs:
            chunk = self._create_chunk_from_content(doc, doc.id, doc)
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_content(self, document: Document, chunk_index: int, lang_doc) -> DocumentChunk:
        return DocumentChunk.create(
                document_id=document.id,
                collection_id=document.collection_id,
                content=lang_doc.page_content,
                chunk_index=chunk_index,
                chunk_type="recursive",
                metadata=lang_doc.metadata,
            )
