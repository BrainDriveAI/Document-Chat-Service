from dataclasses import dataclass
from typing import Dict, List, Optional
import uuid


@dataclass
class DocumentChunk:
    """Domain entity representing a document chunk with embeddings"""
    id: str
    document_id: str
    collection_id: str
    content: str
    chunk_index: int
    chunk_type: str  # paragraph, heading, table, list
    parent_chunk_id: Optional[str]  # For hierarchical chunking
    metadata: Dict
    embedding_vector: Optional[List[float]] = None
    
    @classmethod
    def create(
        cls,
        document_id: str,
        collection_id: str,
        content: str,
        chunk_index: int,
        chunk_type: str = "paragraph",
        parent_chunk_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> "DocumentChunk":
        """Factory method to create a new chunk"""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            collection_id=collection_id,
            content=content,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            parent_chunk_id=parent_chunk_id,
            metadata=metadata or {}
        )
    