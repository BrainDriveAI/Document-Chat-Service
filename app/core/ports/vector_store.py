from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..domain.entities.document_chunk import DocumentChunk
from ..domain.value_objects.embedding import SearchQuery, EmbeddingVector


class VectorStore(ABC):
    """Port for vector storage and retrieval services"""

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of document chunks with embeddings

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def search_similar(
            self,
            query_embedding: EmbeddingVector,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            collection_id: Optional collection to search within
            top_k: Number of results to return
            filters: Additional metadata filters

        Returns:
            List of similar document chunks
        """
        pass

    @abstractmethod
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get chunks by document id."""
        pass

    @abstractmethod
    async def hybrid_search(
            self,
            query: SearchQuery,
            alpha: float = 0.5  # Weight between dense and sparse search
    ) -> List[DocumentChunk]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query with parameters
            alpha: Weight between dense (0.0) and sparse (1.0) search

        Returns:
            List of relevant document chunks
        """
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        pass

    @abstractmethod
    async def delete_by_collection_id(self, collection_id: str) -> bool:
        """Delete all chunks for a specific collection"""
        pass

    @abstractmethod
    async def get_all_chunks_in_collection(
        self,
        collection_id: str,
        limit: Optional[int] = None,
        include_embeddings: bool = True
    ) -> List[DocumentChunk]:
        """
        Get all chunks in a collection (for collection-level operations).
        
        Args:
            collection_id: Collection ID
            limit: Maximum number of chunks to return
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        pass

