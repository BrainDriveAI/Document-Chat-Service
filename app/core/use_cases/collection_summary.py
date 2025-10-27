from typing import List, Optional

from ..domain.entities.document_chunk import DocumentChunk
from ..ports.vector_store import VectorStore
from ..ports.llm_service import LLMService
from ..ports.clustering_service import ClusteringService
from ..domain.exceptions import DomainException


class CollectionSummaryError(DomainException):
    """Error during collection summary generation"""
    pass


class CollectionSummaryUseCase:
    """Use case for generating collection-level summaries"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        clustering_service: ClusteringService
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.clustering_service = clustering_service  # <-- Store the service

    async def generate_collection_summary(
        self,
        collection_id: str,
        query: Optional[str] = None,
        sample_size: int = 20,
        max_context_chars: int = 8000
    ) -> str:
        """
        Generate a summary of an entire collection.
        
        Strategy:
        1. Get diverse sample of chunks across collection (using clustering service)
        2. Combine sample chunks into context
        3. Generate summary using LLM
        
        Args:
            collection_id: ID of collection to summarize
            query: Optional specific question about the collection
            sample_size: Number of chunks to sample
            max_context_chars: Maximum characters for context
            
        Returns:
            Summary text
        """
        try:
            # Get diverse sample of chunks
            sample_chunks = await self._get_diverse_sample(
                collection_id=collection_id,
                sample_size=sample_size
            )
            
            if not sample_chunks:
                return "No documents found in this collection."
            
            # Build context from samples
            context = self._build_context(sample_chunks, max_context_chars)
            
            # Generate summary
            if query:
                summary = await self._generate_targeted_summary(context, query)
            else:
                summary = await self._generate_general_summary(context, collection_id)
            
            return summary
            
        except Exception as e:
            raise CollectionSummaryError(f"Failed to generate collection summary: {str(e)}")
    
    async def _get_diverse_sample(
        self,
        collection_id: str,
        sample_size: int
    ) -> List[DocumentChunk]:
        """
        Get diverse representative chunks from collection using the ClusteringService.
        """
        try:
            # 1. Get all chunks from collection (with embeddings)
            # Note: This might need pagination for very large collections
            all_chunks = await self.vector_store.get_all_chunks_in_collection(
                collection_id=collection_id,
                limit=1000  # Reasonable limit to avoid memory issues
            )
            
            if not all_chunks:
                return []
            
            # 2. If we have fewer chunks than sample size, return all
            # (The adapter also handles this, but this is a cheap check)
            if len(all_chunks) <= sample_size:
                return all_chunks

            # 3. Delegate clustering and sampling to the adapter
            # The adapter itself handles logic for missing embeddings
            sampled_chunks = await self.clustering_service.get_diverse_representatives(
                chunks_with_embeddings=all_chunks,
                k=sample_size
            )
            
            return sampled_chunks
            
        except Exception as e:
            print(f"Error in diverse sampling, falling back to random: {e}")
            # Fallback: random sample (same as original)
            import random
            all_chunks_fallback = await self.vector_store.get_all_chunks_in_collection(
                collection_id=collection_id,
                limit=sample_size * 2
            )
            if not all_chunks_fallback:
                return []
            return random.sample(all_chunks_fallback, min(sample_size, len(all_chunks_fallback)))
    
    def _build_context(
        self,
        chunks: List[DocumentChunk],
        max_chars: int
    ) -> str:
        """Build context string from chunks, respecting character limit"""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Excerpt {i}]\n{chunk.content}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "\n".join(context_parts)
    
    async def _generate_general_summary(
        self,
        context: str,
        collection_id: str
    ) -> str:
        """Generate a general overview summary of the collection"""
        
        prompt = f"""
            Based on the following excerpts from a document collection, provide a comprehensive summary that covers:

            1. Main topics and themes covered in the collection
            2. Key concepts or subjects discussed
            3. Overall scope and purpose of the documents

            Be concise but comprehensive. Aim for 3-5 paragraphs.

            Document Excerpts:
            {context}

            Collection Summary:
        """

        return await self.llm_service.generate_response(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
    
    async def _generate_targeted_summary(
        self,
        context: str,
        query: str
    ) -> str:
        """Generate a summary focused on answering a specific question"""
        
        prompt = f"""
            Based on the following excerpts from a document collection, answer this question about the collection:

            Question: {query}

            Provide a comprehensive answer based on the excerpts. If the excerpts don't fully answer the question, note what information is available.

            Document Excerpts:
            {context}

            Answer:
        """

        return await self.llm_service.generate_response(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
