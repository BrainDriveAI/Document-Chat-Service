# app/core/use_cases/search_documents.py

import asyncio
from typing import List, Optional, Dict
from ..domain.entities.document_chunk import DocumentChunk
from ..domain.entities.query_transformation import QueryTransformationMethod
from ..domain.value_objects.embedding import SearchQuery
from ..ports.embedding_service import EmbeddingService
from ..ports.vector_store import VectorStore
from ..ports.bm25_service import BM25Service
from ..ports.rank_fusion_service import RankFusionService
from .query_transformation import QueryTransformationUseCase
from .intent_classification import IntentClassificationUseCase, IntentKind
from .collection_summary import CollectionSummaryUseCase


class SearchDocumentsUseCase:
    """Use case for searching documents with advanced query processing"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        bm25_service: BM25Service,
        rank_fusion_service: RankFusionService,
        query_transformation_use_case: QueryTransformationUseCase,
        intent_classification_use_case: IntentClassificationUseCase,
        collection_summary_use_case: CollectionSummaryUseCase,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_service = bm25_service
        self.rank_fusion_service = rank_fusion_service
        self.query_transformation_use_case = query_transformation_use_case
        self.intent_classification_use_case = intent_classification_use_case
        self.collection_summary_use_case = collection_summary_use_case
    
    async def search_documents(
        self,
        query_text: str,
        collection_id: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        alpha: float = 0.5,
        use_intent_classification: bool = True,
        query_transformation_enabled: bool = True,
        query_transformation_methods: Optional[List[QueryTransformationMethod]] = None,
        max_history_turns: int = 3,
    ) -> Dict:
        """
        Search for relevant document chunks with intelligent query processing.
        
        Pipeline:
        1. Classify Intent (optional)
        2. Handle special intents (CHAT, COLLECTION_SUMMARY)
        3. Transform Query (optional)
        4. Execute Retrieval (vector, BM25, or hybrid)
        5. Return results with metadata
        
        Args:
            query_text: Search query
            collection_id: Optional collection to search within
            chat_history: Conversation history for contextualization
            top_k: Number of results to return
            filters: Additional metadata filters
            use_hybrid: Whether to use hybrid search (vector + keyword)
            alpha: Weight for vector vs BM25 (0.5 = equal weight)
            use_intent_classification: Whether to classify intent first
            query_transformation_enabled: Whether to transform queries
            query_transformation_methods: Methods to use for transformation
            max_history_turns: Max conversation turns to consider
            
        Returns:
            Dictionary with:
                - chunks: List of DocumentChunk
                - intent: Detected intent (if classification enabled)
                - transformed_queries: List of queries used (if transformation enabled)
                - metadata: Additional search metadata
        """
        
        # Step 1: Intent Classification
        intent = None
        if use_intent_classification:
            intent = await self.intent_classification_use_case.classify_intent(
                query=query_text,
                chat_history=chat_history,
                collection_id=collection_id
            )
            
            # Handle special intents
            if intent.kind == IntentKind.CHAT:
                return {
                    "chunks": [],
                    "intent": intent,
                    "message": "This appears to be general conversation. No document retrieval needed.",
                    "metadata": {"requires_retrieval": False}
                }
            
            if intent.kind == IntentKind.COLLECTION_SUMMARY and intent.requires_collection_scan:
                summary = await self.collection_summary_use_case.generate_collection_summary(
                    collection_id=collection_id,
                    query=query_text
                )
                return {
                    "chunks": [],
                    "intent": intent,
                    "summary": summary,
                    "metadata": {"is_collection_summary": True}
                }
        
        # Step 2: Query Transformation
        transformed_queries = [query_text]
        
        if query_transformation_enabled and query_transformation_methods:
            transformed_queries = await self.query_transformation_use_case.transform_query(
                query=query_text,
                methods=query_transformation_methods,
                chat_history=chat_history,
                max_history_turns=max_history_turns
            )
        
        print(f"Transformed queries ({len(transformed_queries)}): {transformed_queries}")
        
        # Step 3: Execute Retrieval
        if use_hybrid:
            chunks = await self._hybrid_search(
                queries=transformed_queries,
                collection_id=collection_id,
                top_k=top_k,
                filters=filters,
                alpha=alpha
            )
        else:
            chunks = await self._vector_only_search(
                queries=transformed_queries,
                collection_id=collection_id,
                top_k=top_k,
                filters=filters
            )
        
        return {
            "chunks": chunks,
            "intent": intent,
            "transformed_queries": transformed_queries,
            "metadata": {
                "num_queries_used": len(transformed_queries),
                "search_type": "hybrid" if use_hybrid else "vector_only",
                "total_results": len(chunks)
            }
        }
    
    async def _hybrid_search(
        self,
        queries: List[str],
        collection_id: Optional[str],
        top_k: int,
        filters: Optional[Dict],
        alpha: float
    ) -> List[DocumentChunk]:
        """Execute hybrid search for multiple queries and fuse results"""
        
        all_vector_results = []
        all_bm25_results = []
        
        for query in queries:
            print(f"Executing hybrid search for: {query}")
            
            # Generate query embedding
            query_embedding_task = self.embedding_service.generate_embedding(query)
            
            # Get BM25 search results
            bm25_search_task = self.bm25_service.search(
                query_text=query,
                collection_id=collection_id,
                top_k=top_k * 2,
                filters=filters
            )
            
            # Wait for embedding generation
            query_embedding = await query_embedding_task
            
            # Execute both searches in parallel
            vector_results, bm25_results = await asyncio.gather(
                self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    collection_id=collection_id,
                    top_k=top_k * 2,
                    filters=filters
                ),
                bm25_search_task,
            )
            
            all_vector_results.extend(vector_results)
            all_bm25_results.extend(bm25_results)
        
        # Remove duplicates while preserving order
        all_vector_results = self._deduplicate_chunks(all_vector_results)
        all_bm25_results = self._deduplicate_chunks(all_bm25_results)
        
        # Perform rank fusion
        return self.rank_fusion_service.fuse_results(
            vector_results=all_vector_results,
            bm25_results=all_bm25_results,
            alpha=alpha,
            top_k=top_k,
        )
    
    async def _vector_only_search(
        self,
        queries: List[str],
        collection_id: Optional[str],
        top_k: int,
        filters: Optional[Dict]
    ) -> List[DocumentChunk]:
        """Execute vector-only search for multiple queries"""
        
        all_vector_results = []
        
        for query in queries:
            print(f"Executing vector search for: {query}")
            
            # Generate embedding and search
            query_embedding = await self.embedding_service.generate_embedding(query)
            vector_results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                collection_id=collection_id,
                top_k=top_k,
                filters=filters
            )
            
            all_vector_results.extend(vector_results)
        
        # Remove duplicates and limit to top_k
        deduplicated = self._deduplicate_chunks(all_vector_results)
        return deduplicated[:top_k]
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks while preserving order"""
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    async def search_similar_chunks(
        self,
        reference_chunk_id: str,
        collection_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Find chunks similar to a reference chunk.
        
        Args:
            reference_chunk_id: ID of the reference chunk
            collection_id: Optional collection to limit search
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        # Get the reference chunk
        reference_chunk = await self.vector_store.get_chunk_by_id(reference_chunk_id)
        
        if not reference_chunk or not reference_chunk.embedding_vector:
            return []
        
        # Search for similar chunks using the reference embedding
        similar_chunks = await self.vector_store.search_similar(
            query_embedding=reference_chunk.embedding_vector,
            collection_id=collection_id,
            top_k=top_k + 1,  # +1 to account for the reference itself
            filters=None
        )
        
        # Remove the reference chunk from results
        return [chunk for chunk in similar_chunks if chunk.id != reference_chunk_id][:top_k]
