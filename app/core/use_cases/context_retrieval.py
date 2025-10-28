import asyncio
from typing import List, Optional, Dict

from ..domain.entities.document_chunk import DocumentChunk
from ..domain.entities.query_transformation import QueryTransformationMethod
from ..domain.entities.context_result import ContextResult, GenerationType
from ..domain.entities.search_intent import IntentKind, Intent
from ..ports.embedding_service import EmbeddingService
from ..ports.vector_store import VectorStore
from ..ports.bm25_service import BM25Service
from ..ports.rank_fusion_service import RankFusionService
from .query_transformation import QueryTransformationUseCase
from .intent_classification import IntentClassificationUseCase
from .collection_summary import CollectionSummaryUseCase


class ContextRetrievalUseCase:
    """
    Use case for retrieving relevant context based on user queries.
    
    Responsibilities:
    - Classify user intent
    - Transform queries for better retrieval
    - Find relevant document chunks
    - Return consistent context structure
    
    Does NOT:
    - Generate LLM responses
    - Make decisions about what to do with context
    """
    
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
    
    async def retrieve_context(
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
    ) -> ContextResult:
        """
        Retrieve relevant context for a given query.
        Always returns consistent ContextResult structure.
        
        Args:
            query_text: User's query
            collection_id: Optional collection to search in
            chat_history: Conversation history for contextualization
            top_k: Number of chunks to return
            filters: Metadata filters
            use_hybrid: Use hybrid search (vector + BM25)
            alpha: Weight for vector vs BM25
            use_intent_classification: Enable intent classification
            query_transformation_enabled: Enable query transformation
            query_transformation_methods: Transformation methods to use
            max_history_turns: Max conversation turns to consider
            
        Returns:
            ContextResult with chunks and metadata
        """
        
        # Step 1: Classify Intent
        intent = None
        if use_intent_classification:
            intent = await self.intent_classification_use_case.classify_intent(
                query=query_text,
                chat_history=chat_history,
                collection_id=collection_id
            )
        else:
            # Default to retrieval intent if classification disabled
            intent = Intent(
                intent_kind=IntentKind.RETRIEVAL,
                requires_retrieval=True,
                confidence=1.0,
                reasoning="Intent classification disabled, defaulting to retrieval"
            )
        
        # Step 2: Route based on intent
        if intent.type == IntentKind.CHAT:
            return ContextResult.create_chat_result(intent)
        
        elif intent.type == IntentKind.COLLECTION_SUMMARY or intent.requires_collection_scan:
            return await self._handle_collection_summary_intent(
                query_text=query_text,
                collection_id=collection_id,
                intent=intent,
                sample_size=top_k
            )
        
        else:
            # Normal retrieval (RETRIEVAL, CLARIFICATION, COMPARISON, LISTING)
            return await self._handle_retrieval_intent(
                query_text=query_text,
                collection_id=collection_id,
                chat_history=chat_history,
                top_k=top_k,
                filters=filters,
                use_hybrid=use_hybrid,
                alpha=alpha,
                intent=intent,
                query_transformation_enabled=query_transformation_enabled,
                query_transformation_methods=query_transformation_methods,
                max_history_turns=max_history_turns
            )
    
    async def _handle_collection_summary_intent(
        self,
        query_text: str,
        collection_id: str,
        intent,
        sample_size: int
    ) -> ContextResult:
        """
        Handle collection-level summary requests.
        Returns diverse sample of chunks for summarization.
        """
        # Get diverse sample of chunks (NOT the generated summary)
        sample_chunks = await self.collection_summary_use_case.get_sample_chunks(
            collection_id=collection_id,
            sample_size=sample_size
        )
        
        return ContextResult.create_retrieval_result(
            chunks=sample_chunks,
            intent=intent,
            generation_type=GenerationType.SUMMARY,
            metadata={
                "sample_size": len(sample_chunks),
                "is_collection_wide": True,
                "query": query_text,
                "message": f"Retrieved {len(sample_chunks)} diverse chunks from collection for summarization"
            }
        )
    
    async def _handle_retrieval_intent(
        self,
        query_text: str,
        collection_id: Optional[str],
        chat_history: Optional[List[Dict]],
        top_k: int,
        filters: Optional[Dict],
        use_hybrid: bool,
        alpha: float,
        intent,
        query_transformation_enabled: bool,
        query_transformation_methods: Optional[List[QueryTransformationMethod]],
        max_history_turns: int
    ) -> ContextResult:
        """
        Handle normal retrieval intents (search for relevant chunks).
        """
        
        # Step 1: Query Transformation
        transformed_queries = [query_text]
        
        if query_transformation_enabled and query_transformation_methods:
            transformed_queries = await self.query_transformation_use_case.transform_query(
                query=query_text,
                methods=query_transformation_methods,
                chat_history=chat_history,
                max_history_turns=max_history_turns
            )
        
        print(f"Transformed queries ({len(transformed_queries)}): {transformed_queries}")
        
        # Step 2: Execute Retrieval
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
        
        # Step 3: Determine generation type based on intent
        generation_type = self._map_intent_to_generation_type(intent.type)
        
        return ContextResult.create_retrieval_result(
            chunks=chunks,
            intent=intent,
            generation_type=generation_type,
            metadata={
                "num_queries_used": len(transformed_queries),
                "transformed_queries": transformed_queries,
                "search_type": "hybrid" if use_hybrid else "vector_only",
                "total_results": len(chunks),
                "alpha": alpha if use_hybrid else None
            }
        )
    
    def _map_intent_to_generation_type(self, intent_type: IntentKind) -> GenerationType:
        """Map intent type to generation type"""
        mapping = {
            IntentKind.RETRIEVAL: GenerationType.ANSWER,
            IntentKind.CLARIFICATION: GenerationType.ANSWER,
            IntentKind.SUMMARY: GenerationType.SUMMARY,
            IntentKind.COMPARISON: GenerationType.COMPARISON,
            IntentKind.LISTING: GenerationType.LISTING,
        }
        return mapping.get(intent_type, GenerationType.ANSWER)
    
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

