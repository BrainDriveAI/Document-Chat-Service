import asyncio
from typing import List, Optional, Dict
from ..domain.entities.document_chunk import DocumentChunk
from app.core.domain.entities.query_transformation import QueryTransformationMethod
from ..domain.value_objects.embedding import SearchQuery
from ..ports.embedding_service import EmbeddingService
from ..ports.vector_store import VectorStore
from ..ports.bm25_service import BM25Service
from ..ports.rank_fusion_service import RankFusionService
from .query_transformation import QueryTransformationUseCase


class SearchDocumentsUseCase:
    """Use case for searching documents"""

    def __init__(
            self,
            embedding_service: EmbeddingService,
            vector_store: VectorStore,
            bm25_service: BM25Service,
            rank_fusion_service: RankFusionService,
            query_transformation_use_case: QueryTransformationUseCase,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_service = bm25_service
        self.rank_fusion_service = rank_fusion_service
        self.query_transformation_use_case = query_transformation_use_case

    async def search_documents(
            self,
            query_text: str,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None,
            use_hybrid: bool = True,
            alpha: float = 0.5,  # Weight for vector vs BM25 (0.5 = equal weight)
            chat_history: Optional[List[Dict]] = None,
            use_query_transformation: bool = True,
            query_transformation_methods: Optional[List[QueryTransformationMethod]] = None,
    ) -> List[DocumentChunk]:
        """
        Search for relevant document chunks

        Args:
            query_text: Search query
            collection_id: Optional collection to search within
            top_k: Number of results to return
            filters: Additional metadata filters
            use_hybrid: Whether to use hybrid search (vector + keyword)
            alpha: Alpha parameter for hybrid search (vector + keyword)
            use_query_transformation: Whether to use query transformation

        Returns:
            List of relevant document chunks
        """
        # Identify intent

        # Transform query if needed

        # return relevant chunks
        if use_query_transformation:
            transformed_queries = await self.query_transformation_use_case.transform_query(query_text)
        else:
            transformed_queries = [query_text]

        all_vector_results = []
        all_bm25_results = []

        if use_hybrid:
            for query in transformed_queries:
                print(f"TRANSFORMED QUERY: {query}")
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

            # Remove duplicates
            all_vector_results = list({doc.id: doc for doc in all_vector_results}.values())
            all_bm25_results = list({doc.id: doc for doc in all_bm25_results}.values())

            # Perform rank fusion
            return self.rank_fusion_service.fuse_results(
                vector_results=all_vector_results,
                bm25_results=all_bm25_results,
                alpha=alpha,
                top_k=top_k,
            )
        else:
            for query in transformed_queries:
                # Use pure vector search
                query_embedding = await self.embedding_service.generate_embedding(query)
                vector_results = await self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    collection_id=collection_id,
                    top_k=top_k,
                    filters=filters
                )
                all_vector_results.extend(vector_results)

            # Remove duplicates
            return list({doc.id: doc for doc in all_vector_results}.values())[:top_k]

    async def combine_documents(
            self,
            query: SearchQuery,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            alpha: float = 0.5
    ) -> List[DocumentChunk]:
        """Combine vector and BM25 results using Reciprocal Rank Fusion"""

        # Create score maps
        vector_scores = {chunk.id: chunk.metadata.get('similarity', 0) for chunk in vector_results}
        bm25_scores = {chunk.id: chunk.metadata.get('bm25_score', 0) for chunk in bm25_results}

        # Normalize scores to [0, 1] range
        if vector_scores:
            max_vec_score = max(vector_scores.values())
            vector_scores = {k: v / max_vec_score for k, v in vector_scores.items()}

        if bm25_scores:
            max_bm25_score = max(bm25_scores.values())
            bm25_scores = {k: v / max_bm25_score for k, v in bm25_scores.items()}

        # Combine all unique chunks
        all_chunks = {}
        for chunk in vector_results + bm25_results:
            if chunk.id not in all_chunks:
                all_chunks[chunk.id] = chunk

        # Calculate hybrid scores
        hybrid_scores = []
        for chunk_id, chunk in all_chunks.items():
            vec_score = vector_scores.get(chunk_id, 0)
            bm25_score = bm25_scores.get(chunk_id, 0)

            # Weighted combination
            hybrid_score = alpha * vec_score + (1 - alpha) * bm25_score

            # Update metadata with hybrid score
            chunk.metadata = {**chunk.metadata, 'hybrid_score': hybrid_score}
            hybrid_scores.append((hybrid_score, chunk))

        # Sort by hybrid score and return top results
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in hybrid_scores[:query.top_k]]

    async def search_similar_chunks(
            self,
            reference_chunk_id: str,
            collection_id: Optional[str] = None,
            top_k: int = 5
    ) -> List[DocumentChunk]:
        """Find chunks similar to a reference chunk"""
        # This would require getting the reference chunk first
        # and then searching for similar ones
        # Implementation depends on how you want to handle this
        pass
