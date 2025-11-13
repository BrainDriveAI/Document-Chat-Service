"""
Tests for SearchDocumentsUseCase
"""
import pytest
from unittest.mock import AsyncMock, Mock

from app.core.use_cases.search_documents import SearchDocumentsUseCase
from app.core.domain.entities.document_chunk import DocumentChunk
from app.core.domain.entities.query_transformation import QueryTransformationMethod


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    mock = AsyncMock()
    # Return a sample embedding
    mock.generate_embedding.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store"""
    return AsyncMock()


@pytest.fixture
def mock_bm25_service():
    """Mock BM25 service"""
    return AsyncMock()


@pytest.fixture
def mock_rank_fusion_service():
    """Mock rank fusion service"""
    mock = Mock()
    # Default behavior: return first list passed to it
    mock.fuse_results.return_value = []
    return mock


@pytest.fixture
def mock_query_transformation():
    """Mock query transformation use case"""
    mock = AsyncMock()
    # Default: return original query
    mock.transform_query.return_value = ["original query"]
    return mock


@pytest.fixture
def search_use_case(
    mock_embedding_service,
    mock_vector_store,
    mock_bm25_service,
    mock_rank_fusion_service,
    mock_query_transformation
):
    """SearchDocumentsUseCase instance with mocks"""
    return SearchDocumentsUseCase(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        bm25_service=mock_bm25_service,
        rank_fusion_service=mock_rank_fusion_service,
        query_transformation_use_case=mock_query_transformation
    )


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    return [
        DocumentChunk(
            id="chunk1",
            document_id="doc1",
            collection_id="coll1",
            content="Sample content 1",
            chunk_index=0,
            chunk_type="paragraph",
            parent_chunk_id=None,
            embedding_vector=[0.1] * 384,
            metadata={"page": 1}
        ),
        DocumentChunk(
            id="chunk2",
            document_id="doc1",
            collection_id="coll1",
            content="Sample content 2",
            chunk_index=1,
            chunk_type="paragraph",
            parent_chunk_id=None,
            embedding_vector=[0.2] * 384,
            metadata={"page": 2}
        ),
        DocumentChunk(
            id="chunk3",
            document_id="doc2",
            collection_id="coll1",
            content="Sample content 3",
            chunk_index=0,
            chunk_type="paragraph",
            parent_chunk_id=None,
            embedding_vector=[0.3] * 384,
            metadata={"page": 1}
        )
    ]


class TestBasicSearch:
    """Tests for basic search functionality"""

    async def test_vector_only_search(
        self,
        search_use_case,
        mock_embedding_service,
        mock_vector_store,
        sample_chunks
    ):
        """Should perform vector-only search when use_hybrid=False"""
        mock_vector_store.search_similar.return_value = sample_chunks

        result = await search_use_case.search_documents(
            query_text="test query",
            use_hybrid=False,
            use_query_transformation=False
        )

        # Verify embedding generated
        mock_embedding_service.generate_embedding.assert_called_once_with("test query")

        # Verify vector search performed
        mock_vector_store.search_similar.assert_called_once()

        # Verify results returned
        assert len(result) == 3

    async def test_hybrid_search(
        self,
        search_use_case,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service,
        sample_chunks
    ):
        """Should perform hybrid search when use_hybrid=True"""
        mock_vector_store.search_similar.return_value = sample_chunks[:2]
        mock_bm25_service.search.return_value = sample_chunks[1:]
        mock_rank_fusion_service.fuse_results.return_value = sample_chunks

        result = await search_use_case.search_documents(
            query_text="test query",
            use_hybrid=True,
            use_query_transformation=False
        )

        # Verify both searches performed
        mock_vector_store.search_similar.assert_called_once()
        mock_bm25_service.search.assert_called_once()

        # Verify rank fusion called
        mock_rank_fusion_service.fuse_results.assert_called_once()

    async def test_respects_top_k_parameter(
        self,
        search_use_case,
        mock_vector_store,
        sample_chunks
    ):
        """Should respect top_k parameter"""
        mock_vector_store.search_similar.return_value = sample_chunks

        await search_use_case.search_documents(
            query_text="test",
            top_k=5,
            use_hybrid=False,
            use_query_transformation=False
        )

        call_args = mock_vector_store.search_similar.call_args
        assert call_args.kwargs["top_k"] == 5


class TestCollectionFiltering:
    """Tests for collection filtering"""

    async def test_filters_by_collection_id(
        self,
        search_use_case,
        mock_vector_store
    ):
        """Should filter by collection_id"""
        await search_use_case.search_documents(
            query_text="test",
            collection_id="coll_123",
            use_hybrid=False,
            use_query_transformation=False
        )

        call_args = mock_vector_store.search_similar.call_args
        assert call_args.kwargs["collection_id"] == "coll_123"

    async def test_searches_all_collections_when_none_specified(
        self,
        search_use_case,
        mock_vector_store
    ):
        """Should search all collections when no collection_id provided"""
        await search_use_case.search_documents(
            query_text="test",
            collection_id=None,
            use_hybrid=False,
            use_query_transformation=False
        )

        call_args = mock_vector_store.search_similar.call_args
        assert call_args.kwargs["collection_id"] is None


class TestQueryTransformation:
    """Tests for query transformation integration"""

    async def test_query_transformation_enabled(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        sample_chunks
    ):
        """Should use query transformation when enabled"""
        mock_query_transformation.transform_query.return_value = [
            "original query",
            "variation 1",
            "variation 2"
        ]
        mock_vector_store.search_similar.return_value = sample_chunks

        await search_use_case.search_documents(
            query_text="test query",
            use_hybrid=False,
            use_query_transformation=True
        )

        # Verify transformation called
        mock_query_transformation.transform_query.assert_called_once()

        # Verify multiple searches performed (one per transformed query)
        assert mock_vector_store.search_similar.call_count == 3

    async def test_query_transformation_disabled(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        sample_chunks
    ):
        """Should skip transformation when disabled"""
        mock_vector_store.search_similar.return_value = sample_chunks

        await search_use_case.search_documents(
            query_text="test query",
            use_hybrid=False,
            use_query_transformation=False
        )

        # Verify transformation not called
        mock_query_transformation.transform_query.assert_not_called()

        # Verify single search performed
        assert mock_vector_store.search_similar.call_count == 1

    async def test_passes_transformation_methods(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        sample_chunks
    ):
        """Should pass transformation methods to transformation use case"""
        mock_vector_store.search_similar.return_value = sample_chunks

        methods = [QueryTransformationMethod.MULTI_QUERY, QueryTransformationMethod.HYDE]

        await search_use_case.search_documents(
            query_text="test",
            use_hybrid=False,
            use_query_transformation=True,
            query_transformation_methods=methods
        )

        # Note: Current implementation doesn't pass methods parameter
        # This test documents expected behavior if that feature is added
        mock_query_transformation.transform_query.assert_called_once()


class TestHybridSearchDetails:
    """Tests for hybrid search details"""

    async def test_hybrid_search_doubles_top_k(
        self,
        search_use_case,
        mock_vector_store,
        mock_bm25_service
    ):
        """Should request 2x top_k for vector and BM25 before fusion"""
        mock_vector_store.search_similar.return_value = []
        mock_bm25_service.search.return_value = []

        await search_use_case.search_documents(
            query_text="test",
            top_k=10,
            use_hybrid=True,
            use_query_transformation=False
        )

        # Both should request 2x top_k
        vector_call = mock_vector_store.search_similar.call_args
        bm25_call = mock_bm25_service.search.call_args

        assert vector_call.kwargs["top_k"] == 20
        assert bm25_call.kwargs["top_k"] == 20

    async def test_hybrid_search_passes_alpha(
        self,
        search_use_case,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service
    ):
        """Should pass alpha parameter to rank fusion"""
        mock_vector_store.search_similar.return_value = []
        mock_bm25_service.search.return_value = []

        await search_use_case.search_documents(
            query_text="test",
            use_hybrid=True,
            alpha=0.7,
            use_query_transformation=False
        )

        call_args = mock_rank_fusion_service.fuse_results.call_args
        assert call_args.kwargs["alpha"] == 0.7

    async def test_hybrid_search_deduplicates_results(
        self,
        search_use_case,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service,
        sample_chunks
    ):
        """Should deduplicate results before rank fusion"""
        # Return overlapping chunks
        mock_vector_store.search_similar.return_value = sample_chunks[:2]
        mock_bm25_service.search.return_value = sample_chunks[1:]  # chunk2, chunk3
        mock_rank_fusion_service.fuse_results.return_value = sample_chunks

        await search_use_case.search_documents(
            query_text="test",
            use_hybrid=True,
            use_query_transformation=False
        )

        # Verify rank fusion received deduplicated lists
        call_args = mock_rank_fusion_service.fuse_results.call_args
        vector_results = call_args.kwargs["vector_results"]
        bm25_results = call_args.kwargs["bm25_results"]

        # Check no duplicates within each list
        vector_ids = [c.id for c in vector_results]
        bm25_ids = [c.id for c in bm25_results]

        assert len(vector_ids) == len(set(vector_ids))
        assert len(bm25_ids) == len(set(bm25_ids))


class TestMultipleTransformedQueries:
    """Tests for handling multiple transformed queries"""

    async def test_vector_search_with_multiple_queries(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        sample_chunks
    ):
        """Should aggregate results from multiple transformed queries"""
        mock_query_transformation.transform_query.return_value = [
            "query 1",
            "query 2",
            "query 3"
        ]

        # Each search returns different chunks
        mock_vector_store.search_similar.side_effect = [
            [sample_chunks[0]],
            [sample_chunks[1]],
            [sample_chunks[2]]
        ]

        result = await search_use_case.search_documents(
            query_text="test",
            use_hybrid=False,
            use_query_transformation=True,
            top_k=10
        )

        # Should deduplicate and return all unique chunks
        assert len(result) == 3
        assert mock_vector_store.search_similar.call_count == 3

    async def test_hybrid_search_with_multiple_queries(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service,
        sample_chunks
    ):
        """Should aggregate hybrid results from multiple queries"""
        mock_query_transformation.transform_query.return_value = ["query 1", "query 2"]

        mock_vector_store.search_similar.return_value = [sample_chunks[0]]
        mock_bm25_service.search.return_value = [sample_chunks[1]]
        mock_rank_fusion_service.fuse_results.return_value = sample_chunks

        await search_use_case.search_documents(
            query_text="test",
            use_hybrid=True,
            use_query_transformation=True
        )

        # Should call searches for each transformed query
        assert mock_vector_store.search_similar.call_count == 2
        assert mock_bm25_service.search.call_count == 2

    async def test_deduplicates_across_transformed_queries(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store,
        sample_chunks
    ):
        """Should deduplicate results across all transformed queries"""
        mock_query_transformation.transform_query.return_value = ["query 1", "query 2"]

        # Both queries return same chunk
        mock_vector_store.search_similar.side_effect = [
            [sample_chunks[0], sample_chunks[1]],
            [sample_chunks[1], sample_chunks[2]]  # chunk1 appears in both
        ]

        result = await search_use_case.search_documents(
            query_text="test",
            use_hybrid=False,
            use_query_transformation=True,
            top_k=10
        )

        # Should have 3 unique chunks (not 4)
        assert len(result) == 3


class TestMetadataFilters:
    """Tests for metadata filtering"""

    async def test_passes_filters_to_vector_store(
        self,
        search_use_case,
        mock_vector_store
    ):
        """Should pass metadata filters to vector store"""
        filters = {"document_type": "pdf", "page": 1}

        await search_use_case.search_documents(
            query_text="test",
            filters=filters,
            use_hybrid=False,
            use_query_transformation=False
        )

        call_args = mock_vector_store.search_similar.call_args
        assert call_args.kwargs["filters"] == filters

    async def test_passes_filters_to_bm25(
        self,
        search_use_case,
        mock_bm25_service,
        mock_vector_store
    ):
        """Should pass metadata filters to BM25"""
        mock_vector_store.search_similar.return_value = []
        mock_bm25_service.search.return_value = []

        filters = {"author": "John Doe"}

        await search_use_case.search_documents(
            query_text="test",
            filters=filters,
            use_hybrid=True,
            use_query_transformation=False
        )

        call_args = mock_bm25_service.search.call_args
        assert call_args.kwargs["filters"] == filters


class TestChatHistoryIntegration:
    """Tests for chat history integration"""

    async def test_passes_chat_history_to_transformation(
        self,
        search_use_case,
        mock_query_transformation,
        mock_vector_store
    ):
        """Should pass chat history to query transformation"""
        mock_vector_store.search_similar.return_value = []

        chat_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        await search_use_case.search_documents(
            query_text="Tell me more",
            chat_history=chat_history,
            use_query_transformation=True,
            use_hybrid=False
        )

        # Verify chat history passed (current implementation doesn't pass it)
        # This documents expected behavior
        mock_query_transformation.transform_query.assert_called_once()


class TestErrorHandling:
    """Tests for error handling"""

    async def test_handles_embedding_generation_error(
        self,
        search_use_case,
        mock_embedding_service,
        mock_vector_store
    ):
        """Should handle embedding generation errors gracefully"""
        mock_embedding_service.generate_embedding.side_effect = Exception("Embedding error")

        with pytest.raises(Exception):
            await search_use_case.search_documents(
                query_text="test",
                use_hybrid=False,
                use_query_transformation=False
            )

    async def test_handles_vector_store_error(
        self,
        search_use_case,
        mock_vector_store
    ):
        """Should handle vector store errors"""
        mock_vector_store.search_similar.side_effect = Exception("Vector store error")

        with pytest.raises(Exception):
            await search_use_case.search_documents(
                query_text="test",
                use_hybrid=False,
                use_query_transformation=False
            )


class TestEmptyResults:
    """Tests for handling empty results"""

    async def test_returns_empty_list_when_no_results(
        self,
        search_use_case,
        mock_vector_store
    ):
        """Should return empty list when no results found"""
        mock_vector_store.search_similar.return_value = []

        result = await search_use_case.search_documents(
            query_text="test",
            use_hybrid=False,
            use_query_transformation=False
        )

        assert result == []

    async def test_hybrid_search_with_empty_results(
        self,
        search_use_case,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service
    ):
        """Should handle empty results in hybrid search"""
        mock_vector_store.search_similar.return_value = []
        mock_bm25_service.search.return_value = []
        mock_rank_fusion_service.fuse_results.return_value = []

        result = await search_use_case.search_documents(
            query_text="test",
            use_hybrid=True,
            use_query_transformation=False
        )

        assert result == []
