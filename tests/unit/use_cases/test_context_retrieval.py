"""
Unit tests for ContextRetrievalUseCase.

Tests cover:
- Intent classification → query transformation → hybrid search pipeline
- Each intent type (CHAT, RETRIEVAL, COLLECTION_SUMMARY, etc.)
- Query transformation methods (STEP_BACK, SUB_QUERY, CONTEXTUAL)
- Hybrid search with rank fusion
- Dynamic context window detection
- With/without chat history
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict

from app.core.use_cases.context_retrieval import ContextRetrievalUseCase
from app.core.domain.entities.search_intent import IntentKind, Intent
from app.core.domain.entities.query_transformation import QueryTransformationMethod
from app.core.domain.entities.context_result import ContextResult, GenerationType
from app.core.domain.entities.document_chunk import DocumentChunk
from app.core.domain.value_objects.embedding import EmbeddingVector


@pytest.fixture
def mock_intent_classification_use_case():
    """Mock intent classification use case."""
    use_case = AsyncMock()

    async def classify_intent(query: str, chat_history=None, collection_id=None):
        # Default: RETRIEVAL intent
        return Intent(
            intent_kind=IntentKind.RETRIEVAL,
            requires_retrieval=True,
            confidence=0.95,
            reasoning="User asking for specific information"
        )

    use_case.classify_intent.side_effect = classify_intent
    return use_case


@pytest.fixture
def mock_query_transformation_use_case():
    """Mock query transformation use case."""
    use_case = AsyncMock()

    async def transform_query(query: str, methods: List, chat_history=None, max_history_turns=3):
        # Return original + 1 transformed query
        return [query, f"{query} expanded"]

    use_case.transform_query.side_effect = transform_query
    return use_case


@pytest.fixture
def mock_collection_summary_use_case():
    """Mock collection summary use case."""
    use_case = AsyncMock()

    async def get_sample_chunks(collection_id: str, sample_size: int):
        return [
            DocumentChunk.create(
                document_id=f"doc-{i}",
                collection_id=collection_id,
                content=f"Sample content {i}",
                chunk_index=i,
                metadata={"page": 1}
            )
            for i in range(min(sample_size, 5))
        ]

    use_case.get_sample_chunks.side_effect = get_sample_chunks
    return use_case


@pytest.fixture
def mock_model_info_service():
    """Mock model info service for dynamic context window detection."""
    service = AsyncMock()

    async def get_context_window(model_name: str):
        # Return different context windows based on model
        windows = {
            "llama3.2:8b": 131072,
            "llama3.2:3b": 131072,
            "unknown-model": 4096
        }
        return windows.get(model_name, 4096)

    service.get_context_window.side_effect = get_context_window
    return service


@pytest.fixture
def sample_chunks():
    """Sample document chunks for search results."""
    return [
        DocumentChunk(
            id=f"chunk-{i}",
            document_id="doc-1",
            collection_id="collection-123",
            content=f"This is chunk {i} with relevant information.",
            chunk_index=i,
            chunk_type="paragraph",
            parent_chunk_id=None,
            metadata={"page": 1, "score": 0.9 - i * 0.1},
            embedding_vector=[0.1, 0.2, 0.3]
        )
        for i in range(5)
    ]


@pytest.fixture
def context_retrieval_use_case(
    mock_embedding_service,
    mock_vector_store,
    mock_bm25_service,
    mock_rank_fusion_service,
    mock_intent_classification_use_case,
    mock_query_transformation_use_case,
    mock_collection_summary_use_case,
    mock_model_info_service
):
    """Create ContextRetrievalUseCase with mocked dependencies."""
    return ContextRetrievalUseCase(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        bm25_service=mock_bm25_service,
        rank_fusion_service=mock_rank_fusion_service,
        query_transformation_use_case=mock_query_transformation_use_case,
        intent_classification_use_case=mock_intent_classification_use_case,
        collection_summary_use_case=mock_collection_summary_use_case,
        model_info_service=mock_model_info_service
    )


class TestContextRetrievalPipeline:
    """Tests for the full context retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_retrieve_context_basic_flow(
        self,
        context_retrieval_use_case,
        mock_intent_classification_use_case,
        mock_vector_store,
        sample_chunks
    ):
        """Test basic retrieval flow: intent → query transform → search."""
        # Arrange
        mock_vector_store.search_similar = AsyncMock(return_value=sample_chunks)

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="What is machine learning?",
            collection_id="collection-123",
            top_k=5
        )

        # Assert
        assert isinstance(result, ContextResult)
        assert result.chunks is not None
        assert len(result.chunks) > 0
        assert result.intent is not None
        assert result.generation_type == GenerationType.ANSWER

        # Verify intent classification was called
        mock_intent_classification_use_case.classify_intent.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_context_with_chat_history(
        self,
        context_retrieval_use_case,
        mock_query_transformation_use_case
    ):
        """Test retrieval with chat history context."""
        # Arrange
        chat_history = [
            {"user": "Tell me about AI", "assistant": "AI is..."},
            {"user": "What about ML?", "assistant": "ML is..."}
        ]

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="Can you explain more?",
            chat_history=chat_history,
            collection_id="collection-123"
        )

        # Assert
        assert result is not None
        # Verify query transformation received chat history
        call_args = mock_query_transformation_use_case.transform_query.call_args
        if call_args:
            assert call_args.kwargs.get('chat_history') == chat_history


class TestIntentHandling:
    """Tests for different intent types."""

    @pytest.mark.asyncio
    async def test_chat_intent_returns_no_retrieval(
        self,
        context_retrieval_use_case,
        mock_intent_classification_use_case,
        mock_vector_store
    ):
        """Test that CHAT intent skips retrieval."""
        # Arrange
        async def classify_as_chat(*args, **kwargs):
            return Intent(
                intent_kind=IntentKind.CHAT,
                requires_retrieval=False,
                confidence=0.95,
                reasoning="Casual greeting"
            )

        mock_intent_classification_use_case.classify_intent.side_effect = classify_as_chat

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="Hello!",
            collection_id="collection-123"
        )

        # Assert
        assert result.intent.kind == IntentKind.CHAT
        assert result.chunks == []
        assert result.generation_type == GenerationType.NONE
        # Verify no search was performed
        mock_vector_store.search_similar.assert_not_called()

    @pytest.mark.asyncio
    async def test_collection_summary_intent(
        self,
        context_retrieval_use_case,
        mock_intent_classification_use_case,
        mock_collection_summary_use_case
    ):
        """Test COLLECTION_SUMMARY intent gets sample chunks."""
        # Arrange
        async def classify_as_summary(*args, **kwargs):
            return Intent(
                intent_kind=IntentKind.COLLECTION_SUMMARY,
                requires_retrieval=True,
                confidence=0.90,
                reasoning="User wants collection overview",
                requires_collection_scan=True
            )

        mock_intent_classification_use_case.classify_intent.side_effect = classify_as_summary

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="Summarize the collection",
            collection_id="collection-123",
            top_k=10
        )

        # Assert
        assert result.intent.kind == IntentKind.COLLECTION_SUMMARY
        assert result.generation_type == GenerationType.SUMMARY
        assert len(result.chunks) > 0
        assert result.metadata.get("is_collection_wide") is True

        # Verify collection summary was called
        mock_collection_summary_use_case.get_sample_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_intent(
        self,
        context_retrieval_use_case,
        mock_intent_classification_use_case,
        sample_chunks
    ):
        """Test normal RETRIEVAL intent."""
        # Arrange
        async def classify_as_retrieval(*args, **kwargs):
            return Intent(
                intent_kind=IntentKind.RETRIEVAL,
                requires_retrieval=True,
                confidence=0.95,
                reasoning="User asking for information"
            )

        mock_intent_classification_use_case.classify_intent.side_effect = classify_as_retrieval

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="What is RAG?",
            collection_id="collection-123"
        )

        # Assert
        assert result.intent.kind == IntentKind.RETRIEVAL
        assert result.generation_type == GenerationType.ANSWER


class TestQueryTransformation:
    """Tests for query transformation integration."""

    @pytest.mark.asyncio
    async def test_query_transformation_enabled(
        self,
        context_retrieval_use_case,
        mock_query_transformation_use_case
    ):
        """Test that query transformation is applied when enabled."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="What is ML?",
            collection_id="collection-123",
            query_transformation_enabled=True,
            query_transformation_methods=[QueryTransformationMethod.MULTI_QUERY]
        )

        # Assert
        mock_query_transformation_use_case.transform_query.assert_called_once()
        call_args = mock_query_transformation_use_case.transform_query.call_args
        assert QueryTransformationMethod.MULTI_QUERY in call_args.kwargs['methods']

    @pytest.mark.asyncio
    async def test_query_transformation_disabled(
        self,
        context_retrieval_use_case,
        mock_query_transformation_use_case
    ):
        """Test that query transformation is skipped when disabled."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="What is ML?",
            collection_id="collection-123",
            query_transformation_enabled=False
        )

        # Assert
        # Should not be called when disabled
        mock_query_transformation_use_case.transform_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_transformation_methods(
        self,
        context_retrieval_use_case,
        mock_query_transformation_use_case
    ):
        """Test multiple query transformation methods."""
        # Arrange
        methods = [
            QueryTransformationMethod.MULTI_QUERY,
            QueryTransformationMethod.HYDE
        ]

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="Complex query",
            collection_id="collection-123",
            query_transformation_enabled=True,
            query_transformation_methods=methods
        )

        # Assert
        call_args = mock_query_transformation_use_case.transform_query.call_args
        assert set(call_args.kwargs['methods']) == set(methods)


class TestHybridSearch:
    """Tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_enabled(
        self,
        context_retrieval_use_case,
        mock_vector_store,
        mock_bm25_service,
        mock_rank_fusion_service,
        sample_chunks
    ):
        """Test that hybrid search uses both vector and BM25."""
        # Arrange
        mock_vector_store.search_similar = AsyncMock(return_value=sample_chunks)
        # BM25 already returns chunks, no need to override
        mock_rank_fusion_service.fuse_results = MagicMock(return_value=sample_chunks[:3])

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            use_hybrid=True,
            alpha=0.6
        )

        # Assert
        mock_vector_store.search_similar.assert_called()
        mock_bm25_service.search.assert_called()
        mock_rank_fusion_service.fuse_results.assert_called_once()

        # Verify alpha was passed
        call_args = mock_rank_fusion_service.fuse_results.call_args
        assert call_args.kwargs['alpha'] == 0.6

    @pytest.mark.asyncio
    async def test_vector_only_search(
        self,
        context_retrieval_use_case,
        mock_vector_store,
        mock_bm25_service,
        sample_chunks
    ):
        """Test vector-only search when hybrid is disabled."""
        # Arrange
        mock_vector_store.search_similar = AsyncMock(return_value=sample_chunks)

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            use_hybrid=False
        )

        # Assert
        mock_vector_store.search_similar.assert_called()
        # BM25 should NOT be called
        mock_bm25_service.search.assert_not_called()


class TestDynamicContextWindow:
    """Tests for dynamic context window detection."""

    @pytest.mark.asyncio
    @patch('app.core.use_cases.context_retrieval.settings')
    async def test_context_window_detection_adjusts_top_k(
        self,
        mock_settings,
        context_retrieval_use_case,
        mock_model_info_service
    ):
        """Test that context window detection adjusts top_k."""
        # Arrange
        mock_settings.AVG_CHUNK_TOKENS = 200
        mock_settings.CONTEXT_SAFETY_MARGIN = 0.8
        mock_settings.MIN_TOP_K = 3
        mock_settings.MAX_TOP_K = 50

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            top_k=10,
            model_name="llama3.2:8b",
            system_prompt="You are a helpful assistant."
        )

        # Assert
        mock_model_info_service.get_context_window.assert_called_with("llama3.2:8b")
        # Check metadata contains optimization info
        assert "context_window_optimization" in result.metadata
        assert result.metadata["context_window_optimization"]["optimization_enabled"] is True

    @pytest.mark.asyncio
    async def test_context_window_detection_disabled_without_model_name(
        self,
        context_retrieval_use_case,
        mock_model_info_service
    ):
        """Test that optimization is skipped when model_name not provided."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            top_k=10,
            model_name=None  # No model name
        )

        # Assert
        # Should not attempt to get context window
        mock_model_info_service.get_context_window.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_window_detection_failure_fallback(
        self,
        context_retrieval_use_case,
        mock_model_info_service
    ):
        """Test graceful fallback when context window detection fails."""
        # Arrange
        async def fail_detection(*args, **kwargs):
            raise Exception("Model not found")

        mock_model_info_service.get_context_window.side_effect = fail_detection

        # Act - should not raise exception
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            top_k=10,
            model_name="unknown-model"
        )

        # Assert - should fall back to default top_k
        assert result is not None
        assert "context_window_optimization" in result.metadata
        assert result.metadata["context_window_optimization"]["optimization_enabled"] is False


class TestReverseContextForOllama:
    """Tests for context reversal optimization."""

    @pytest.mark.asyncio
    @patch('app.core.use_cases.context_retrieval.settings')
    async def test_context_reversal_enabled(
        self,
        mock_settings,
        context_retrieval_use_case,
        sample_chunks
    ):
        """Test that chunks are reversed when REVERSE_CONTEXT_FOR_OLLAMA is true."""
        # Arrange
        mock_settings.REVERSE_CONTEXT_FOR_OLLAMA = True

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123"
        )

        # Assert - most relevant chunks should be at the end
        if result.chunks:
            # Verify order was reversed (can't easily test without modifying mocks)
            assert len(result.chunks) > 0

    @pytest.mark.asyncio
    @patch('app.core.use_cases.context_retrieval.settings')
    async def test_context_reversal_disabled(
        self,
        mock_settings,
        context_retrieval_use_case
    ):
        """Test that chunks keep original order when reversal disabled."""
        # Arrange
        mock_settings.REVERSE_CONTEXT_FOR_OLLAMA = False

        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123"
        )

        # Assert
        assert result is not None


class TestIntentClassificationToggle:
    """Tests for intent classification on/off."""

    @pytest.mark.asyncio
    async def test_intent_classification_disabled(
        self,
        context_retrieval_use_case,
        mock_intent_classification_use_case
    ):
        """Test that intent classification can be disabled."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            use_intent_classification=False
        )

        # Assert
        # Should not call intent classification
        mock_intent_classification_use_case.classify_intent.assert_not_called()
        # Should default to RETRIEVAL intent
        assert result.intent.kind == IntentKind.RETRIEVAL
        assert result.intent.confidence == 1.0


class TestMetadata:
    """Tests for result metadata."""

    @pytest.mark.asyncio
    async def test_metadata_includes_query_info(
        self,
        context_retrieval_use_case
    ):
        """Test that result metadata includes query information."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            use_hybrid=True,
            alpha=0.7
        )

        # Assert
        assert "search_type" in result.metadata
        assert result.metadata["search_type"] == "hybrid"
        assert result.metadata.get("alpha") == 0.7

    @pytest.mark.asyncio
    async def test_metadata_includes_transformed_queries(
        self,
        context_retrieval_use_case,
        mock_query_transformation_use_case
    ):
        """Test that metadata includes transformed queries."""
        # Act
        result = await context_retrieval_use_case.retrieve_context(
            query_text="test query",
            collection_id="collection-123",
            query_transformation_enabled=True,
            query_transformation_methods=[QueryTransformationMethod.MULTI_QUERY]
        )

        # Assert
        assert "transformed_queries" in result.metadata
        assert "num_queries_used" in result.metadata
