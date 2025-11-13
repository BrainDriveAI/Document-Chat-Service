"""
Unit tests for SimplifiedDocumentProcessingUseCase.

Tests cover:
- Document processing pipeline (happy path)
- Contextual retrieval enabled/disabled
- Embedding generation with batching
- Error handling (document processor, embedding failures)
- Cleanup on failure (document status rollback)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from app.core.use_cases.simple_document import SimplifiedDocumentProcessingUseCase
from app.core.domain.entities.document import Document, DocumentStatus, DocumentType
from app.core.domain.entities.document_chunk import DocumentChunk
from app.core.domain.entities.structured_element import StructuredElement
from app.core.domain.value_objects.embedding import EmbeddingVector
from app.core.domain.exceptions import DocumentProcessingError


@pytest.fixture
def mock_document_processor():
    """Mock document processor."""
    processor = AsyncMock()

    # Default behavior: return sample chunks and complete text
    async def process_document(document):
        chunks = [
            DocumentChunk.create(
                document_id=document.id,
                collection_id=document.collection_id,
                content=f"Chunk {i} content",
                chunk_index=i,
                metadata={"page": 1}
            )
            for i in range(3)
        ]
        complete_text = "Complete document text for context"
        return chunks, complete_text

    processor.process_document.side_effect = process_document
    return processor


@pytest.fixture
def document_processing_use_case(
    mock_document_repository,
    mock_document_processor,
    mock_embedding_service,
    mock_vector_store,
    mock_collection_repository,
    mock_llm_service,
    mock_bm25_service
):
    """Create SimplifiedDocumentProcessingUseCase with mocked dependencies."""
    return SimplifiedDocumentProcessingUseCase(
        document_repo=mock_document_repository,
        document_processor=mock_document_processor,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        collection_repo=mock_collection_repository,
        llm_service=mock_llm_service,
        contextual_llm=None,  # Disabled by default
        bm25_service=mock_bm25_service
    )


class TestSimplifiedDocumentProcessingHappyPath:
    """Tests for successful document processing scenarios."""

    @pytest.mark.asyncio
    async def test_process_document_success(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_repository
    ):
        """Test successful document processing pipeline."""
        # Act
        result = await document_processing_use_case.process_document(sample_document)

        # Assert
        assert result.status == DocumentStatus.PROCESSED
        assert result.chunk_count == 3
        assert result.processed_at is not None

        # Verify document was saved twice (processing + processed)
        assert mock_document_repository.save.call_count == 2

    @pytest.mark.asyncio
    async def test_process_document_marks_as_processing(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_repository
    ):
        """Test that document is marked as processing before processing starts."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert - first save call should have processing status
        first_call = mock_document_repository.save.call_args_list[0]
        saved_doc = first_call[0][0]
        assert saved_doc.status == DocumentStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_process_document_generates_embeddings(
        self,
        document_processing_use_case,
        sample_document,
        mock_embedding_service
    ):
        """Test that embeddings are generated for all chunks."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert
        mock_embedding_service.generate_embeddings_batch.assert_called_once()
        call_args = mock_embedding_service.generate_embeddings_batch.call_args
        texts = call_args[0][0]
        assert len(texts) == 3  # 3 chunks
        assert all(isinstance(text, str) for text in texts)

    @pytest.mark.asyncio
    async def test_process_document_indexes_in_vector_store(
        self,
        document_processing_use_case,
        sample_document,
        mock_vector_store
    ):
        """Test that chunks are indexed in vector store."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert
        mock_vector_store.add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_indexes_in_bm25(
        self,
        document_processing_use_case,
        sample_document,
        mock_bm25_service
    ):
        """Test that chunks are indexed in BM25."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert
        mock_bm25_service.index_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_increments_collection_count(
        self,
        document_processing_use_case,
        sample_document,
        mock_collection_repository
    ):
        """Test that collection document count is incremented."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection_repository.find_by_id.return_value = mock_collection

        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert
        mock_collection_repository.find_by_id.assert_called_once_with(sample_document.collection_id)
        mock_collection.increment_document_count.assert_called_once()
        mock_collection_repository.save.assert_called_once_with(mock_collection)


class TestContextualRetrieval:
    """Tests for contextual retrieval feature."""

    @pytest.mark.asyncio
    @patch('app.core.use_cases.simple_document.settings')
    async def test_contextual_retrieval_enabled(
        self,
        mock_settings,
        mock_document_repository,
        mock_document_processor,
        mock_embedding_service,
        mock_vector_store,
        mock_collection_repository,
        mock_llm_service,
        mock_bm25_service,
        sample_document
    ):
        """Test document processing with contextual retrieval enabled."""
        # Arrange
        mock_settings.ENABLE_CONTEXTUAL_RETRIEVAL = True
        contextual_llm = AsyncMock()

        # Mock contextual LLM to return context
        async def generate_response(prompt, **kwargs):
            return "Contextual information about this chunk"
        contextual_llm.generate_response.side_effect = generate_response

        use_case = SimplifiedDocumentProcessingUseCase(
            document_repo=mock_document_repository,
            document_processor=mock_document_processor,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            collection_repo=mock_collection_repository,
            llm_service=mock_llm_service,
            contextual_llm=contextual_llm,
            bm25_service=mock_bm25_service
        )

        # Act
        result = await use_case.process_document(sample_document)

        # Assert
        assert result.status == DocumentStatus.PROCESSED
        # Contextual LLM should have been called for each chunk
        assert contextual_llm.generate_response.call_count >= 1

    @pytest.mark.asyncio
    @patch('app.core.use_cases.simple_document.settings')
    async def test_contextual_retrieval_disabled(
        self,
        mock_settings,
        document_processing_use_case,
        sample_document
    ):
        """Test that contextual retrieval is skipped when disabled."""
        # Arrange
        mock_settings.ENABLE_CONTEXTUAL_RETRIEVAL = False

        # Act
        result = await document_processing_use_case.process_document(sample_document)

        # Assert
        assert result.status == DocumentStatus.PROCESSED
        # No contextual LLM calls should have been made
        assert document_processing_use_case.contextual_llm is None

    @pytest.mark.asyncio
    @patch('app.core.use_cases.simple_document.settings')
    async def test_contextual_retrieval_failure_continues_processing(
        self,
        mock_settings,
        mock_document_repository,
        mock_document_processor,
        mock_embedding_service,
        mock_vector_store,
        mock_collection_repository,
        mock_llm_service,
        mock_bm25_service,
        sample_document
    ):
        """Test that processing continues if contextual retrieval fails."""
        # Arrange
        mock_settings.ENABLE_CONTEXTUAL_RETRIEVAL = True
        contextual_llm = AsyncMock()
        contextual_llm.generate_response.side_effect = Exception("LLM service unavailable")

        use_case = SimplifiedDocumentProcessingUseCase(
            document_repo=mock_document_repository,
            document_processor=mock_document_processor,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            collection_repo=mock_collection_repository,
            llm_service=mock_llm_service,
            contextual_llm=contextual_llm,
            bm25_service=mock_bm25_service
        )

        # Act - should not raise exception
        result = await use_case.process_document(sample_document)

        # Assert - processing should complete with original chunks
        assert result.status == DocumentStatus.PROCESSED
        assert result.chunk_count == 3


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_document_processor_failure(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_processor
    ):
        """Test handling of document processor failure."""
        # Arrange
        mock_document_processor.process_document.side_effect = Exception("PDF parsing failed")

        # Act & Assert
        with pytest.raises(DocumentProcessingError, match="Document processing failed"):
            await document_processing_use_case.process_document(sample_document)

    @pytest.mark.asyncio
    async def test_no_chunks_created_raises_error(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_processor
    ):
        """Test that error is raised if no chunks are created."""
        # Arrange
        mock_document_processor.process_document.return_value = ([], "")

        # Act & Assert
        with pytest.raises(DocumentProcessingError, match="No chunks created"):
            await document_processing_use_case.process_document(sample_document)

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(
        self,
        document_processing_use_case,
        sample_document,
        mock_embedding_service
    ):
        """Test handling of embedding generation failure."""
        # Arrange
        mock_embedding_service.generate_embeddings_batch.side_effect = Exception("Ollama service down")

        # Act & Assert
        with pytest.raises(DocumentProcessingError, match="Embedding generation failed"):
            await document_processing_use_case.process_document(sample_document)

    @pytest.mark.asyncio
    async def test_embedding_count_mismatch(
        self,
        document_processing_use_case,
        sample_document,
        mock_embedding_service
    ):
        """Test that error is raised if embedding count doesn't match chunk count."""
        # Arrange - return fewer embeddings than chunks
        async def bad_generate_embeddings(texts):
            return [EmbeddingVector(values=[0.1, 0.2], model_name="test", dimensions=2)]

        mock_embedding_service.generate_embeddings_batch.side_effect = bad_generate_embeddings

        # Act & Assert
        with pytest.raises(DocumentProcessingError):
            await document_processing_use_case.process_document(sample_document)


class TestCleanupOnFailure:
    """Tests for document status cleanup on failure."""

    @pytest.mark.asyncio
    async def test_document_marked_as_failed_on_error(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_processor,
        mock_document_repository
    ):
        """Test that document is marked as failed when processing fails."""
        # Arrange
        sample_document.mark_processing()  # Simulate processing state
        mock_document_processor.process_document.side_effect = Exception("Processing failed")

        # Act
        with pytest.raises(DocumentProcessingError):
            await document_processing_use_case.process_document(sample_document)

        # Assert - document should be marked as failed in finally block
        # Check the last save call
        last_save_call = mock_document_repository.save.call_args_list[-1]
        saved_doc = last_save_call[0][0]
        assert saved_doc.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_cleanup_handles_save_failure_gracefully(
        self,
        document_processing_use_case,
        sample_document,
        mock_document_processor,
        mock_document_repository
    ):
        """Test that cleanup failure doesn't mask the original error."""
        # Arrange
        sample_document.mark_processing()
        mock_document_processor.process_document.side_effect = Exception("Processing failed")

        # Make the final save (in cleanup) fail
        save_call_count = [0]
        original_save = mock_document_repository.save

        async def save_with_failure(doc):
            save_call_count[0] += 1
            if save_call_count[0] > 1:  # Fail on second call (cleanup)
                raise Exception("Database connection lost")
            return await original_save(doc)

        mock_document_repository.save.side_effect = save_with_failure

        # Act & Assert - should still raise original DocumentProcessingError
        with pytest.raises(DocumentProcessingError, match="Processing failed"):
            await document_processing_use_case.process_document(sample_document)


class TestBatchProcessing:
    """Tests for batch processing of embeddings."""

    @pytest.mark.asyncio
    async def test_embeddings_generated_in_batch(
        self,
        document_processing_use_case,
        sample_document,
        mock_embedding_service
    ):
        """Test that all chunk embeddings are generated in a single batch call."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert - should be called once with all texts
        assert mock_embedding_service.generate_embeddings_batch.call_count == 1
        call_args = mock_embedding_service.generate_embeddings_batch.call_args
        texts = call_args[0][0]
        assert len(texts) == 3  # All 3 chunks in one batch

    @pytest.mark.asyncio
    async def test_chunks_have_embedding_vectors_after_processing(
        self,
        document_processing_use_case,
        sample_document,
        mock_vector_store
    ):
        """Test that chunks have embedding vectors attached after generation."""
        # Act
        await document_processing_use_case.process_document(sample_document)

        # Assert - check chunks passed to vector store have embeddings
        call_args = mock_vector_store.add_chunks.call_args
        chunks = call_args[0][0]

        assert len(chunks) == 3
        # Each chunk should have embedding_vector set
        assert all(chunk.embedding_vector is not None for chunk in chunks)
        assert all(isinstance(chunk.embedding_vector, list) for chunk in chunks)
