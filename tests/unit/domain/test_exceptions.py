"""
Unit tests for domain exceptions.

Tests cover:
- Exception hierarchy
- Custom exception attributes
- Error message handling
"""

import pytest

from app.core.domain.exceptions import (
    DomainException,
    DocumentNotFoundError,
    CollectionNotFoundError,
    DocumentProcessingError,
    DocumentDeletionError,
    VectorStoreDeletionError,
    FileDeletionError,
    DatabaseDeletionError,
    CollectionCountUpdateError,
    PartialDocumentDeletionError,
    EmbeddingGenerationError,
    InvalidDocumentTypeError,
    ChatSessionNotFoundError,
    ChatSessionProcessingError,
    TokenizationError,
    CollectionSummaryError,
    IntentClassificationError,
    EvaluationNotFoundError,
    EvaluationInitializationError,
    JudgeServiceError,
    TestCaseLoadError,
)


class TestDomainExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_domain_exception(self):
        """Test that all custom exceptions inherit from DomainException."""
        exceptions = [
            DocumentNotFoundError,
            CollectionNotFoundError,
            DocumentProcessingError,
            DocumentDeletionError,
            EmbeddingGenerationError,
            InvalidDocumentTypeError,
            ChatSessionNotFoundError,
            ChatSessionProcessingError,
            TokenizationError,
            CollectionSummaryError,
            IntentClassificationError,
            EvaluationNotFoundError,
            EvaluationInitializationError,
            JudgeServiceError,
            TestCaseLoadError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, DomainException)

    def test_domain_exception_inherits_from_exception(self):
        """Test that DomainException inherits from base Exception."""
        assert issubclass(DomainException, Exception)


class TestBasicExceptions:
    """Tests for basic exception behavior."""

    def test_document_not_found_error(self):
        """Test DocumentNotFoundError."""
        error = DocumentNotFoundError("Document doc-123 not found")

        assert str(error) == "Document doc-123 not found"
        assert isinstance(error, DomainException)

    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError."""
        error = CollectionNotFoundError("Collection col-456 not found")

        assert str(error) == "Collection col-456 not found"

    def test_document_processing_error(self):
        """Test DocumentProcessingError."""
        error = DocumentProcessingError("Failed to process document")

        assert str(error) == "Failed to process document"


class TestDocumentDeletionErrors:
    """Tests for document deletion error hierarchy."""

    def test_document_deletion_error_inheritance(self):
        """Test that all deletion errors inherit from DocumentDeletionError."""
        deletion_errors = [
            VectorStoreDeletionError,
            FileDeletionError,
            DatabaseDeletionError,
            CollectionCountUpdateError,
            PartialDocumentDeletionError,
        ]

        for exc_class in deletion_errors:
            assert issubclass(exc_class, DocumentDeletionError)

    def test_document_deletion_error_with_results(self):
        """Test DocumentDeletionError with deletion_results attribute."""
        deletion_results = {
            "vector_store": True,
            "file_system": True,
            "database": False
        }

        error = DocumentDeletionError(
            "Failed to delete document",
            deletion_results=deletion_results
        )

        assert str(error) == "Failed to delete document"
        assert error.deletion_results == deletion_results

    def test_document_deletion_error_default_results(self):
        """Test DocumentDeletionError with default empty results."""
        error = DocumentDeletionError("Deletion failed")

        assert error.deletion_results == {}

    def test_partial_document_deletion_error(self):
        """Test PartialDocumentDeletionError with failed_operations."""
        deletion_results = {
            "vector_store": True,
            "file_system": False,
            "database": True
        }
        failed_operations = ["file_system"]

        error = PartialDocumentDeletionError(
            "Partial deletion failure",
            deletion_results=deletion_results,
            failed_operations=failed_operations
        )

        assert str(error) == "Partial deletion failure"
        assert error.deletion_results == deletion_results
        assert error.failed_operations == failed_operations

    def test_vector_store_deletion_error(self):
        """Test VectorStoreDeletionError."""
        error = VectorStoreDeletionError(
            "Failed to delete from vector store",
            deletion_results={"vector_store": False}
        )

        assert "Failed to delete from vector store" in str(error)
        assert isinstance(error, DocumentDeletionError)


class TestChatExceptions:
    """Tests for chat-related exceptions."""

    def test_chat_session_not_found_error(self):
        """Test ChatSessionNotFoundError."""
        error = ChatSessionNotFoundError("Session session-789 not found")

        assert str(error) == "Session session-789 not found"

    def test_chat_session_processing_error(self):
        """Test ChatSessionProcessingError."""
        error = ChatSessionProcessingError("Failed to process chat message")

        assert str(error) == "Failed to process chat message"


class TestEvaluationExceptions:
    """Tests for evaluation-related exceptions."""

    def test_evaluation_not_found_error(self):
        """Test EvaluationNotFoundError."""
        error = EvaluationNotFoundError("Evaluation eval-123 not found")

        assert str(error) == "Evaluation eval-123 not found"

    def test_evaluation_initialization_error(self):
        """Test EvaluationInitializationError."""
        error = EvaluationInitializationError("Failed to initialize test collection")

        assert str(error) == "Failed to initialize test collection"

    def test_judge_service_error(self):
        """Test JudgeServiceError."""
        error = JudgeServiceError("Judge evaluation failed")

        assert str(error) == "Judge evaluation failed"

    def test_test_case_load_error(self):
        """Test TestCaseLoadError."""
        error = TestCaseLoadError("Failed to load test_cases.json")

        assert str(error) == "Failed to load test_cases.json"


class TestMiscellaneousExceptions:
    """Tests for miscellaneous domain exceptions."""

    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError."""
        error = EmbeddingGenerationError("Ollama service unavailable")

        assert str(error) == "Ollama service unavailable"

    def test_invalid_document_type_error(self):
        """Test InvalidDocumentTypeError."""
        error = InvalidDocumentTypeError("Document type .xyz not supported")

        assert str(error) == "Document type .xyz not supported"

    def test_tokenization_error(self):
        """Test TokenizationError."""
        error = TokenizationError("Failed to tokenize text")

        assert str(error) == "Failed to tokenize text"

    def test_collection_summary_error(self):
        """Test CollectionSummaryError."""
        error = CollectionSummaryError("Failed to generate collection summary")

        assert str(error) == "Failed to generate collection summary"

    def test_intent_classification_error(self):
        """Test IntentClassificationError."""
        error = IntentClassificationError("Failed to classify intent")

        assert str(error) == "Failed to classify intent"


class TestExceptionRaising:
    """Tests for actually raising and catching exceptions."""

    def test_raise_and_catch_domain_exception(self):
        """Test raising and catching DomainException."""
        with pytest.raises(DomainException):
            raise DomainException("Test exception")

    def test_catch_specific_exception_as_domain_exception(self):
        """Test catching specific exception as DomainException."""
        with pytest.raises(DomainException):
            raise DocumentNotFoundError("Document not found")

    def test_catch_specific_exception_type(self):
        """Test catching specific exception type."""
        with pytest.raises(DocumentNotFoundError) as exc_info:
            raise DocumentNotFoundError("Document doc-123 not found")

        assert "doc-123" in str(exc_info.value)
