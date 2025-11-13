"""
Unit tests for domain entities (Document, Collection, Chat, etc.).

Tests cover:
- Factory methods
- State transitions
- Business logic validation
- Entity behaviors
"""

import pytest
from datetime import datetime, UTC
from uuid import UUID

from app.core.domain.entities.collection import Collection, CollectionWithDocuments
from app.core.domain.entities.document import Document, DocumentStatus, DocumentType
from app.core.domain.entities.chat import ChatMessage, ChatSession
from app.core.domain.entities.document_chunk import DocumentChunk


class TestCollection:
    """Tests for Collection entity."""

    def test_create_collection(self):
        """Test creating a new collection using factory method."""
        collection = Collection.create(
            name="Test Collection",
            description="A test collection",
            color="#FF0000"
        )

        assert collection.id is not None
        assert UUID(collection.id)  # Valid UUID
        assert collection.name == "Test Collection"
        assert collection.description == "A test collection"
        assert collection.color == "#FF0000"
        assert isinstance(collection.created_at, datetime)
        assert isinstance(collection.updated_at, datetime)
        assert collection.document_count == 0

    def test_create_collection_default_color(self):
        """Test creating collection with default color."""
        collection = Collection.create(
            name="Test",
            description="Test"
        )

        assert collection.color == "#3B82F6"  # Default blue

    def test_increment_document_count(self):
        """Test incrementing document count."""
        collection = Collection.create(name="Test", description="Test")
        initial_updated_at = collection.updated_at

        collection.increment_document_count()

        assert collection.document_count == 1
        assert collection.updated_at > initial_updated_at

    def test_decrement_document_count(self):
        """Test decrementing document count."""
        collection = Collection.create(name="Test", description="Test")
        collection.document_count = 5

        collection.decrement_document_count()

        assert collection.document_count == 4

    def test_decrement_document_count_doesnt_go_negative(self):
        """Test that decrement doesn't allow negative counts."""
        collection = Collection.create(name="Test", description="Test")
        assert collection.document_count == 0

        collection.decrement_document_count()

        assert collection.document_count == 0  # Should stay at 0

    def test_update_document_count_positive(self):
        """Test updating document count with positive increment."""
        collection = Collection.create(name="Test", description="Test")

        collection.update_document_count(5)

        assert collection.document_count == 5

    def test_update_document_count_negative(self):
        """Test updating document count with negative increment."""
        collection = Collection.create(name="Test", description="Test")
        collection.document_count = 10

        collection.update_document_count(-3)

        assert collection.document_count == 7

    def test_update_document_count_raises_on_negative_result(self):
        """Test that update raises error if result would be negative."""
        collection = Collection.create(name="Test", description="Test")
        collection.document_count = 2

        with pytest.raises(ValueError, match="Document count cannot be negative"):
            collection.update_document_count(-5)


class TestDocument:
    """Tests for Document entity."""

    def test_create_document(self):
        """Test creating a new document using factory method."""
        document = Document.create(
            filename="test.pdf",
            original_filename="original_test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            document_type=DocumentType.PDF,
            collection_id="collection-123",
            metadata={"author": "Test Author"}
        )

        assert document.id is not None
        assert UUID(document.id)  # Valid UUID
        assert document.filename == "test.pdf"
        assert document.original_filename == "original_test.pdf"
        assert document.file_path == "/path/to/test.pdf"
        assert document.file_size == 1024
        assert document.document_type == DocumentType.PDF
        assert document.collection_id == "collection-123"
        assert document.status == DocumentStatus.UPLOADED
        assert document.processed_at is None
        assert document.metadata == {"author": "Test Author"}
        assert document.chunk_count == 0

    def test_create_document_default_metadata(self):
        """Test creating document with default empty metadata."""
        document = Document.create(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            document_type=DocumentType.PDF,
            collection_id="collection-123"
        )

        assert document.metadata == {}

    def test_mark_processing(self):
        """Test marking document as processing."""
        document = Document.create(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path",
            file_size=100,
            document_type=DocumentType.PDF,
            collection_id="col-1"
        )

        document.mark_processing()

        assert document.status == DocumentStatus.PROCESSING

    def test_mark_processed(self):
        """Test marking document as successfully processed."""
        document = Document.create(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path",
            file_size=100,
            document_type=DocumentType.PDF,
            collection_id="col-1"
        )
        document.mark_processing()

        document.mark_processed(chunk_count=10)

        assert document.status == DocumentStatus.PROCESSED
        assert document.chunk_count == 10
        assert document.processed_at is not None
        assert isinstance(document.processed_at, datetime)

    def test_mark_failed(self):
        """Test marking document as failed."""
        document = Document.create(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path",
            file_size=100,
            document_type=DocumentType.PDF,
            collection_id="col-1"
        )
        document.mark_processing()

        document.mark_failed()

        assert document.status == DocumentStatus.FAILED


class TestChatSession:
    """Tests for ChatSession entity."""

    def test_create_chat_session(self):
        """Test creating a new chat session."""
        session = ChatSession.create(
            name="Test Session",
            collection_id="collection-123"
        )

        assert session.id is not None
        assert UUID(session.id)  # Valid UUID
        assert session.name == "Test Session"
        assert session.collection_id == "collection-123"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.message_count == 0

    def test_create_chat_session_without_collection(self):
        """Test creating session without collection ID."""
        session = ChatSession.create(name="General Chat")

        assert session.collection_id is None


class TestChatMessage:
    """Tests for ChatMessage entity."""

    def test_create_chat_message(self):
        """Test creating a new chat message."""
        message = ChatMessage.create(
            session_id="session-123",
            user_message="What is this about?",
            assistant_response="This document discusses...",
            retrieved_chunks=["chunk-1", "chunk-2"],
            response_time_ms=500,
            collection_id="collection-123"
        )

        assert message.id is not None
        assert UUID(message.id)  # Valid UUID
        assert message.session_id == "session-123"
        assert message.user_message == "What is this about?"
        assert message.assistant_response == "This document discusses..."
        assert message.retrieved_chunks == ["chunk-1", "chunk-2"]
        assert message.response_time_ms == 500
        assert message.collection_id == "collection-123"
        assert isinstance(message.created_at, datetime)

    def test_create_chat_message_without_collection(self):
        """Test creating message without collection ID."""
        message = ChatMessage.create(
            session_id="session-123",
            user_message="Hello",
            assistant_response="Hi!",
            retrieved_chunks=[],
            response_time_ms=100
        )

        assert message.collection_id is None


class TestDocumentChunk:
    """Tests for DocumentChunk entity."""

    def test_create_document_chunk(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-123",
            collection_id="collection-123",
            content="This is chunk content",
            chunk_index=0,
            chunk_type="paragraph",
            parent_chunk_id=None,
            metadata={"page": 1, "heading": "Introduction"},
            embedding_vector=None
        )

        assert chunk.id == "chunk-1"
        assert chunk.document_id == "doc-123"
        assert chunk.collection_id == "collection-123"
        assert chunk.content == "This is chunk content"
        assert chunk.chunk_index == 0
        assert chunk.chunk_type == "paragraph"
        assert chunk.parent_chunk_id is None
        assert chunk.metadata == {"page": 1, "heading": "Introduction"}
        assert chunk.embedding_vector is None

    def test_create_document_chunk_with_parent(self):
        """Test creating chunk with parent (hierarchical chunking)."""
        chunk = DocumentChunk(
            id="chunk-2",
            document_id="doc-123",
            collection_id="collection-123",
            content="Child chunk content",
            chunk_index=1,
            chunk_type="paragraph",
            parent_chunk_id="chunk-1",
            metadata={},
            embedding_vector=None
        )

        assert chunk.parent_chunk_id == "chunk-1"

    def test_create_document_chunk_factory_method(self):
        """Test creating chunk using factory method."""
        chunk = DocumentChunk.create(
            document_id="doc-456",
            collection_id="col-789",
            content="Factory created chunk",
            chunk_index=0,
            chunk_type="heading",
            parent_chunk_id=None,
            metadata={"level": 1}
        )

        assert chunk.id is not None  # Auto-generated
        assert chunk.document_id == "doc-456"
        assert chunk.collection_id == "col-789"
        assert chunk.content == "Factory created chunk"
        assert chunk.chunk_type == "heading"
        assert chunk.metadata == {"level": 1}


class TestCollectionWithDocuments:
    """Tests for CollectionWithDocuments data class."""

    def test_collection_with_documents(self, sample_collection: Collection, sample_document: Document):
        """Test creating CollectionWithDocuments."""
        cwd = CollectionWithDocuments(
            collection=sample_collection,
            documents=[sample_document]
        )

        assert cwd.collection == sample_collection
        assert len(cwd.documents) == 1
        assert cwd.documents[0] == sample_document

    def test_to_dict(self, sample_collection: Collection, sample_document: Document):
        """Test converting to dictionary."""
        cwd = CollectionWithDocuments(
            collection=sample_collection,
            documents=[sample_document]
        )

        result = cwd.to_dict()

        assert "collection" in result
        assert "documents" in result
        assert result["collection"]["id"] == sample_collection.id
        assert result["collection"]["name"] == sample_collection.name
        assert len(result["documents"]) == 1
        assert result["documents"][0]["id"] == sample_document.id
        assert result["documents"][0]["filename"] == sample_document.filename
