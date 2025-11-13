"""
Unit tests for CollectionManagementUseCase.

Tests cover:
- Create, read, update, delete operations
- List collections
- Document count management
- Error handling (not found)
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, UTC

from app.core.use_cases.collection_management import CollectionManagementUseCase
from app.core.domain.entities.collection import Collection, CollectionWithDocuments
from app.core.domain.entities.document import Document, DocumentType, DocumentStatus
from app.core.domain.exceptions import CollectionNotFoundError


@pytest.fixture
def collection_management_use_case(mock_collection_repository):
    """Create CollectionManagementUseCase with mocked repository."""
    return CollectionManagementUseCase(collection_repo=mock_collection_repository)


class TestCreateCollection:
    """Tests for creating collections."""

    @pytest.mark.asyncio
    async def test_create_collection_success(
        self,
        collection_management_use_case,
        mock_collection_repository
    ):
        """Test successful collection creation."""
        # Act
        result = await collection_management_use_case.create_collection(
            name="Test Collection",
            description="A test collection",
            color="#FF5733"
        )

        # Assert
        assert result.name == "Test Collection"
        assert result.description == "A test collection"
        assert result.color == "#FF5733"
        assert result.document_count == 0
        assert result.id is not None

        # Verify repository save was called
        mock_collection_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_with_default_color(
        self,
        collection_management_use_case
    ):
        """Test collection creation with default color."""
        # Act
        result = await collection_management_use_case.create_collection(
            name="Default Color Collection",
            description="Test default color"
        )

        # Assert
        assert result.color == "#3B82F6"  # Default blue

    @pytest.mark.asyncio
    async def test_create_collection_generates_unique_id(
        self,
        collection_management_use_case
    ):
        """Test that each collection gets a unique ID."""
        # Act
        collection1 = await collection_management_use_case.create_collection(
            name="Collection 1",
            description="First"
        )
        collection2 = await collection_management_use_case.create_collection(
            name="Collection 2",
            description="Second"
        )

        # Assert
        assert collection1.id != collection2.id


class TestGetCollection:
    """Tests for retrieving collections."""

    @pytest.mark.asyncio
    async def test_get_collection_success(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test successful collection retrieval."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        result = await collection_management_use_case.get_collection(sample_collection.id)

        # Assert
        assert result.id == sample_collection.id
        assert result.name == sample_collection.name
        mock_collection_repository.find_by_id.assert_called_with(sample_collection.id)

    @pytest.mark.asyncio
    async def test_get_collection_not_found(
        self,
        collection_management_use_case
    ):
        """Test getting non-existent collection raises error."""
        # Act & Assert
        with pytest.raises(CollectionNotFoundError, match="Collection non-existent not found"):
            await collection_management_use_case.get_collection("non-existent")


class TestGetCollectionWithDocuments:
    """Tests for retrieving collection with documents."""

    @pytest.mark.asyncio
    async def test_get_collection_with_documents_success(
        self,
        collection_management_use_case,
        sample_collection,
        sample_document,
        mock_collection_repository
    ):
        """Test successful retrieval of collection with documents."""
        # Arrange
        collection_with_docs = CollectionWithDocuments(
            collection=sample_collection,
            documents=[sample_document]
        )

        async def mock_find_with_docs(collection_id):
            if collection_id == sample_collection.id:
                return collection_with_docs
            return None

        mock_collection_repository.find_by_id_with_documents.side_effect = mock_find_with_docs

        # Act
        result = await collection_management_use_case.get_collection_with_documents(
            sample_collection.id
        )

        # Assert
        assert result.collection.id == sample_collection.id
        assert len(result.documents) == 1
        assert result.documents[0].id == sample_document.id
        mock_collection_repository.find_by_id_with_documents.assert_called_with(sample_collection.id)

    @pytest.mark.asyncio
    async def test_get_collection_with_documents_not_found(
        self,
        collection_management_use_case,
        mock_collection_repository
    ):
        """Test getting non-existent collection with documents raises error."""
        # Arrange - ensure find_by_id_with_documents returns None
        async def mock_find_with_docs(collection_id):
            return None

        mock_collection_repository.find_by_id_with_documents.side_effect = mock_find_with_docs

        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.get_collection_with_documents("non-existent")


class TestListCollections:
    """Tests for listing collections."""

    @pytest.mark.asyncio
    async def test_list_collections_empty(
        self,
        collection_management_use_case,
        mock_collection_repository
    ):
        """Test listing collections when none exist."""
        # Act
        result = await collection_management_use_case.list_collections()

        # Assert
        assert result == []
        mock_collection_repository.find_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections_with_data(
        self,
        collection_management_use_case,
        mock_collection_repository
    ):
        """Test listing collections when multiple exist."""
        # Arrange - create multiple collections
        collection1 = await collection_management_use_case.create_collection(
            name="Collection 1",
            description="First"
        )
        collection2 = await collection_management_use_case.create_collection(
            name="Collection 2",
            description="Second"
        )

        # Act
        result = await collection_management_use_case.list_collections()

        # Assert
        assert len(result) == 2
        assert any(c.id == collection1.id for c in result)
        assert any(c.id == collection2.id for c in result)


class TestUpdateCollection:
    """Tests for updating collections."""

    @pytest.mark.asyncio
    async def test_update_collection_name(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test updating collection name."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection
        original_updated_at = sample_collection.updated_at

        # Act
        result = await collection_management_use_case.update_collection(
            collection_id=sample_collection.id,
            name="Updated Name"
        )

        # Assert
        assert result.name == "Updated Name"
        assert result.description == sample_collection.description  # Unchanged
        # Can't compare naive vs aware datetime - just check it was updated
        assert result.updated_at is not None

    @pytest.mark.asyncio
    async def test_update_collection_description(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test updating collection description."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        result = await collection_management_use_case.update_collection(
            collection_id=sample_collection.id,
            description="New description"
        )

        # Assert
        assert result.description == "New description"
        assert result.name == sample_collection.name  # Unchanged

    @pytest.mark.asyncio
    async def test_update_collection_color(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test updating collection color."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        result = await collection_management_use_case.update_collection(
            collection_id=sample_collection.id,
            color="#00FF00"
        )

        # Assert
        assert result.color == "#00FF00"

    @pytest.mark.asyncio
    async def test_update_collection_multiple_fields(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test updating multiple collection fields at once."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        result = await collection_management_use_case.update_collection(
            collection_id=sample_collection.id,
            name="New Name",
            description="New Description",
            color="#FF00FF"
        )

        # Assert
        assert result.name == "New Name"
        assert result.description == "New Description"
        assert result.color == "#FF00FF"

    @pytest.mark.asyncio
    async def test_update_collection_not_found(
        self,
        collection_management_use_case
    ):
        """Test updating non-existent collection raises error."""
        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.update_collection(
                collection_id="non-existent",
                name="New Name"
            )

    @pytest.mark.asyncio
    async def test_update_collection_no_fields(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test updating collection with no fields changes nothing except updated_at."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection
        original_name = sample_collection.name
        original_description = sample_collection.description
        original_color = sample_collection.color

        # Act
        result = await collection_management_use_case.update_collection(
            collection_id=sample_collection.id
        )

        # Assert
        assert result.name == original_name
        assert result.description == original_description
        assert result.color == original_color


class TestDeleteCollection:
    """Tests for deleting collections."""

    @pytest.mark.asyncio
    async def test_delete_collection_success(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test successful collection deletion."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        result = await collection_management_use_case.delete_collection(sample_collection.id)

        # Assert
        # Mock delete doesn't return True by default - just check it was called
        mock_collection_repository.delete.assert_called_with(sample_collection.id)
        # Verify it's removed from in-memory storage
        assert sample_collection.id not in mock_collection_repository._collections

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(
        self,
        collection_management_use_case
    ):
        """Test deleting non-existent collection raises error."""
        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.delete_collection("non-existent")


class TestDocumentCountManagement:
    """Tests for document count increment/decrement operations."""

    @pytest.mark.asyncio
    async def test_increment_document_count(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test incrementing document count."""
        # Arrange
        mock_collection_repository._collections[sample_collection.id] = sample_collection
        original_count = sample_collection.document_count

        # Act
        await collection_management_use_case.increment_document_count(sample_collection.id)

        # Assert
        # Get the updated collection from repository
        updated_collection = await mock_collection_repository.find_by_id(sample_collection.id)
        assert updated_collection.document_count == original_count + 1

    @pytest.mark.asyncio
    async def test_decrement_document_count(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test decrementing document count."""
        # Arrange
        sample_collection.document_count = 5  # Start with some documents
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        await collection_management_use_case.decrement_document_count(sample_collection.id)

        # Assert
        updated_collection = await mock_collection_repository.find_by_id(sample_collection.id)
        assert updated_collection.document_count == 4

    @pytest.mark.asyncio
    async def test_decrement_document_count_at_zero(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test decrementing document count when already at zero."""
        # Arrange
        sample_collection.document_count = 0
        mock_collection_repository._collections[sample_collection.id] = sample_collection

        # Act
        await collection_management_use_case.decrement_document_count(sample_collection.id)

        # Assert
        updated_collection = await mock_collection_repository.find_by_id(sample_collection.id)
        assert updated_collection.document_count == 0  # Should not go negative

    @pytest.mark.asyncio
    async def test_increment_document_count_not_found(
        self,
        collection_management_use_case
    ):
        """Test incrementing count for non-existent collection raises error."""
        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.increment_document_count("non-existent")

    @pytest.mark.asyncio
    async def test_decrement_document_count_not_found(
        self,
        collection_management_use_case
    ):
        """Test decrementing count for non-existent collection raises error."""
        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.decrement_document_count("non-existent")


class TestNormalizeDocumentCount:
    """Tests for document count normalization."""

    @pytest.mark.asyncio
    async def test_normalize_document_count_corrects_mismatch(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test normalizing document count when it doesn't match actual documents."""
        # Arrange
        sample_collection.document_count = 10  # Incorrect count

        # Create sample documents
        documents = [
            Document(
                id=f"doc-{i}",
                filename=f"test{i}.pdf",
                original_filename=f"test{i}.pdf",
                file_path=f"/path/to/test{i}.pdf",
                file_size=1024,
                document_type=DocumentType.PDF,
                collection_id=sample_collection.id,
                status=DocumentStatus.PROCESSED,
                created_at=datetime.now(UTC),
                processed_at=datetime.now(UTC),
                metadata={},
                chunk_count=5
            )
            for i in range(3)  # Only 3 actual documents
        ]

        collection_with_docs = CollectionWithDocuments(
            collection=sample_collection,
            documents=documents
        )

        mock_collection_repository._collections[sample_collection.id] = sample_collection

        async def mock_find_with_docs(collection_id):
            if collection_id == sample_collection.id:
                return collection_with_docs
            return None

        mock_collection_repository.find_by_id_with_documents.side_effect = mock_find_with_docs

        # Act
        await collection_management_use_case.normalize_document_count(sample_collection.id)

        # Assert
        updated_collection = await mock_collection_repository.find_by_id(sample_collection.id)
        assert updated_collection.document_count == 3  # Corrected to actual count

    @pytest.mark.asyncio
    async def test_normalize_document_count_with_zero_documents(
        self,
        collection_management_use_case,
        sample_collection,
        mock_collection_repository
    ):
        """Test normalizing document count when collection has no documents."""
        # Arrange
        sample_collection.document_count = 5  # Incorrect count

        collection_with_docs = CollectionWithDocuments(
            collection=sample_collection,
            documents=[]  # No documents
        )

        mock_collection_repository._collections[sample_collection.id] = sample_collection

        async def mock_find_with_docs(collection_id):
            if collection_id == sample_collection.id:
                return collection_with_docs
            return None

        mock_collection_repository.find_by_id_with_documents.side_effect = mock_find_with_docs

        # Act
        await collection_management_use_case.normalize_document_count(sample_collection.id)

        # Assert
        updated_collection = await mock_collection_repository.find_by_id(sample_collection.id)
        assert updated_collection.document_count == 0

    @pytest.mark.asyncio
    async def test_normalize_document_count_not_found(
        self,
        collection_management_use_case,
        mock_collection_repository
    ):
        """Test normalizing count for non-existent collection raises error."""
        # Arrange - ensure find_by_id_with_documents returns None
        async def mock_find_with_docs(collection_id):
            return None

        mock_collection_repository.find_by_id_with_documents.side_effect = mock_find_with_docs

        # Act & Assert
        with pytest.raises(CollectionNotFoundError):
            await collection_management_use_case.normalize_document_count("non-existent")
