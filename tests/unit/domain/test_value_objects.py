"""
Unit tests for domain value objects (EmbeddingVector, SearchQuery, etc.).

Tests cover:
- Validation logic in __post_init__
- Immutability
- Computed properties
"""

import pytest

from app.core.domain.value_objects.embedding import EmbeddingVector, SearchQuery


class TestEmbeddingVector:
    """Tests for EmbeddingVector value object."""

    def test_create_embedding_vector(self):
        """Test creating a valid embedding vector."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding = EmbeddingVector(
            values=values,
            model_name="test-model",
            dimensions=5
        )

        assert embedding.values == values
        assert embedding.model_name == "test-model"
        assert embedding.dimensions == 5

    def test_embedding_vector_validates_dimensions(self):
        """Test that embedding vector validates dimensions match."""
        with pytest.raises(ValueError, match="Expected 5 dimensions, got 3"):
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model_name="test-model",
                dimensions=5
            )

    def test_embedding_vector_magnitude(self):
        """Test calculating vector magnitude."""
        embedding = EmbeddingVector(
            values=[3.0, 4.0],
            model_name="test-model",
            dimensions=2
        )

        # Magnitude of [3, 4] should be 5 (pythagorean theorem)
        assert embedding.magnitude == 5.0

    def test_embedding_vector_is_immutable(self):
        """Test that embedding vector is frozen (immutable)."""
        embedding = EmbeddingVector(
            values=[0.1, 0.2],
            model_name="test-model",
            dimensions=2
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            embedding.values = [0.3, 0.4]

    def test_embedding_vector_zero_magnitude(self):
        """Test magnitude of zero vector."""
        embedding = EmbeddingVector(
            values=[0.0, 0.0, 0.0],
            model_name="test-model",
            dimensions=3
        )

        assert embedding.magnitude == 0.0


class TestSearchQuery:
    """Tests for SearchQuery value object."""

    def test_create_search_query(self):
        """Test creating a valid search query."""
        query = SearchQuery(
            text="What is machine learning?",
            collection_id="collection-123",
            filters={"document_id": "doc-456"},
            top_k=5,
            similarity_threshold=0.8
        )

        assert query.text == "What is machine learning?"
        assert query.collection_id == "collection-123"
        assert query.filters == {"document_id": "doc-456"}
        assert query.top_k == 5
        assert query.similarity_threshold == 0.8

    def test_search_query_default_values(self):
        """Test search query with default top_k and similarity_threshold."""
        query = SearchQuery(
            text="test query",
            collection_id=None,
            filters={}
        )

        assert query.top_k == 10  # Default
        assert query.similarity_threshold == 0.7  # Default

    def test_search_query_validates_top_k_positive(self):
        """Test that top_k must be positive."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            SearchQuery(
                text="test",
                collection_id=None,
                filters={},
                top_k=0
            )

        with pytest.raises(ValueError, match="top_k must be positive"):
            SearchQuery(
                text="test",
                collection_id=None,
                filters={},
                top_k=-5
            )

    def test_search_query_validates_similarity_threshold_range(self):
        """Test that similarity_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            SearchQuery(
                text="test",
                collection_id=None,
                filters={},
                similarity_threshold=-0.1
            )

        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            SearchQuery(
                text="test",
                collection_id=None,
                filters={},
                similarity_threshold=1.5
            )

    def test_search_query_boundary_values(self):
        """Test search query with boundary values for similarity_threshold."""
        # 0.0 should be valid
        query1 = SearchQuery(
            text="test",
            collection_id=None,
            filters={},
            similarity_threshold=0.0
        )
        assert query1.similarity_threshold == 0.0

        # 1.0 should be valid
        query2 = SearchQuery(
            text="test",
            collection_id=None,
            filters={},
            similarity_threshold=1.0
        )
        assert query2.similarity_threshold == 1.0

    def test_search_query_is_immutable(self):
        """Test that search query is frozen (immutable)."""
        query = SearchQuery(
            text="test",
            collection_id=None,
            filters={}
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            query.text = "modified"
