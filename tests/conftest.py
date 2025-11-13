"""
Shared pytest fixtures for all tests.

This module provides reusable fixtures for:
- Mock repositories (collection, document, chat, evaluation)
- Mock services (embedding, LLM, vector store, BM25, rank fusion)
- FastAPI TestClient
- In-memory databases (SQLite, Chroma)
- Sample domain entities
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.domain.entities.chat import ChatMessage, ChatSession
from app.core.domain.entities.collection import Collection
from app.core.domain.entities.document import Document
from app.core.domain.entities.document_chunk import DocumentChunk
from app.core.domain.entities.structured_element import StructuredElement
from app.core.domain.entities.evaluation import EvaluationRun
from app.core.domain.value_objects.embedding import EmbeddingVector
from app.core.ports.bm25_service import BM25Service
from app.core.ports.embedding_service import EmbeddingService
from app.core.ports.llm_service import LLMService
from app.core.ports.rank_fusion_service import RankFusionService
from app.core.ports.vector_store import VectorStore


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests (session scope)."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# SAMPLE DOMAIN ENTITIES
# ============================================================================

@pytest.fixture
def sample_collection() -> Collection:
    """Sample collection entity."""
    return Collection(
        id="collection-123",
        name="Test Collection",
        description="A test collection",
        color="#3B82F6",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        updated_at=datetime(2024, 1, 1, 12, 0, 0),
        document_count=0
    )


@pytest.fixture
def sample_document(sample_collection: Collection) -> Document:
    """Sample document entity."""
    from app.core.domain.entities.document import DocumentStatus, DocumentType
    return Document(
        id="doc-456",
        filename="test_document.pdf",
        original_filename="original_test.pdf",
        file_path="/data/uploads/test_document.pdf",
        file_size=1024,
        document_type=DocumentType.PDF,
        collection_id=sample_collection.id,
        status=DocumentStatus.PROCESSED,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        processed_at=datetime(2024, 1, 1, 12, 5, 0),
        metadata={},
        chunk_count=10
    )


@pytest.fixture
def sample_structured_elements() -> List[StructuredElement]:
    """Sample structured elements from document processor."""
    return [
        StructuredElement(
            content="Introduction",
            element_type="heading",
            level=1,
            page_number=1,
            bbox=None,
        ),
        StructuredElement(
            content="This is the first paragraph of the document.",
            element_type="paragraph",
            level=None,
            page_number=1,
            bbox=None,
        ),
        StructuredElement(
            content="This is the second paragraph with more content.",
            element_type="paragraph",
            level=None,
            page_number=1,
            bbox=None,
        ),
    ]


@pytest.fixture
def sample_chunks(sample_document: Document) -> List[DocumentChunk]:
    """Sample document chunks."""
    return [
        DocumentChunk(
            id="chunk-1",
            document_id=sample_document.id,
            collection_id=sample_document.collection_id,
            content="This is the first chunk of text.",
            chunk_index=0,
            chunk_type="paragraph",
            parent_chunk_id=None,
            metadata={"page": 1, "heading": "Introduction"},
            embedding_vector=None,
        ),
        DocumentChunk(
            id="chunk-2",
            document_id=sample_document.id,
            collection_id=sample_document.collection_id,
            content="This is the second chunk of text.",
            chunk_index=1,
            chunk_type="paragraph",
            parent_chunk_id=None,
            metadata={"page": 1, "heading": "Introduction"},
            embedding_vector=None,
        ),
    ]


@pytest.fixture
def sample_embeddings() -> List[EmbeddingVector]:
    """Sample embedding vectors."""
    return [
        EmbeddingVector(values=[0.1, 0.2, 0.3, 0.4, 0.5], model_name="test-model", dimensions=5),
        EmbeddingVector(values=[0.6, 0.7, 0.8, 0.9, 1.0], model_name="test-model", dimensions=5),
    ]


@pytest.fixture
def sample_chat_session(sample_collection: Collection) -> ChatSession:
    """Sample chat session."""
    return ChatSession(
        id="session-789",
        collection_id=sample_collection.id,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_chat_messages(sample_chat_session: ChatSession) -> List[ChatMessage]:
    """Sample chat messages."""
    return [
        ChatMessage(
            id="msg-1",
            session_id=sample_chat_session.id,
            role="user",
            content="What is this document about?",
            created_at=datetime(2024, 1, 1, 12, 1, 0),
        ),
        ChatMessage(
            id="msg-2",
            session_id=sample_chat_session.id,
            role="assistant",
            content="This document discusses...",
            created_at=datetime(2024, 1, 1, 12, 1, 5),
        ),
    ]


@pytest.fixture
def sample_evaluation_run(sample_collection: Collection) -> EvaluationRun:
    """Sample evaluation run."""
    return EvaluationRun(
        id="eval-run-1",
        collection_id=sample_collection.id,
        queries=["Query 1", "Query 2"],
        ground_truths=["Answer 1", "Answer 2"],
        status="in_progress",
        total_queries=2,
        completed_queries=0,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


# ============================================================================
# MOCK SERVICES
# ============================================================================

@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Mock embedding service."""
    service = AsyncMock(spec=EmbeddingService)

    # Default behavior: return dummy embeddings
    async def generate_embedding(text: str) -> EmbeddingVector:
        # Generate deterministic embedding based on text length
        dim = 5  # Small dimension for testing
        values = [float(i + len(text) % 10) / 10 for i in range(dim)]
        return EmbeddingVector(values=values, model_name="test-model", dimensions=dim)

    async def generate_embeddings_batch(texts: List[str]) -> List[EmbeddingVector]:
        return [await generate_embedding(text) for text in texts]

    service.generate_embedding.side_effect = generate_embedding
    service.generate_embeddings_batch.side_effect = generate_embeddings_batch

    return service


@pytest.fixture
def mock_llm_service() -> LLMService:
    """Mock LLM service."""
    service = AsyncMock(spec=LLMService)

    # Default behavior: return canned responses
    async def generate_response(prompt: str, **kwargs) -> str:
        return "This is a generated response."

    async def generate_chat_response(messages: List[dict], **kwargs) -> str:
        return "This is a chat response."

    service.generate_response.side_effect = generate_response
    service.generate_chat_response.side_effect = generate_chat_response

    return service


@pytest.fixture
def mock_vector_store() -> VectorStore:
    """Mock vector store."""
    store = AsyncMock(spec=VectorStore)

    # In-memory storage for testing
    stored_chunks = []

    async def add_chunks(chunks: List[DocumentChunk], embeddings: List[EmbeddingVector], **kwargs):
        for chunk, embedding in zip(chunks, embeddings):
            stored_chunks.append((chunk, embedding))

    async def search(query_embedding: EmbeddingVector, collection_id: str = None, top_k: int = 5, **kwargs):
        # Return stored chunks (simplified)
        results = [(chunk, 0.9 - i * 0.1) for i, (chunk, _) in enumerate(stored_chunks[:top_k])]
        return results

    async def delete_by_document(document_id: str):
        nonlocal stored_chunks
        stored_chunks = [(c, e) for c, e in stored_chunks if c.document_id != document_id]

    store.add_chunks.side_effect = add_chunks
    store.search.side_effect = search
    store.delete_by_document.side_effect = delete_by_document
    store._stored_chunks = stored_chunks  # For inspection in tests

    return store


@pytest.fixture
def mock_bm25_service() -> BM25Service:
    """Mock BM25 service."""
    service = AsyncMock(spec=BM25Service)

    async def search(query: str, collection_id: str = None, top_k: int = 5):
        # Return dummy results
        return [
            ("chunk-1", 2.5),
            ("chunk-2", 2.0),
        ]

    service.search.side_effect = search

    return service


@pytest.fixture
def mock_rank_fusion_service() -> RankFusionService:
    """Mock rank fusion service."""
    service = Mock(spec=RankFusionService)

    def fuse_results(vector_results: List, bm25_results: List, alpha: float = 0.5):
        # Simple fusion: combine and deduplicate
        all_chunk_ids = set()
        fused = []

        for chunk, score in vector_results:
            if chunk.id not in all_chunk_ids:
                all_chunk_ids.add(chunk.id)
                fused.append((chunk, score * alpha))

        for chunk_id, score in bm25_results:
            if chunk_id not in all_chunk_ids:
                all_chunk_ids.add(chunk_id)
                # Find chunk by id (simplified)
                fused.append((chunk_id, score * (1 - alpha)))

        return sorted(fused, key=lambda x: x[1], reverse=True)

    service.fuse_results.side_effect = fuse_results

    return service


# ============================================================================
# MOCK REPOSITORIES
# ============================================================================

@pytest.fixture
def mock_collection_repository():
    """Mock collection repository."""
    repo = AsyncMock()

    # In-memory storage
    collections = {}

    async def save(collection: Collection) -> Collection:
        collections[collection.id] = collection
        return collection

    async def find_by_id(collection_id: str):
        return collections.get(collection_id)

    async def find_all():
        return list(collections.values())

    async def delete(collection_id: str):
        if collection_id in collections:
            del collections[collection_id]

    repo.save.side_effect = save
    repo.find_by_id.side_effect = find_by_id
    repo.find_all.side_effect = find_all
    repo.delete.side_effect = delete
    repo._collections = collections  # For inspection

    return repo


@pytest.fixture
def mock_document_repository():
    """Mock document repository."""
    repo = AsyncMock()

    # In-memory storage
    documents = {}

    async def save(document: Document) -> Document:
        documents[document.id] = document
        return document

    async def find_by_id(document_id: str):
        return documents.get(document_id)

    async def find_all(collection_id: str = None):
        if collection_id:
            return [d for d in documents.values() if d.collection_id == collection_id]
        return list(documents.values())

    async def delete(document_id: str):
        if document_id in documents:
            del documents[document_id]

    repo.save.side_effect = save
    repo.find_by_id.side_effect = find_by_id
    repo.find_all.side_effect = find_all
    repo.delete.side_effect = delete
    repo._documents = documents  # For inspection

    return repo


@pytest.fixture
def mock_chat_repository():
    """Mock chat repository."""
    repo = AsyncMock()

    # In-memory storage
    sessions = {}
    messages = {}

    async def save_session(session: ChatSession) -> ChatSession:
        sessions[session.id] = session
        return session

    async def find_session_by_id(session_id: str):
        return sessions.get(session_id)

    async def save_message(message: ChatMessage) -> ChatMessage:
        if message.session_id not in messages:
            messages[message.session_id] = []
        messages[message.session_id].append(message)
        return message

    async def find_messages_by_session_id(session_id: str):
        return messages.get(session_id, [])

    repo.save_session.side_effect = save_session
    repo.find_session_by_id.side_effect = find_session_by_id
    repo.save_message.side_effect = save_message
    repo.find_messages_by_session_id.side_effect = find_messages_by_session_id
    repo._sessions = sessions
    repo._messages = messages

    return repo


@pytest.fixture
def mock_evaluation_repository():
    """Mock evaluation repository."""
    repo = AsyncMock()

    # In-memory storage
    runs = {}
    results = {}

    async def save_run(run: EvaluationRun) -> EvaluationRun:
        runs[run.id] = run
        return run

    async def find_run_by_id(run_id: str):
        return runs.get(run_id)

    async def save_result(result: EvaluationResult) -> EvaluationResult:
        if result.run_id not in results:
            results[result.run_id] = []
        results[result.run_id].append(result)
        return result

    repo.save_run.side_effect = save_run
    repo.find_run_by_id.side_effect = find_run_by_id
    repo.save_result.side_effect = save_result
    repo._runs = runs
    repo._results = results

    return repo


# ============================================================================
# FASTAPI TEST CLIENT
# ============================================================================

@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """FastAPI test client.

    Note: This fixture imports app lazily to avoid initialization
    of real services during test collection.
    """
    from app.main import app

    with TestClient(app) as client:
        yield client


# ============================================================================
# DATABASE FIXTURES (IN-MEMORY)
# ============================================================================

@pytest.fixture
async def in_memory_db_engine():
    """Create in-memory SQLite database engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create tables
    from app.adapters.persistence.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(in_memory_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Async database session for testing."""
    async_session = sessionmaker(
        in_memory_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def mock_file_upload():
    """Mock file upload object."""
    mock_file = MagicMock()
    mock_file.filename = "test.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.file = MagicMock()
    mock_file.file.read.return_value = b"fake pdf content"
    return mock_file
