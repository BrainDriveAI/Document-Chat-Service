"""
Tests for ChatInteractionUseCase
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
from datetime import datetime, UTC

from app.core.use_cases.chat_interaction import ChatInteractionUseCase
from app.core.domain.entities.chat import ChatSession, ChatMessage
from app.core.domain.exceptions import ChatSessionNotFoundError


def async_gen_mock(items):
    """Helper to create async generator mock"""
    async def _gen():
        for item in items:
            yield item
    return _gen()


@pytest.fixture
def mock_chat_repo():
    """Mock chat repository"""
    return AsyncMock()


@pytest.fixture
def mock_orchestrator():
    """Mock chat orchestrator"""
    return AsyncMock()


@pytest.fixture
def chat_use_case(mock_chat_repo, mock_orchestrator):
    """ChatInteractionUseCase instance with mocks"""
    return ChatInteractionUseCase(
        chat_repo=mock_chat_repo,
        orchestrator=mock_orchestrator
    )


@pytest.fixture
def sample_session():
    """Sample chat session"""
    return ChatSession.create(
        name="Test Session",
        collection_id="coll_123"
    )


@pytest.fixture
def sample_message():
    """Sample chat message"""
    return ChatMessage.create(
        session_id="session_123",
        user_message="What is machine learning?",
        assistant_response="Machine learning is...",
        retrieved_chunks=[],
        response_time_ms=500,
        collection_id="coll_123"
    )


class TestCreateSession:
    """Tests for create_session method"""

    async def test_create_session_success(self, chat_use_case, mock_chat_repo, sample_session):
        """Should create new session successfully"""
        mock_chat_repo.save_session.return_value = sample_session

        result = await chat_use_case.create_session(
            name="Test Session",
            collection_id="coll_123"
        )

        assert result.name == "Test Session"
        assert result.collection_id == "coll_123"
        mock_chat_repo.save_session.assert_called_once()

    async def test_create_session_without_collection(self, chat_use_case, mock_chat_repo):
        """Should create session without collection_id"""
        session = ChatSession.create(name="Test", collection_id=None)
        mock_chat_repo.save_session.return_value = session

        result = await chat_use_case.create_session(name="Test")

        assert result.collection_id is None
        mock_chat_repo.save_session.assert_called_once()


class TestGetSession:
    """Tests for get_session method"""

    async def test_get_session_success(self, chat_use_case, mock_chat_repo, sample_session):
        """Should retrieve existing session"""
        mock_chat_repo.find_session.return_value = sample_session

        result = await chat_use_case.get_session("session_123")

        assert result == sample_session
        mock_chat_repo.find_session.assert_called_once_with("session_123")

    async def test_get_session_not_found(self, chat_use_case, mock_chat_repo):
        """Should raise error when session not found"""
        mock_chat_repo.find_session.return_value = None

        with pytest.raises(ChatSessionNotFoundError):
            await chat_use_case.get_session("nonexistent")


class TestListSessions:
    """Tests for list_sessions method"""

    async def test_list_sessions_success(self, chat_use_case, mock_chat_repo, sample_session):
        """Should list all sessions"""
        sessions = [sample_session, ChatSession.create(name="Session 2")]
        mock_chat_repo.find_all_sessions.return_value = sessions

        result = await chat_use_case.list_sessions()

        assert len(result) == 2
        mock_chat_repo.find_all_sessions.assert_called_once()

    async def test_list_sessions_empty(self, chat_use_case, mock_chat_repo):
        """Should return empty list when no sessions"""
        mock_chat_repo.find_all_sessions.return_value = []

        result = await chat_use_case.list_sessions()

        assert result == []


class TestDeleteSession:
    """Tests for delete_session method"""

    async def test_delete_session_success(self, chat_use_case, mock_chat_repo, sample_session):
        """Should delete existing session"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.delete_session.return_value = True

        result = await chat_use_case.delete_session("session_123")

        assert result is True
        mock_chat_repo.delete_session.assert_called_once_with("session_123")

    async def test_delete_session_not_found(self, chat_use_case, mock_chat_repo):
        """Should raise error when deleting nonexistent session"""
        mock_chat_repo.find_session.return_value = None

        with pytest.raises(ChatSessionNotFoundError):
            await chat_use_case.delete_session("nonexistent")

        mock_chat_repo.delete_session.assert_not_called()


class TestGetChatHistory:
    """Tests for get_chat_history method"""

    async def test_get_chat_history_success(self, chat_use_case, mock_chat_repo, sample_session, sample_message):
        """Should retrieve chat history"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = [sample_message]

        result = await chat_use_case.get_chat_history("session_123", limit=50)

        assert len(result) == 1
        assert result[0] == sample_message
        mock_chat_repo.find_messages.assert_called_once_with("session_123", 50)

    async def test_get_chat_history_custom_limit(self, chat_use_case, mock_chat_repo, sample_session):
        """Should respect custom limit"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []

        await chat_use_case.get_chat_history("session_123", limit=10)

        mock_chat_repo.find_messages.assert_called_once_with("session_123", 10)

    async def test_get_chat_history_session_not_found(self, chat_use_case, mock_chat_repo):
        """Should raise error when session not found"""
        mock_chat_repo.find_session.return_value = None

        with pytest.raises(ChatSessionNotFoundError):
            await chat_use_case.get_chat_history("nonexistent")


class TestProcessMessage:
    """Tests for process_message method"""

    async def test_process_message_success(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session, sample_message):
        """Should process message and save response"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []
        mock_orchestrator.process_query.return_value = sample_message
        mock_chat_repo.save_message.return_value = sample_message
        mock_chat_repo.save_session.return_value = sample_session

        result = await chat_use_case.process_message(
            session_id="session_123",
            user_message="What is ML?"
        )

        assert result == sample_message
        mock_orchestrator.process_query.assert_called_once()
        mock_chat_repo.save_message.assert_called_once()
        mock_chat_repo.save_session.assert_called_once()

    async def test_process_message_includes_chat_history(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session, sample_message):
        """Should include chat history in orchestrator call"""
        mock_chat_repo.find_session.return_value = sample_session
        previous_messages = [sample_message]
        mock_chat_repo.find_messages.return_value = previous_messages
        mock_orchestrator.process_query.return_value = sample_message
        mock_chat_repo.save_message.return_value = sample_message
        mock_chat_repo.save_session.return_value = sample_session

        await chat_use_case.process_message(
            session_id="session_123",
            user_message="Tell me more"
        )

        # Verify orchestrator called with history
        call_args = mock_orchestrator.process_query.call_args
        assert call_args.kwargs["chat_history"] == previous_messages
        assert call_args.kwargs["collection_id"] == sample_session.collection_id

    async def test_process_message_updates_session_stats(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session, sample_message):
        """Should update session message count and timestamp"""
        initial_count = sample_session.message_count
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []
        mock_orchestrator.process_query.return_value = sample_message
        mock_chat_repo.save_message.return_value = sample_message
        mock_chat_repo.save_session.return_value = sample_session

        await chat_use_case.process_message(
            session_id="session_123",
            user_message="Test"
        )

        # Verify session was updated
        saved_session_call = mock_chat_repo.save_session.call_args[0][0]
        assert saved_session_call.message_count == initial_count + 1
        assert saved_session_call.updated_at is not None

    async def test_process_message_adds_response_time(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session, sample_message):
        """Should calculate and add response time"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []

        # Create message with placeholder response time
        message_no_time = ChatMessage.create(
            session_id="session_123",
            user_message="Test",
            assistant_response="Response",
            retrieved_chunks=[],
            response_time_ms=0,  # Will be overwritten
            collection_id="coll_123"
        )
        mock_orchestrator.process_query.return_value = message_no_time
        mock_chat_repo.save_message.return_value = message_no_time
        mock_chat_repo.save_session.return_value = sample_session

        await chat_use_case.process_message(
            session_id="session_123",
            user_message="Test"
        )

        # Verify response time was set to non-zero value
        assert message_no_time.response_time_ms is not None
        assert message_no_time.response_time_ms >= 0


class TestProcessStreamingMessage:
    """Tests for process_streaming_message method"""

    async def test_process_streaming_message_success(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session):
        """Should process streaming message and yield chunks"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []

        # Mock streaming response using helper
        stream_data = [
            {"response": "Hello ", "retrieved_chunks": []},
            {"response": "world"}
        ]
        mock_orchestrator.process_streaming_query = Mock(return_value=async_gen_mock(stream_data))

        saved_message = ChatMessage.create(
            session_id="session_123",
            user_message="Test",
            assistant_response="Hello world",
            retrieved_chunks=[],
            response_time_ms=100,
            collection_id="coll_123"
        )
        mock_chat_repo.save_message.return_value = saved_message
        mock_chat_repo.save_session.return_value = sample_session

        # Collect all chunks
        chunks = []
        async for chunk in chat_use_case.process_streaming_message(
            session_id="session_123",
            user_message="Test"
        ):
            chunks.append(chunk)

        # Verify chunks
        assert len(chunks) == 3  # 2 response chunks + 1 complete chunk
        assert chunks[0]["response"] == "Hello "
        assert chunks[1]["response"] == "world"
        assert chunks[2]["complete"] is True
        assert "message_id" in chunks[2]

    async def test_process_streaming_message_saves_complete_message(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session):
        """Should save complete message after streaming"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []

        stream_data = [
            {"response": "Part 1", "retrieved_chunks": ["chunk1"]},
            {"response": " Part 2"}
        ]
        mock_orchestrator.process_streaming_query = Mock(return_value=async_gen_mock(stream_data))

        saved_message = ChatMessage.create(
            session_id="session_123",
            user_message="Test",
            assistant_response="Part 1 Part 2",
            retrieved_chunks=["chunk1"],
            response_time_ms=150,
            collection_id="coll_123"
        )
        mock_chat_repo.save_message.return_value = saved_message
        mock_chat_repo.save_session.return_value = sample_session

        # Consume all chunks
        async for _ in chat_use_case.process_streaming_message(
            session_id="session_123",
            user_message="Test"
        ):
            pass

        # Verify message was saved with complete response
        mock_chat_repo.save_message.assert_called_once()
        saved_call = mock_chat_repo.save_message.call_args[0][0]
        assert saved_call.assistant_response == "Part 1 Part 2"

    async def test_process_streaming_message_updates_session(self, chat_use_case, mock_chat_repo, mock_orchestrator, sample_session):
        """Should update session stats after streaming"""
        initial_count = sample_session.message_count
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = []

        stream_data = [{"response": "Test"}]
        mock_orchestrator.process_streaming_query = Mock(return_value=async_gen_mock(stream_data))

        saved_message = ChatMessage.create(
            session_id="session_123",
            user_message="Test",
            assistant_response="Test",
            retrieved_chunks=[],
            response_time_ms=100,
            collection_id="coll_123"
        )
        mock_chat_repo.save_message.return_value = saved_message
        mock_chat_repo.save_session.return_value = sample_session

        # Consume all chunks
        async for _ in chat_use_case.process_streaming_message(
            session_id="session_123",
            user_message="Test"
        ):
            pass

        # Verify session updated
        saved_session_call = mock_chat_repo.save_session.call_args[0][0]
        assert saved_session_call.message_count == initial_count + 1


class TestListChatMessages:
    """Tests for list_chat_messages method"""

    async def test_list_chat_messages_success(self, chat_use_case, mock_chat_repo, sample_session, sample_message):
        """Should list all messages for session"""
        mock_chat_repo.find_session.return_value = sample_session
        mock_chat_repo.find_messages.return_value = [sample_message]

        result = await chat_use_case.list_chat_messages("session_123")

        assert len(result) == 1
        assert result[0] == sample_message

    async def test_list_chat_messages_session_not_found(self, chat_use_case, mock_chat_repo):
        """Should raise error when session not found"""
        mock_chat_repo.find_session.return_value = None

        with pytest.raises(ChatSessionNotFoundError):
            await chat_use_case.list_chat_messages("nonexistent")
