import time
from datetime import datetime
from typing import List, Optional, AsyncGenerator, Dict, Any
from ..domain.entities.chat import ChatSession, ChatMessage
from ..domain.exceptions import ChatSessionNotFoundError
from ..ports.orchestrator import ChatOrchestrator
from ..ports.repositories import ChatRepository


class ChatInteractionUseCase:
    """Use case for chat interactions with documents"""

    def __init__(
            self,
            chat_repo: ChatRepository,
            orchestrator: ChatOrchestrator
    ):
        self.chat_repo = chat_repo
        self.orchestrator = orchestrator

    async def create_session(
            self,
            name: str,
            collection_id: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession.create(name=name, collection_id=collection_id)
        return await self.chat_repo.save_session(session)

    async def get_session(self, session_id: str) -> ChatSession:
        """Get a chat session by ID"""
        session = await self.chat_repo.find_session(session_id)
        if not session:
            raise ChatSessionNotFoundError(f"Session {session_id} not found")
        return session

    async def list_sessions(self) -> List[ChatSession]:
        """List all chat sessions"""
        return await self.chat_repo.find_all_sessions()

    async def get_chat_history(
            self,
            session_id: str,
            limit: int = 50
    ) -> List[ChatMessage]:
        """Get chat history for a session"""
        await self.get_session(session_id)  # Validate session exists
        return await self.chat_repo.find_messages(session_id, limit)

    async def process_message(
            self,
            session_id: str,
            user_message: str
    ) -> ChatMessage:
        """Process a user message and generate response"""
        session = await self.get_session(session_id)

        # Get recent chat history for context
        chat_history = await self.get_chat_history(session_id, limit=10)

        # Process the query using orchestrator
        start_time = time.time()
        response = await self.orchestrator.process_query(
            user_query=user_message,
            session_id=session_id,
            collection_id=session.collection_id,
            chat_history=chat_history
        )
        response_time = int((time.time() - start_time) * 1000)

        # Update response time
        response.response_time_ms = response_time

        # Save the message
        saved_message = await self.chat_repo.save_message(response)

        # Update session stats
        session.message_count += 1
        session.updated_at = datetime.utcnow()
        await self.chat_repo.save_session(session)

        return saved_message

    async def process_streaming_message(
            self,
            session_id: str,
            user_message: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user message with streaming response"""
        session = await self.get_session(session_id)

        # Get recent chat history for context
        chat_history = await self.get_chat_history(session_id, limit=10)

        # Track the complete response and metadata for saving later
        complete_response = ""
        retrieved_chunks = []
        start_time = time.time()

        # Process the query using orchestrator with streaming
        async for chunk in self.orchestrator.process_streaming_query(
                user_query=user_message,
                session_id=session_id,
                collection_id=session.collection_id,
                chat_history=chat_history
        ):
            # Accumulate the complete response
            if "response" in chunk:
                complete_response += chunk["response"]

            # Capture retrieved chunks from the first chunk
            if "retrieved_chunks" in chunk:
                retrieved_chunks = chunk["retrieved_chunks"]

            # Yield the chunk to the client
            yield chunk

        # After streaming is complete, save the full message and update session stats
        response_time_ms = int((time.time() - start_time) * 1000)

        # Create the complete ChatMessage
        chat_message = ChatMessage.create(
            session_id=session_id,
            user_message=user_message,
            assistant_response=complete_response,
            retrieved_chunks=retrieved_chunks,
            response_time_ms=response_time_ms,
            collection_id=session.collection_id
        )

        # Save the complete message
        await self.chat_repo.save_message(chat_message)

        # Update session stats
        session.message_count += 1
        session.updated_at = datetime.utcnow()
        await self.chat_repo.save_session(session)

    async def list_chat_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all chat messages for a session"""
        session = await self.get_session(session_id)

        return await self.chat_repo.find_messages(session.id)
