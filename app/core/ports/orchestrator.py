from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator
from ..domain.entities.chat import ChatMessage


class ChatOrchestrator(ABC):
    """Port for orchestrating the chat with documents workflow"""

    @abstractmethod
    async def process_query(
            self,
            user_query: str,
            session_id: str,
            collection_id: Optional[str] = None,
            chat_history: Optional[List[ChatMessage]] = None
    ) -> ChatMessage:
        """
        Process a user query and return a chat message with response.

        Args:
            user_query: User's question
            session_id: Chat session ID
            collection_id: Optional collection to search within
            chat_history: Previous messages for context

        Returns:
            ChatMessage with the response and metadata
        """
        pass

    @abstractmethod
    async def process_streaming_query(
            self,
            user_query: str,
            session_id: str,
            collection_id: Optional[str] = None,
            chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user query with streaming response.

        Yields:
            Dictionary with response chunks and metadata
        """
        pass
