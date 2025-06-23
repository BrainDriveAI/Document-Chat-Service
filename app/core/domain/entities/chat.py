from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Optional
import uuid


@dataclass
class ChatMessage:
    """Domain entity representing a chat message"""
    id: str
    session_id: str
    collection_id: Optional[str]
    user_message: str
    assistant_response: str
    retrieved_chunks: List[str]  # List of chunk IDs used for response
    created_at: datetime
    response_time_ms: int
    
    @classmethod
    def create(
        cls,
        session_id: str,
        user_message: str,
        assistant_response: str,
        retrieved_chunks: List[str],
        response_time_ms: int,
        collection_id: Optional[str] = None
    ) -> "ChatMessage":
        """Factory method to create a new chat message"""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            collection_id=collection_id,
            user_message=user_message,
            assistant_response=assistant_response,
            retrieved_chunks=retrieved_chunks,
            created_at=datetime.now(UTC),
            response_time_ms=response_time_ms
        )


@dataclass
class ChatSession:
    """Domain entity representing a chat session"""
    id: str
    name: str
    collection_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    
    @classmethod
    def create(cls, name: str, collection_id: Optional[str] = None) -> "ChatSession":
        """Factory method to create a new chat session"""
        now = datetime.now(UTC)
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            collection_id=collection_id,
            created_at=now,
            updated_at=now
        )
