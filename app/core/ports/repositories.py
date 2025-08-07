from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.entities.collection import Collection, CollectionWithDocuments
from ..domain.entities.document import Document
from ..domain.entities.chat import ChatSession, ChatMessage


class CollectionRepository(ABC):
    """Repository for collection persistence"""

    @abstractmethod
    async def save(self, collection: Collection) -> Collection:
        """Save a collection"""
        pass

    @abstractmethod
    async def find_by_id(self, collection_id: str) -> Optional[Collection]:
        """Find collection by ID"""
        pass

    @abstractmethod
    async def find_by_id_with_documents(self, collection_id: str) -> Optional[CollectionWithDocuments]:
        """Find collection by ID along with all its associated documents"""
        pass

    @abstractmethod
    async def find_all(self) -> List[Collection]:
        """Get all collections"""
        pass

    @abstractmethod
    async def delete(self, collection_id: str) -> bool:
        """Delete a collection"""
        pass


class DocumentRepository(ABC):
    """Repository for document persistence"""

    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save a document"""
        pass

    @abstractmethod
    async def find_by_id(self, document_id: str) -> Optional[Document]:
        """Find document by ID"""
        pass

    @abstractmethod
    async def find_by_collection_id(self, collection_id: str) -> List[Document]:
        """Find all documents in a collection"""
        pass

    @abstractmethod
    async def find_all(self) -> List[Document]:
        """Get all documents"""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document"""
        pass


class ChatRepository(ABC):
    """Repository for chat persistence"""

    @abstractmethod
    async def save_session(self, session: ChatSession) -> ChatSession:
        """Save a chat session"""
        pass

    @abstractmethod
    async def save_message(self, message: ChatMessage) -> ChatMessage:
        """Save a chat message"""
        pass

    @abstractmethod
    async def find_session(self, session_id: str) -> Optional[ChatSession]:
        """Find chat session by ID"""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete chat session by ID"""
        pass

    @abstractmethod
    async def find_messages(
            self,
            session_id: str,
            limit: int = 50
    ) -> List[ChatMessage]:
        """Find messages for a session"""
        pass

    @abstractmethod
    async def find_all_sessions(self) -> List[ChatSession]:
        """Get all chat sessions"""
        pass
