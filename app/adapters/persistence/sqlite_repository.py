import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, ForeignKey, select, desc, delete
from ...core.ports.repositories import DocumentRepository, CollectionRepository, ChatRepository
from ...core.domain.entities.document import Document, DocumentStatus, DocumentType
from ...core.domain.entities.collection import Collection as DomainCollection, CollectionWithDocuments
from ...core.domain.entities.chat import ChatSession as DomainChatSession, ChatMessage as DomainChatMessage

Base = declarative_base()


# ORM models
class CollectionModel(Base):
    __tablename__ = "collections"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    color = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    document_count = Column(Integer, nullable=False, default=0)


class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    document_type = Column(String, nullable=False)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    # Rename attribute to avoid conflict; column name stays "metadata"
    metadata_json = Column('metadata', JSON, nullable=True)
    chunk_count = Column(Integer, nullable=False, default=0)


class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    message_count = Column(Integer, nullable=False, default=0)


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=True)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, nullable=False)  # store list of chunk IDs as JSON
    created_at = Column(DateTime, nullable=False)
    response_time_ms = Column(Integer, nullable=False)



# Repository implementations
class SQLiteRepositoryMixin:
    """
    Mixin to create engine, sessionmaker, and initialize tables.
    """

    def __init__(self, database_url: str):
        """
        Args:
            database_url: SQLAlchemy database URL, e.g. "sqlite+aiosqlite:///./data/app.db"
        """
        # echo can be controlled by settings or environment
        self._engine = create_async_engine(database_url, echo=False, future=True)
        self._async_session = sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            future=True
        )

    async def init_models(self):
        """
        Create tables if they do not exist.
        Call this once on startup.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


class SQLiteCollectionRepository(CollectionRepository, SQLiteRepositoryMixin):
    """Repository for Collection persistence using SQLite via SQLAlchemy Async"""

    def __init__(self, database_url: str):
        SQLiteRepositoryMixin.__init__(self, database_url)

    async def save(self, collection: DomainCollection) -> DomainCollection:
        """
        Insert or update a collection.
        """
        async with self._async_session() as db_session:
            async with db_session.begin():
                existing = await db_session.get(CollectionModel, collection.id)
                if existing:
                    existing.name = collection.name
                    existing.description = collection.description
                    existing.color = collection.color
                    existing.updated_at = collection.updated_at
                    existing.document_count = collection.document_count
                else:
                    model = CollectionModel(
                        id=collection.id,
                        name=collection.name,
                        description=collection.description,
                        color=collection.color,
                        created_at=collection.created_at,
                        updated_at=collection.updated_at,
                        document_count=collection.document_count
                    )
                    db_session.add(model)
        return collection

    async def find_by_id(self, collection_id: str) -> Optional[DomainCollection]:
        async with self._async_session() as db_session:
            result = await db_session.get(CollectionModel, collection_id)
            if not result:
                return None
            domain = DomainCollection(
                id=result.id,
                name=result.name,
                description=result.description,
                color=result.color,
                created_at=result.created_at,
                updated_at=result.updated_at,
                document_count=result.document_count
            )
            return domain

    async def find_by_id_with_documents(self, collection_id: str) -> Optional[CollectionWithDocuments]:
        """
        Fetch a collection by ID along with all its associated documents.

        Args:
            collection_id: The ID of the collection to fetch

        Returns:
            CollectionWithDocuments object containing the collection and its documents,
            or None if the collection doesn't exist
        """
        async with self._async_session() as db_session:
            # Fetch collection
            collection_result = await db_session.get(CollectionModel, collection_id)
            if not collection_result:
                return None

            # Convert collection to domain object
            collection = DomainCollection(
                id=collection_result.id,
                name=collection_result.name,
                description=collection_result.description,
                color=collection_result.color,
                created_at=collection_result.created_at,
                updated_at=collection_result.updated_at,
                document_count=collection_result.document_count
            )

            # Fetch associated documents
            documents_result = await db_session.execute(
                select(DocumentModel).where(DocumentModel.collection_id == collection_id)
            )
            document_records = documents_result.scalars().all()

            # Convert documents to domain objects
            documents: List[Document] = []
            for rec in document_records:
                try:
                    doc_type = DocumentType(rec.document_type)
                except ValueError:
                    doc_type = DocumentType.PDF

                try:
                    status = DocumentStatus(rec.status)
                except ValueError:
                    status = DocumentStatus.UPLOADED

                documents.append(Document(
                    id=rec.id,
                    filename=rec.filename,
                    original_filename=rec.original_filename,
                    file_path=rec.file_path,
                    file_size=rec.file_size,
                    document_type=doc_type,
                    collection_id=rec.collection_id,
                    status=status,
                    created_at=rec.created_at,
                    processed_at=rec.processed_at,
                    metadata=rec.metadata_json or {},
                    chunk_count=rec.chunk_count
                ))

            return CollectionWithDocuments(collection=collection, documents=documents)

    async def find_all(self) -> List[DomainCollection]:
        async with self._async_session() as db_session:
            result = await db_session.execute(select(CollectionModel).order_by(desc(CollectionModel.created_at)))
            records = result.scalars().all()
            collections: List[DomainCollection] = []
            for rec in records:
                collections.append(DomainCollection(
                    id=rec.id,
                    name=rec.name,
                    description=rec.description,
                    color=rec.color,
                    created_at=rec.created_at,
                    updated_at=rec.updated_at,
                    document_count=rec.document_count
                ))
            return collections

    async def delete(self, collection_id: str) -> bool:
        async with self._async_session() as db_session:
            async with db_session.begin():
                obj = await db_session.get(CollectionModel, collection_id)
                if not obj:
                    return False
                await db_session.delete(obj)
            return True


class SQLiteDocumentRepository(DocumentRepository, SQLiteRepositoryMixin):
    """Repository for Document persistence using SQLite via SQLAlchemy Async"""

    def __init__(self, database_url: str):
        SQLiteRepositoryMixin.__init__(self, database_url)

    async def save(self, document: Document) -> Document:
        """
        Insert or update a document record.
        """
        async with self._async_session() as db_session:
            async with db_session.begin():
                existing = await db_session.get(DocumentModel, document.id)
                if existing:
                    existing.filename = document.filename
                    existing.original_filename = document.original_filename
                    existing.file_path = document.file_path
                    existing.file_size = document.file_size
                    existing.document_type = document.document_type.value
                    existing.collection_id = document.collection_id
                    existing.status = document.status.value
                    existing.created_at = document.created_at
                    existing.processed_at = document.processed_at
                    existing.metadata_json = document.metadata or {}
                    existing.chunk_count = document.chunk_count
                else:
                    model = DocumentModel(
                        id=document.id,
                        filename=document.filename,
                        original_filename=document.original_filename,
                        file_path=document.file_path,
                        file_size=document.file_size,
                        document_type=document.document_type.value,
                        collection_id=document.collection_id,
                        status=document.status.value,
                        created_at=document.created_at,
                        processed_at=document.processed_at,
                        metadata_json=document.metadata or {},
                        chunk_count=document.chunk_count
                    )
                    db_session.add(model)
        return document

    async def find_by_id(self, document_id: str) -> Optional[Document]:
        async with self._async_session() as db_session:
            result = await db_session.get(DocumentModel, document_id)
            if not result:
                return None
            try:
                doc_type = DocumentType(result.document_type)
            except ValueError:
                doc_type = DocumentType.PDF
            try:
                status = DocumentStatus(result.status)
            except ValueError:
                status = DocumentStatus.UPLOADED
            domain = Document(
                id=result.id,
                filename=result.filename,
                original_filename=result.original_filename,
                file_path=result.file_path,
                file_size=result.file_size,
                document_type=doc_type,
                collection_id=result.collection_id,
                status=status,
                created_at=result.created_at,
                processed_at=result.processed_at,
                metadata=result.metadata_json or {},
                chunk_count=result.chunk_count
            )
            return domain

    async def find_by_collection_id(self, collection_id: str) -> List[Document]:
        async with self._async_session() as db_session:
            result = await db_session.execute(
                select(DocumentModel).where(DocumentModel.collection_id == collection_id)
            )
            records = result.scalars().all()
            documents: List[Document] = []
            for rec in records:
                try:
                    doc_type = DocumentType(rec.document_type)
                except ValueError:
                    doc_type = DocumentType.PDF
                try:
                    status = DocumentStatus(rec.status)
                except ValueError:
                    status = DocumentStatus.UPLOADED
                documents.append(Document(
                    id=rec.id,
                    filename=rec.filename,
                    original_filename=rec.original_filename,
                    file_path=rec.file_path,
                    file_size=rec.file_size,
                    document_type=doc_type,
                    collection_id=rec.collection_id,
                    status=status,
                    created_at=rec.created_at,
                    processed_at=rec.processed_at,
                    metadata=rec.metadata_json or {},
                    chunk_count=rec.chunk_count
                ))
            return documents

    async def find_all(self) -> List[Document]:
        async with self._async_session() as db_session:
            result = await db_session.execute(select(DocumentModel))
            records = result.scalars().all()
            all_documents: List[Document] = []
            for rec in records:
                try:
                    doc_type = DocumentType(rec.document_type)
                except ValueError:
                    doc_type = DocumentType.PDF
                try:
                    status = DocumentStatus(rec.status)
                except ValueError:
                    status = DocumentStatus.UPLOADED
                all_documents.append(Document(
                    id=rec.id,
                    filename=rec.filename,
                    original_filename=rec.original_filename,
                    file_path=rec.file_path,
                    file_size=rec.file_size,
                    document_type=doc_type,
                    collection_id=rec.collection_id,
                    status=status,
                    created_at=rec.created_at,
                    processed_at=rec.processed_at,
                    metadata=rec.metadata_json or {},
                    chunk_count=rec.chunk_count
                ))
            return all_documents

    async def delete(self, document_id: str) -> bool:
        async with self._async_session() as db_session:
            async with db_session.begin():
                obj = await db_session.get(DocumentModel, document_id)
                if not obj:
                    return False
                await db_session.delete(obj)
            return True


class SQLiteChatRepository(ChatRepository, SQLiteRepositoryMixin):
    """Repository for ChatSession and ChatMessage persistence"""

    def __init__(self, database_url: str):
        SQLiteRepositoryMixin.__init__(self, database_url)

    async def save_session(self, session: DomainChatSession) -> DomainChatSession:
        async with self._async_session() as db_session:
            async with db_session.begin():
                existing = await db_session.get(ChatSessionModel, session.id)
                if existing:
                    existing.name = session.name
                    existing.collection_id = session.collection_id
                    existing.updated_at = session.updated_at
                    existing.message_count = session.message_count
                else:
                    model = ChatSessionModel(
                        id=session.id,
                        name=session.name,
                        collection_id=session.collection_id,
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                        message_count=session.message_count
                    )
                    db_session.add(model)
        return session

    async def find_session(self, session_id: str) -> Optional[DomainChatSession]:
        async with self._async_session() as db_session:
            result = await db_session.get(ChatSessionModel, session_id)
            if not result:
                return None
            domain = DomainChatSession(
                id=result.id,
                name=result.name,
                collection_id=result.collection_id,
                created_at=result.created_at,
                updated_at=result.updated_at,
                message_count=result.message_count
            )
            return domain

    async def find_all_sessions(self) -> List[DomainChatSession]:
        async with self._async_session() as db_session:
            result = await db_session.execute(select(ChatSessionModel).order_by(desc(ChatSessionModel.updated_at)))
            records = result.scalars().all()
            sessions: List[DomainChatSession] = []
            for rec in records:
                sessions.append(DomainChatSession(
                    id=rec.id,
                    name=rec.name,
                    collection_id=rec.collection_id,
                    created_at=rec.created_at,
                    updated_at=rec.updated_at,
                    message_count=rec.message_count
                ))
            return sessions

    async def delete_session(self, session_id: str) -> bool:
        async with self._async_session() as db_session:
            async with db_session.begin():
                # 1. Delete all chat messages associated with the session
                await db_session.execute(
                    delete(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
                )

                # 2. Delete the chat session itself
                session_to_delete = await db_session.get(ChatSessionModel, session_id)
                if not session_to_delete:
                    return False
                await db_session.delete(session_to_delete)
            return True

    async def save_message(self, message: DomainChatMessage) -> DomainChatMessage:
        async with self._async_session() as db_session:
            async with db_session.begin():
                model = ChatMessageModel(
                    id=message.id,
                    session_id=message.session_id,
                    collection_id=message.collection_id,
                    user_message=message.user_message,
                    assistant_response=message.assistant_response,
                    retrieved_chunks=message.retrieved_chunks,
                    created_at=message.created_at,
                    response_time_ms=message.response_time_ms
                )
                db_session.add(model)
        return message

    async def find_messages(self, session_id: str, limit: int = 50) -> List[DomainChatMessage]:
        async with self._async_session() as db_session:
            stmt = select(ChatMessageModel).where(
                ChatMessageModel.session_id == session_id
            ).order_by(ChatMessageModel.created_at.asc()).limit(limit)
            result = await db_session.execute(stmt)
            records = result.scalars().all()
            messages: List[DomainChatMessage] = []
            for rec in records:
                messages.append(DomainChatMessage(
                    id=rec.id,
                    session_id=rec.session_id,
                    collection_id=rec.collection_id,
                    user_message=rec.user_message,
                    assistant_response=rec.assistant_response,
                    retrieved_chunks=rec.retrieved_chunks or [],
                    created_at=rec.created_at,
                    response_time_ms=rec.response_time_ms
                ))
            return messages
