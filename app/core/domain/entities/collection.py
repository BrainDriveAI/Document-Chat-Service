import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from .document import Document


@dataclass
class Collection:
    """Domain entity representing a document collection (e.g., Health, Finance, Work)"""
    id: str
    name: str
    description: str
    color: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0

    @classmethod
    def create(cls, name: str, description: str, color: str = "#3B82F6") -> "Collection":
        """Factory method to create a new collection"""
        now = datetime.now(UTC)
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            color=color,
            created_at=now,
            updated_at=now
        )

    def increment_document_count(self) -> None:
        """Increment the document count by 1"""
        self.document_count += 1
        self.updated_at = datetime.now(UTC)

    def decrement_document_count(self) -> None:
        """Decrement the document count by 1, ensuring it doesn't go below zero"""
        if self.document_count > 0:
            self.document_count -= 1
            self.updated_at = datetime.now(UTC)

    def update_document_count(self, increment: int) -> None:
        """Update document count by the specified increment (can be positive or negative)"""
        new_count = self.document_count + increment
        if new_count < 0:
            raise ValueError("Document count cannot be negative")
        self.document_count = new_count
        self.updated_at = datetime.now(UTC)


# Data class for combined collection with documents
@dataclass
class CollectionWithDocuments:
    """Data class representing a collection with its associated documents"""

    def __init__(self, collection: Collection, documents: list[Document]):
        self.collection = collection
        self.documents = documents

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary for easy serialization"""
        return {
            "collection": {
                "id": self.collection.id,
                "name": self.collection.name,
                "description": self.collection.description,
                "color": self.collection.color,
                "created_at": self.collection.created_at.isoformat(),
                "updated_at": self.collection.updated_at.isoformat(),
                "document_count": self.collection.document_count
            },
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "original_filename": doc.original_filename,
                    "file_path": doc.file_path,
                    "file_size": doc.file_size,
                    "document_type": doc.document_type.value,
                    "collection_id": doc.collection_id,
                    "status": doc.status.value,
                    "created_at": doc.created_at.isoformat(),
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "metadata": doc.metadata,
                    "chunk_count": doc.chunk_count
                }
                for doc in self.documents
            ]
        }

