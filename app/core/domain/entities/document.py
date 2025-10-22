from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, Optional
from enum import Enum
import uuid


class DocumentStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    MARKDOWN = "md"
    HTML = "html"
    PPTX = "pptx"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Domain entity representing a document"""
    id: str
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    document_type: DocumentType
    collection_id: str
    status: DocumentStatus
    created_at: datetime
    processed_at: Optional[datetime]
    metadata: Dict
    chunk_count: int = 0
    
    @classmethod
    def create(
        cls, 
        filename: str, 
        original_filename: str,
        file_path: str,
        file_size: int,
        document_type: DocumentType,
        collection_id: str,
        metadata: Optional[Dict] = None
    ) -> "Document":
        """Factory method to create a new document"""
        return cls(
            id=str(uuid.uuid4()),
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            document_type=document_type,
            collection_id=collection_id,
            status=DocumentStatus.UPLOADED,
            created_at=datetime.now(UTC),
            processed_at=None,
            metadata=metadata or {},
            chunk_count=0
        )
    
    def mark_processing(self):
        """Mark document as being processed"""
        self.status = DocumentStatus.PROCESSING
    
    def mark_processed(self, chunk_count: int):
        """Mark document as successfully processed"""
        self.status = DocumentStatus.PROCESSED
        self.processed_at = datetime.now(UTC)
        self.chunk_count = chunk_count
    
    def mark_failed(self):
        """Mark document as failed to process"""
        self.status = DocumentStatus.FAILED
