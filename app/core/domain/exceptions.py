from typing import Dict, Any, Optional

class DomainException(Exception):
    """Base exception for domain layer"""
    pass


class DocumentNotFoundError(DomainException):
    """Raised when a document is not found"""
    pass


class CollectionNotFoundError(DomainException):
    """Raised when a collection is not found"""
    pass


class DocumentProcessingError(DomainException):
    """Raised when document processing fails"""
    pass

class DocumentDeletionError(DomainException):
    """Raised when document deletion fails"""
    pass

class DocumentDeletionError(DomainException):
    """Base class for document deletion errors"""
    def __init__(self, message: str, deletion_results: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.deletion_results = deletion_results or {}

class VectorStoreDeletionError(DocumentDeletionError):
    """Raised when vector store deletion fails"""
    pass

class FileDeletionError(DocumentDeletionError):
    """Raised when file deletion fails from storage service"""
    pass

class DatabaseDeletionError(DocumentDeletionError):
    """Raised when database deletion fails from repository"""
    pass

class CollectionCountUpdateError(DocumentDeletionError):
    """Raised when collection count update fails"""
    pass

class PartialDocumentDeletionError(DocumentDeletionError):
    """Raised when some but not all deletion operations fail"""
    def __init__(self, message: str, deletion_results: Dict[str, Any], failed_operations: list):
        super().__init__(message, deletion_results)
        self.failed_operations = failed_operations


class EmbeddingGenerationError(DomainException):
    """Raised when embedding generation fails"""
    pass


class InvalidDocumentTypeError(DomainException):
    """Raised when document type is not supported"""
    pass


class ChatSessionNotFoundError(DomainException):
    """Raised when chat session is not found"""
    pass


class ChatSessionProcessingError(DomainException):
    """Raised when chat session processing fails"""
    pass


class TokenizationError(DomainException):
    """Raised when tokenization fails"""
    pass


class CollectionSummaryError(DomainException):
    """Error during collection summary generation"""
    pass
