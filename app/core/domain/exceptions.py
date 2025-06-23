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
