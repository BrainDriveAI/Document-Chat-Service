from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from .document_chunk import DocumentChunk
from .context_intent_type import IntentType


class GenerationType(Enum):
    """Type of generation suggested for the retrieved context"""
    NONE = "none"  # No generation needed (chat intent)
    ANSWER = "answer"  # Generate answer from context
    SUMMARY = "summary"  # Generate summary from context
    COMPARISON = "comparison"  # Generate comparison
    LISTING = "listing"  # Generate list/enumeration


@dataclass
class ContextResult:
    """
    Unified result from context retrieval.
    Always returns the same structure regardless of intent.
    """
    chunks: List[DocumentChunk]
    intent: IntentType
    requires_generation: bool
    generation_type: GenerationType
    metadata: Dict[str, Any]
    
    @classmethod
    def create_chat_result(cls, intent: IntentType) -> "ContextResult":
        """Factory method for chat intents (no context needed)"""
        return cls(
            chunks=[],
            intent=intent,
            requires_generation=False,
            generation_type=GenerationType.NONE,
            metadata={
                "message": "General conversation - no context retrieval needed"
            }
        )
    
    @classmethod
    def create_retrieval_result(
        cls,
        chunks: List[DocumentChunk],
        intent: IntentType,
        generation_type: GenerationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ContextResult":
        """Factory method for retrieval intents"""
        return cls(
            chunks=chunks,
            intent=intent,
            requires_generation=True,
            generation_type=generation_type,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "collection_id": chunk.collection_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "metadata": chunk.metadata
                }
                for chunk in self.chunks
            ],
            "intent": {
                "type": self.intent.type.value,
                "requires_retrieval": self.intent.requires_retrieval,
                "requires_collection_scan": self.intent.requires_collection_scan,
                "confidence": self.intent.confidence,
                "reasoning": self.intent.reasoning
            },
            "requires_generation": self.requires_generation,
            "generation_type": self.generation_type.value,
            "metadata": self.metadata
        }
