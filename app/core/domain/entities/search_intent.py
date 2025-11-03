from enum import Enum


class IntentKind(Enum):
    """Types of user intents"""
    RETRIEVAL = "retrieval"  # Search for specific information
    SUMMARY = "summary"  # Summarize document(s) or collection
    CLARIFICATION = "clarification"  # Follow-up question about previous answer
    CHAT = "chat"  # General conversation, no retrieval needed
    COMPARISON = "comparison"  # Compare multiple concepts
    LISTING = "listing"  # List items/features/etc
    COLLECTION_SUMMARY = "collection_summary"  # Summary of entire collection


class Intent:
    """Represents the classified intent of a user query"""
    
    def __init__(
        self,
        intent_kind: IntentKind,
        requires_retrieval: bool,
        requires_collection_scan: bool = False,
        confidence: float = 1.0,
        reasoning: str = ""
    ):
        self.kind = intent_kind
        self.requires_retrieval = requires_retrieval
        self.requires_collection_scan = requires_collection_scan
        self.confidence = confidence
        self.reasoning = reasoning
