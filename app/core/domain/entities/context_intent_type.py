from enum import Enum


class IntentType(Enum):
    """Types of user intents"""
    RETRIEVAL = "retrieval"  # Search for specific information
    SUMMARY = "summary"  # Summarize document(s) or collection
    CLARIFICATION = "clarification"  # Follow-up question about previous answer
    CHAT = "chat"  # General conversation, no retrieval needed
    COMPARISON = "comparison"  # Compare multiple concepts
    LISTING = "listing"  # List items/features/etc
    COLLECTION_SUMMARY = "collection_summary"  # Summary of entire collection
