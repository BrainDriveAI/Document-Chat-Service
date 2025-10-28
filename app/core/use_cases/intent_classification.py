import re
import json
from typing import Dict, List, Optional

from ..domain.entities.context_intent_type import IntentType
from ..ports.llm_service import LLMService
from ..domain.exceptions import DomainException


class Intent:
    """Represents the classified intent of a user query"""
    
    def __init__(
        self,
        intent_type: IntentType,
        requires_retrieval: bool,
        requires_collection_scan: bool = False,
        confidence: float = 1.0,
        reasoning: str = ""
    ):
        self.type = intent_type
        self.requires_retrieval = requires_retrieval
        self.requires_collection_scan = requires_collection_scan
        self.confidence = confidence
        self.reasoning = reasoning


class IntentClassificationError(DomainException):
    """Error during intent classification"""
    pass


class IntentClassificationUseCase:
    """Use case for classifying user query intent"""
    
    # Keywords that indicate no retrieval needed
    CHAT_KEYWORDS = [
        r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
        r'^(what|who) (are|is) you',
        r'how are you',
        r'nice to meet',
    ]
    
    # Keywords that indicate collection-level operations
    COLLECTION_KEYWORDS = [
        r'(summary|summarize|overview).*(collection|all documents|entire|whole)',
        r'(collection|all documents).*(summary|overview|key points)',
        r'what (is|are) in (this|the) collection',
        r'give me.*(overview|summary) of (this|the) collection',
    ]
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def classify_intent(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        collection_id: Optional[str] = None
    ) -> Intent:
        """
        Classify the intent of a user query.
        
        Args:
            query: The user's query
            chat_history: Recent conversation history
            collection_id: ID of the collection being queried
            
        Returns:
            Intent object with classification results
        """
        try:
            # Quick heuristic checks first (faster)
            if self._is_chat_intent(query):
                return Intent(
                    intent_type=IntentType.CHAT,
                    requires_retrieval=False,
                    reasoning="Detected as general conversation"
                )
            
            if self._is_collection_summary_intent(query):
                return Intent(
                    intent_type=IntentType.COLLECTION_SUMMARY,
                    requires_retrieval=True,
                    requires_collection_scan=True,
                    reasoning="Detected as collection-level summary request"
                )
            
            # For complex cases, use LLM classification
            return await self._llm_classify_intent(query, chat_history)
            
        except Exception as e:
            # Fallback to safe default: assume retrieval needed
            return Intent(
                intent_type=IntentType.RETRIEVAL,
                requires_retrieval=True,
                confidence=0.5,
                reasoning=f"Classification error, defaulting to retrieval: {str(e)}"
            )
    
    def _is_chat_intent(self, query: str) -> bool:
        """Quick heuristic check for chat intents"""
        query_lower = query.lower().strip()
        
        for pattern in self.CHAT_KEYWORDS:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _is_collection_summary_intent(self, query: str) -> bool:
        """Quick heuristic check for collection summary intents"""
        query_lower = query.lower().strip()
        
        for pattern in self.COLLECTION_KEYWORDS:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    async def _llm_classify_intent(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Intent:
        """Use LLM to classify complex intents"""
        
        # Format chat history
        history_text = self._format_chat_history(chat_history) if chat_history else "No previous conversation."
        
        prompt = f"""Analyze the user's intent based on their query and conversation history.

Conversation History:
{history_text}

Current Query: "{query}"

Available Intent Types:
- RETRIEVAL: User wants specific information from documents (search needed)
- SUMMARY: User wants a summary of specific document(s) 
- CLARIFICATION: User is asking a follow-up question about the previous answer
- COMPARISON: User wants to compare multiple concepts or items
- LISTING: User wants a list of items, features, or options
- CHAT: General conversation, no document search needed

Classify the intent and determine if document retrieval is needed.

Respond in JSON format:
{{
    "intent": "INTENT_TYPE",
    "requires_retrieval": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            intent_type = IntentType(result["intent"].lower())
            
            return Intent(
                intent_type=intent_type,
                requires_retrieval=result["requires_retrieval"],
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            # Fallback
            return Intent(
                intent_type=IntentType.RETRIEVAL,
                requires_retrieval=True,
                confidence=0.5,
                reasoning=f"LLM classification failed: {str(e)}"
            )
    
    def _format_chat_history(self, chat_history: List[Dict], max_turns: int = 3) -> str:
        """Format chat history for the prompt"""
        if not chat_history:
            return "No previous conversation."
        
        # Take last N turns
        recent_history = chat_history[-max_turns * 2:]
        
        formatted = []
        for msg in recent_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        # Try to extract JSON from response
        try:
            # First try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise IntentClassificationError("Could not parse JSON from LLM response")
