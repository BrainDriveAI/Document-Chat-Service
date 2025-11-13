"""
Tests for IntentClassificationUseCase
"""
import pytest
from unittest.mock import AsyncMock
import json

from app.core.use_cases.intent_classification import IntentClassificationUseCase
from app.core.domain.entities.search_intent import IntentKind, Intent


@pytest.fixture
def mock_llm_service():
    """Mock LLM service"""
    return AsyncMock()


@pytest.fixture
def intent_use_case(mock_llm_service):
    """IntentClassificationUseCase instance with mocks"""
    return IntentClassificationUseCase(llm_service=mock_llm_service)


class TestChatIntentDetection:
    """Tests for chat intent detection (heuristic-based)"""

    async def test_detect_greeting_hi(self, intent_use_case):
        """Should detect 'hi' as chat intent"""
        result = await intent_use_case.classify_intent("Hi there!")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False
        assert "conversation" in result.reasoning.lower()

    async def test_detect_greeting_hello(self, intent_use_case):
        """Should detect 'hello' as chat intent"""
        result = await intent_use_case.classify_intent("Hello, how are you?")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_greeting_hey(self, intent_use_case):
        """Should detect 'hey' as chat intent"""
        result = await intent_use_case.classify_intent("hey")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_thanks(self, intent_use_case):
        """Should detect 'thanks' as chat intent"""
        result = await intent_use_case.classify_intent("Thanks for the help!")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_thank_you(self, intent_use_case):
        """Should detect 'thank you' as chat intent"""
        result = await intent_use_case.classify_intent("Thank you very much")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_goodbye(self, intent_use_case):
        """Should detect 'goodbye' as chat intent"""
        result = await intent_use_case.classify_intent("Goodbye!")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_who_are_you(self, intent_use_case):
        """Should detect 'who are you' as chat intent"""
        result = await intent_use_case.classify_intent("What are you?")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_detect_how_are_you(self, intent_use_case):
        """Should detect 'how are you' as chat intent"""
        result = await intent_use_case.classify_intent("How are you doing?")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False

    async def test_case_insensitive_detection(self, intent_use_case):
        """Should detect chat intent regardless of case"""
        result = await intent_use_case.classify_intent("HI THERE!")

        assert result.kind == IntentKind.CHAT
        assert result.requires_retrieval is False


class TestCollectionSummaryDetection:
    """Tests for collection summary intent detection (heuristic-based)"""

    async def test_detect_summarize_collection(self, intent_use_case):
        """Should detect collection summary intent"""
        result = await intent_use_case.classify_intent("Summarize the collection")

        assert result.kind == IntentKind.COLLECTION_SUMMARY
        assert result.requires_retrieval is True
        assert result.requires_collection_scan is True

    async def test_detect_collection_overview(self, intent_use_case):
        """Should detect collection overview request"""
        result = await intent_use_case.classify_intent("Give me an overview of this collection")

        assert result.kind == IntentKind.COLLECTION_SUMMARY
        assert result.requires_retrieval is True
        assert result.requires_collection_scan is True

    async def test_detect_what_in_collection(self, intent_use_case):
        """Should detect 'what is in collection' query"""
        result = await intent_use_case.classify_intent("What is in this collection?")

        assert result.kind == IntentKind.COLLECTION_SUMMARY
        assert result.requires_retrieval is True
        assert result.requires_collection_scan is True

    async def test_detect_all_documents_summary(self, intent_use_case):
        """Should detect summary of all documents"""
        result = await intent_use_case.classify_intent("Summary of all documents")

        assert result.kind == IntentKind.COLLECTION_SUMMARY
        assert result.requires_retrieval is True

    async def test_detect_entire_collection_overview(self, intent_use_case):
        """Should detect entire collection overview"""
        result = await intent_use_case.classify_intent("Give me overview of entire collection")

        assert result.kind == IntentKind.COLLECTION_SUMMARY
        assert result.requires_retrieval is True


class TestLLMClassification:
    """Tests for LLM-based classification (complex queries)"""

    async def test_llm_classification_retrieval(self, intent_use_case, mock_llm_service):
        """Should use LLM for complex retrieval queries"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "User wants specific information"
        })

        result = await intent_use_case.classify_intent("What are the key features of Python?")

        assert result.kind == IntentKind.RETRIEVAL
        assert result.requires_retrieval is True
        assert result.confidence == 0.9
        mock_llm_service.generate_response.assert_called_once()

    async def test_llm_classification_summary(self, intent_use_case, mock_llm_service):
        """Should classify summary intent via LLM"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "SUMMARY",
            "requires_retrieval": True,
            "confidence": 0.85,
            "reasoning": "User wants document summary"
        })

        result = await intent_use_case.classify_intent("Summarize document XYZ")

        assert result.kind == IntentKind.SUMMARY
        assert result.requires_retrieval is True
        assert result.confidence == 0.85

    async def test_llm_classification_clarification(self, intent_use_case, mock_llm_service):
        """Should classify clarification intent via LLM"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "CLARIFICATION",
            "requires_retrieval": True,
            "confidence": 0.8,
            "reasoning": "Follow-up question"
        })

        chat_history = [
            {"role": "user", "content": "What is ML?"},
            {"role": "assistant", "content": "Machine learning is..."}
        ]

        result = await intent_use_case.classify_intent(
            "Can you explain more?",
            chat_history=chat_history
        )

        assert result.kind == IntentKind.CLARIFICATION
        assert result.requires_retrieval is True

    async def test_llm_classification_comparison(self, intent_use_case, mock_llm_service):
        """Should classify comparison intent via LLM"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "COMPARISON",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "User wants to compare concepts"
        })

        result = await intent_use_case.classify_intent("Compare Python and JavaScript")

        assert result.kind == IntentKind.COMPARISON
        assert result.requires_retrieval is True

    async def test_llm_classification_listing(self, intent_use_case, mock_llm_service):
        """Should classify listing intent via LLM"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "LISTING",
            "requires_retrieval": True,
            "confidence": 0.88,
            "reasoning": "User wants a list"
        })

        result = await intent_use_case.classify_intent("List all the features")

        assert result.kind == IntentKind.LISTING
        assert result.requires_retrieval is True

    async def test_llm_classification_with_chat_history(self, intent_use_case, mock_llm_service):
        """Should include chat history in LLM prompt"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "Context-aware retrieval"
        })

        chat_history = [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI is..."}
        ]

        await intent_use_case.classify_intent("What about ML?", chat_history=chat_history)

        # Verify LLM was called with prompt containing history
        call_args = mock_llm_service.generate_response.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Tell me about AI" in prompt
        assert "AI is..." in prompt

    async def test_llm_classification_limits_history(self, intent_use_case, mock_llm_service):
        """Should limit chat history to recent messages"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "Test"
        })

        # Create long chat history
        chat_history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]

        await intent_use_case.classify_intent("Test query", chat_history=chat_history)

        # Verify only recent messages included
        call_args = mock_llm_service.generate_response.call_args
        prompt = call_args.kwargs["prompt"]
        # Should only include last 3 turns (6 messages)
        assert "Message 19" in prompt
        assert "Message 14" in prompt
        assert "Message 13" not in prompt  # Too old


class TestJSONParsing:
    """Tests for JSON response parsing"""

    async def test_parse_clean_json(self, intent_use_case, mock_llm_service):
        """Should parse clean JSON response"""
        mock_llm_service.generate_response.return_value = '{"intent": "RETRIEVAL", "requires_retrieval": true, "confidence": 0.9, "reasoning": "Test"}'

        result = await intent_use_case.classify_intent("Test query")

        assert result.kind == IntentKind.RETRIEVAL

    async def test_parse_json_with_markdown(self, intent_use_case, mock_llm_service):
        """Should extract JSON from markdown code blocks"""
        mock_llm_service.generate_response.return_value = """
        Here's the classification:
        ```json
        {"intent": "RETRIEVAL", "requires_retrieval": true, "confidence": 0.9, "reasoning": "Test"}
        ```
        """

        result = await intent_use_case.classify_intent("Test query")

        assert result.kind == IntentKind.RETRIEVAL

    async def test_parse_json_with_text(self, intent_use_case, mock_llm_service):
        """Should extract JSON from mixed text"""
        mock_llm_service.generate_response.return_value = 'The intent is: {"intent": "RETRIEVAL", "requires_retrieval": true, "confidence": 0.9, "reasoning": "Test"} - hope this helps!'

        result = await intent_use_case.classify_intent("Test query")

        assert result.kind == IntentKind.RETRIEVAL


class TestErrorHandling:
    """Tests for error handling and fallback behavior"""

    async def test_fallback_on_llm_error(self, intent_use_case, mock_llm_service):
        """Should fallback to retrieval on LLM error"""
        mock_llm_service.generate_response.side_effect = Exception("LLM service error")

        # Query that requires LLM classification
        result = await intent_use_case.classify_intent("Complex query about quantum physics")

        # Should fallback to safe default (retrieval)
        assert result.kind == IntentKind.RETRIEVAL
        assert result.requires_retrieval is True
        assert result.confidence == 0.5
        assert "error" in result.reasoning.lower()

    async def test_fallback_on_json_parse_error(self, intent_use_case, mock_llm_service):
        """Should fallback on JSON parsing error"""
        mock_llm_service.generate_response.return_value = "This is not JSON at all!"

        result = await intent_use_case.classify_intent("Complex query")

        # Should fallback to retrieval
        assert result.kind == IntentKind.RETRIEVAL
        assert result.requires_retrieval is True
        assert result.confidence == 0.5

    async def test_fallback_on_invalid_intent_kind(self, intent_use_case, mock_llm_service):
        """Should fallback on invalid intent kind"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "INVALID_INTENT",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "Test"
        })

        result = await intent_use_case.classify_intent("Test query")

        # Should fallback to retrieval
        assert result.kind == IntentKind.RETRIEVAL
        assert result.requires_retrieval is True
        assert result.confidence == 0.5

    async def test_no_confidence_defaults_to_high(self, intent_use_case, mock_llm_service):
        """Should default to 0.8 confidence when not provided"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "reasoning": "Test"
        })

        result = await intent_use_case.classify_intent("Test query")

        assert result.confidence == 0.8


class TestLLMParameters:
    """Tests for LLM service parameters"""

    async def test_llm_called_with_correct_params(self, intent_use_case, mock_llm_service):
        """Should call LLM with appropriate parameters"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "Test"
        })

        await intent_use_case.classify_intent("Complex query")

        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["max_tokens"] == 200
        assert call_args.kwargs["temperature"] == 0.1  # Low temp for classification

    async def test_llm_prompt_structure(self, intent_use_case, mock_llm_service):
        """Should create well-structured prompt"""
        mock_llm_service.generate_response.return_value = json.dumps({
            "intent": "RETRIEVAL",
            "requires_retrieval": True,
            "confidence": 0.9,
            "reasoning": "Test"
        })

        await intent_use_case.classify_intent("Test query")

        call_args = mock_llm_service.generate_response.call_args
        prompt = call_args.kwargs["prompt"]

        # Verify prompt contains key sections
        assert "Current Query:" in prompt
        assert "Available Intent Types:" in prompt
        assert "RETRIEVAL" in prompt
        assert "CHAT" in prompt
        assert "JSON format" in prompt
