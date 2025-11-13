"""
Tests for QueryTransformationUseCase
"""
import pytest
from unittest.mock import AsyncMock

from app.core.use_cases.query_transformation import QueryTransformationUseCase
from app.core.domain.entities.query_transformation import QueryTransformationMethod


@pytest.fixture
def mock_llm_service():
    """Mock LLM service"""
    return AsyncMock()


@pytest.fixture
def query_transformation_use_case(mock_llm_service):
    """QueryTransformationUseCase instance with mocks"""
    return QueryTransformationUseCase(llm_service=mock_llm_service)


class TestBasicTransformation:
    """Tests for basic query transformation"""

    async def test_empty_query_returns_empty(self, query_transformation_use_case):
        """Should return original when query is empty"""
        result = await query_transformation_use_case.transform_query("", [])
        assert result == [""]

    async def test_no_methods_returns_original(self, query_transformation_use_case):
        """Should return original when no methods specified"""
        result = await query_transformation_use_case.transform_query("test query", [])
        assert result == ["test query"]

    async def test_returns_list_of_strings(self, query_transformation_use_case, mock_llm_service):
        """Should always return list of strings"""
        mock_llm_service.generate_response.return_value = "Contextualized query"

        result = await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.CONTEXTUALIZE]
        )

        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)


class TestNeedsContextualization:
    """Tests for _needs_contextualization heuristic"""

    def test_short_queries_need_context(self, query_transformation_use_case):
        """Should identify short queries as needing context"""
        assert query_transformation_use_case._needs_contextualization("it")
        assert query_transformation_use_case._needs_contextualization("this one")
        assert query_transformation_use_case._needs_contextualization("tell me")

    def test_pronouns_need_context(self, query_transformation_use_case):
        """Should detect pronoun usage"""
        assert query_transformation_use_case._needs_contextualization("What is it about?")
        assert query_transformation_use_case._needs_contextualization("Tell me about this")
        assert query_transformation_use_case._needs_contextualization("Show me that")
        assert query_transformation_use_case._needs_contextualization("How do they work?")

    def test_vague_references_need_context(self, query_transformation_use_case):
        """Should detect vague references"""
        assert query_transformation_use_case._needs_contextualization("Tell me more")
        assert query_transformation_use_case._needs_contextualization("What about this?")
        assert query_transformation_use_case._needs_contextualization("Explain it")
        assert query_transformation_use_case._needs_contextualization("Examples")

    def test_specific_queries_dont_need_context(self, query_transformation_use_case):
        """Should identify specific queries as not needing context"""
        assert not query_transformation_use_case._needs_contextualization(
            "What are the key features of Python programming language?"
        )
        assert not query_transformation_use_case._needs_contextualization(
            "How does machine learning work in general?"
        )

    def test_case_insensitive_detection(self, query_transformation_use_case):
        """Should be case insensitive"""
        assert query_transformation_use_case._needs_contextualization("TELL ME MORE")
        assert query_transformation_use_case._needs_contextualization("What Is It?")


class TestContextualization:
    """Tests for CONTEXTUALIZE method"""

    async def test_contextualize_with_history(self, query_transformation_use_case, mock_llm_service):
        """Should contextualize query using chat history"""
        mock_llm_service.generate_response.return_value = "What are the features of Python?"

        chat_history = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]

        result = await query_transformation_use_case.transform_query(
            "What about its features?",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history
        )

        assert "Python" in result[0] or "features" in result[0]
        mock_llm_service.generate_response.assert_called_once()

    async def test_contextualize_without_history_returns_original(self, query_transformation_use_case):
        """Should return original when no history provided"""
        result = await query_transformation_use_case.transform_query(
            "What about its features?",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=None
        )

        assert result == ["What about its features?"]

    async def test_contextualize_limits_history(self, query_transformation_use_case, mock_llm_service):
        """Should limit chat history to max_history_turns"""
        mock_llm_service.generate_response.return_value = "Contextualized query"

        # Create 10 messages
        chat_history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        await query_transformation_use_case.transform_query(
            "Tell me more",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history,
            max_history_turns=3
        )

        # Verify prompt contains recent messages only
        call_args = mock_llm_service.generate_response.call_args
        prompt = call_args.kwargs["prompt"]

        # Should include last 6 messages (3 turns * 2 roles)
        assert "Message 9" in prompt or "Message 8" in prompt
        assert "Message 0" not in prompt  # Too old

    async def test_contextualize_cleans_response(self, query_transformation_use_case, mock_llm_service):
        """Should clean up LLM response"""
        mock_llm_service.generate_response.return_value = '  "What are Python features?"  '

        chat_history = [{"role": "user", "content": "Test"}]

        result = await query_transformation_use_case.transform_query(
            "Tell me more",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history
        )

        assert result[0] == "What are Python features?"

    async def test_contextualize_fallback_on_empty(self, query_transformation_use_case, mock_llm_service):
        """Should fallback to original on empty response"""
        mock_llm_service.generate_response.return_value = ""

        chat_history = [{"role": "user", "content": "Test"}]

        result = await query_transformation_use_case.transform_query(
            "Tell me more",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history
        )

        assert result == ["Tell me more"]

    async def test_contextualize_fallback_on_error(self, query_transformation_use_case, mock_llm_service):
        """Should fallback to original on LLM error"""
        mock_llm_service.generate_response.side_effect = Exception("LLM error")

        chat_history = [{"role": "user", "content": "Test"}]

        result = await query_transformation_use_case.transform_query(
            "Tell me more",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history
        )

        assert result == ["Tell me more"]


class TestMultiQuery:
    """Tests for MULTI_QUERY method"""

    async def test_multi_query_generates_variations(self, query_transformation_use_case, mock_llm_service):
        """Should generate multiple query variations"""
        mock_llm_service.generate_response.return_value = """
        1. How does machine learning function?
        2. What is the working mechanism of ML?
        3. Explain machine learning operations
        """

        result = await query_transformation_use_case.transform_query(
            "How does machine learning work?",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        # Should have original + variations
        assert len(result) >= 3
        mock_llm_service.generate_response.assert_called_once()

    async def test_multi_query_parses_numbered_list(self, query_transformation_use_case, mock_llm_service):
        """Should parse numbered list from LLM"""
        mock_llm_service.generate_response.return_value = """
        1. First query
        2. Second query
        3. Third query
        """

        result = await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        assert "First query" in result or any("First" in q for q in result)

    async def test_multi_query_handles_parenthesis_format(self, query_transformation_use_case, mock_llm_service):
        """Should parse list with parenthesis format"""
        mock_llm_service.generate_response.return_value = """
        1) First query
        2) Second query
        3) Third query
        """

        result = await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        assert len(result) >= 3

    async def test_multi_query_filters_short_queries(self, query_transformation_use_case, mock_llm_service):
        """Should filter out very short queries"""
        mock_llm_service.generate_response.return_value = """
        1. Valid longer query here
        2. x
        3. Another valid query
        """

        result = await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        # Should filter out single-word queries
        assert all(len(q.split()) >= 2 for q in result if q != "test")

    async def test_multi_query_fallback_on_error(self, query_transformation_use_case, mock_llm_service):
        """Should return original on error"""
        mock_llm_service.generate_response.side_effect = Exception("LLM error")

        result = await query_transformation_use_case.transform_query(
            "test query",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        assert "test query" in result


class TestHyDE:
    """Tests for HYDE (Hypothetical Document Embedding) method"""

    async def test_hyde_generates_hypothetical_doc(self, query_transformation_use_case, mock_llm_service):
        """Should generate hypothetical document"""
        mock_llm_service.generate_response.return_value = (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data and improve over time."
        )

        result = await query_transformation_use_case.transform_query(
            "What is machine learning?",
            [QueryTransformationMethod.HYDE]
        )

        # Should have original + HyDE
        assert len(result) >= 2
        assert any("Machine learning" in q for q in result)

    async def test_hyde_cleans_response(self, query_transformation_use_case, mock_llm_service):
        """Should clean up whitespace"""
        # Need at least 10 words to pass the validation
        mock_llm_service.generate_response.return_value = (
            "  \n  Hypothetical document text with enough words to pass the minimum length validation requirement  \n  "
        )

        result = await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.HYDE]
        )

        # Should have cleaned text in results
        assert any("Hypothetical document text" in q for q in result)

    async def test_hyde_fallback_on_short_response(self, query_transformation_use_case, mock_llm_service):
        """Should fallback to original on very short response"""
        mock_llm_service.generate_response.return_value = "Short"

        result = await query_transformation_use_case.transform_query(
            "test query",
            [QueryTransformationMethod.HYDE]
        )

        # Should include original query
        assert "test query" in result

    async def test_hyde_fallback_on_error(self, query_transformation_use_case, mock_llm_service):
        """Should fallback to original on error"""
        mock_llm_service.generate_response.side_effect = Exception("LLM error")

        result = await query_transformation_use_case.transform_query(
            "test query",
            [QueryTransformationMethod.HYDE]
        )

        assert "test query" in result


class TestCombinedMethods:
    """Tests for combining multiple transformation methods"""

    async def test_contextualize_then_multi_query(self, query_transformation_use_case, mock_llm_service):
        """Should contextualize then generate variations"""
        # First call: contextualize
        # Second call: multi-query
        mock_llm_service.generate_response.side_effect = [
            "What are Python features?",
            "1. What features does Python have?\n2. Python capabilities?\n3. Features of Python language"
        ]

        chat_history = [{"role": "user", "content": "Tell me about Python"}]

        result = await query_transformation_use_case.transform_query(
            "What about features?",
            [QueryTransformationMethod.CONTEXTUALIZE, QueryTransformationMethod.MULTI_QUERY],
            chat_history=chat_history
        )

        # Should have contextualized + variations
        assert len(result) >= 3
        assert mock_llm_service.generate_response.call_count == 2

    async def test_multi_query_and_hyde(self, query_transformation_use_case, mock_llm_service):
        """Should generate both multi-query and HyDE"""
        mock_llm_service.generate_response.side_effect = [
            "1. Query variation 1\n2. Query variation 2",
            "Hypothetical document about the topic"
        ]

        result = await query_transformation_use_case.transform_query(
            "test query",
            [QueryTransformationMethod.MULTI_QUERY, QueryTransformationMethod.HYDE]
        )

        # Should have original + variations + HyDE
        assert len(result) >= 3

    async def test_all_methods_combined(self, query_transformation_use_case, mock_llm_service):
        """Should apply all transformation methods"""
        mock_llm_service.generate_response.side_effect = [
            "Contextualized query",
            "1. Variation 1\n2. Variation 2",
            "Hypothetical document"
        ]

        chat_history = [{"role": "user", "content": "Previous question"}]

        result = await query_transformation_use_case.transform_query(
            "Tell me more",
            [
                QueryTransformationMethod.CONTEXTUALIZE,
                QueryTransformationMethod.MULTI_QUERY,
                QueryTransformationMethod.HYDE
            ],
            chat_history=chat_history
        )

        # Should have multiple transformed queries
        assert len(result) >= 3


class TestDeduplication:
    """Tests for query deduplication"""

    async def test_removes_duplicate_queries(self, query_transformation_use_case, mock_llm_service):
        """Should deduplicate identical queries"""
        mock_llm_service.generate_response.return_value = (
            "1. test query\n2. test query\n3. different query"
        )

        result = await query_transformation_use_case.transform_query(
            "test query",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        # Should not have exact duplicates
        assert len(result) == len(set(result))


class TestHelperMethods:
    """Tests for helper methods"""

    def test_format_chat_history(self, query_transformation_use_case):
        """Should format chat history correctly"""
        chat_history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"}
        ]

        formatted = query_transformation_use_case._format_chat_history(chat_history)

        assert "User: First message" in formatted
        assert "Assistant: First response" in formatted

    def test_format_chat_history_truncates_long_messages(self, query_transformation_use_case):
        """Should truncate very long messages"""
        long_message = "x" * 300
        chat_history = [{"role": "user", "content": long_message}]

        formatted = query_transformation_use_case._format_chat_history(chat_history)

        assert len(formatted) < 250  # Should be truncated

    def test_parse_numbered_list(self, query_transformation_use_case):
        """Should parse various numbered list formats"""
        text = """
        1. First item
        2. Second item
        3. Third item
        """

        parsed = query_transformation_use_case._parse_numbered_list(text)

        assert len(parsed) == 3
        assert "First item" in parsed
        assert "Second item" in parsed

    def test_parse_numbered_list_with_quotes(self, query_transformation_use_case):
        """Should strip quotes from parsed items"""
        text = '1. "First item"\n2. \'Second item\''

        parsed = query_transformation_use_case._parse_numbered_list(text)

        assert "First item" in parsed
        assert "Second item" in parsed


class TestLLMParameters:
    """Tests for LLM service parameters"""

    async def test_contextualize_uses_low_temperature(self, query_transformation_use_case, mock_llm_service):
        """Should use low temperature for contextualization"""
        mock_llm_service.generate_response.return_value = "Contextualized"

        chat_history = [{"role": "user", "content": "Test"}]

        await query_transformation_use_case.transform_query(
            "Tell me more",
            [QueryTransformationMethod.CONTEXTUALIZE],
            chat_history=chat_history
        )

        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["temperature"] == 0.3

    async def test_multi_query_uses_high_temperature(self, query_transformation_use_case, mock_llm_service):
        """Should use high temperature for diversity"""
        mock_llm_service.generate_response.return_value = "1. Query"

        await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.MULTI_QUERY]
        )

        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["temperature"] == 0.7

    async def test_hyde_uses_medium_temperature(self, query_transformation_use_case, mock_llm_service):
        """Should use medium temperature for HyDE"""
        mock_llm_service.generate_response.return_value = "Hypothetical document text goes here"

        await query_transformation_use_case.transform_query(
            "test",
            [QueryTransformationMethod.HYDE]
        )

        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["temperature"] == 0.5
