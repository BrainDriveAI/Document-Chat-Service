# app/core/use_cases/query_transformation.py

import re
import logging
from typing import List, Optional, Dict, Set
from ..ports.llm_service import LLMService
from ..domain.entities.query_transformation import QueryTransformationMethod
from ..domain.exceptions import DomainException

logger = logging.getLogger(__name__)


class QueryTransformationError(DomainException):
    """Error during query transformation"""
    pass


class QueryTransformationUseCase:
    """Use case for transforming search queries"""
    
    # Indicators that a query needs contextualization
    CONTEXT_INDICATORS = [
        r'\b(it|this|that|these|those)\b',
        r'\b(he|she|they|them|him|her)\b',
        r'\b(here|there)\b',
        r'^(tell me )?(more|about)',
        r'^(what|how) about',
        r'^(show|give) (me|us)',
        r'^examples?$',
        r'^explain',
        r'^(can you|could you|would you)',
    ]
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def transform_query(
        self,
        query: str,
        methods: List[QueryTransformationMethod],
        chat_history: Optional[List[Dict]] = None,
        max_history_turns: int = 3
    ) -> List[str]:
        """
        Transform a query using specified methods.
        
        Pipeline:
        1. CONTEXTUALIZE (if chat history exists)
        2. Apply MULTI_QUERY and/or HYDE to contextualized query
        
        Args:
            query: The original user query
            methods: List of transformation methods to apply
            chat_history: Recent conversation history
            max_history_turns: Maximum number of conversation turns to consider
            
        Returns:
            List of transformed queries (always includes original/contextualized)
        """
        if not query or not methods:
            return [query]
        
        transformed_queries: Set[str] = set()
        base_query = query
        
        try:
            # Step 1: Contextualize if needed and requested
            if QueryTransformationMethod.CONTEXTUALIZE in methods:
                if chat_history and self._needs_contextualization(query):
                    base_query = await self._contextualize_query(
                        query, chat_history, max_history_turns
                    )
                    transformed_queries.add(base_query)
                else:
                    # No history or doesn't need context, use original
                    transformed_queries.add(query)
            else:
                transformed_queries.add(query)
            
            # Step 2: Apply other methods to the base query
            if QueryTransformationMethod.MULTI_QUERY in methods:
                multi_queries = await self._generate_multi_queries(base_query)
                transformed_queries.update(multi_queries)
            
            if QueryTransformationMethod.HYDE in methods:
                hyde_query = await self._generate_hyde_query(base_query)
                transformed_queries.add(hyde_query)
            
            # Always ensure we have at least the base query
            if not transformed_queries:
                transformed_queries.add(base_query)
            
            return list(transformed_queries)
            
        except Exception as e:
            # Fallback to original query on error
            logger.error(f"Query transformation error: {e}")
            return [query]
    
    def _needs_contextualization(self, query: str) -> bool:
        """
        Determine if a query needs contextualization based on heuristics.
        
        Args:
            query: The user query
            
        Returns:
            True if query likely needs chat history context
        """
        query_lower = query.lower().strip()
        
        # Very short queries often need context
        if len(query_lower.split()) <= 3:
            return True
        
        # Check for context indicator patterns
        for pattern in self.CONTEXT_INDICATORS:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    async def _contextualize_query(
        self,
        query: str,
        chat_history: List[Dict],
        max_history_turns: int
    ) -> str:
        """
        Rewrite query as standalone using chat history.
        
        Args:
            query: Original query
            chat_history: Conversation history
            max_history_turns: Max turns to consider
            
        Returns:
            Contextualized standalone query
        """
        # Take recent history
        recent_history = chat_history[-(max_history_turns * 2):]
        history_text = self._format_chat_history(recent_history)
        
        prompt = f"""
            Given the conversation history below, rewrite the user's latest query as a complete, standalone search query that can be understood without the conversation context.

            Conversation History:
            {history_text}

            Latest User Query: "{query}"

            Rewrite this as a clear, standalone search query that captures the full intent. Include relevant context from the conversation history. Return ONLY the rewritten query, nothing else.

            Standalone Query:
        """

        try:
            contextualized = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            # Clean up the response
            contextualized = contextualized.strip().strip('"').strip("'")
            
            # If contextualization failed or returned empty, use original
            if not contextualized or len(contextualized) < 3:
                return query
            
            return contextualized
            
        except Exception as e:
            logger.error(f"Contextualization error: {e}")
            return query
    
    async def _generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            query: Base query
            num_queries: Number of variations to generate
            
        Returns:
            List of query variations
        """
        prompt = f"""
            Generate {num_queries} alternative search queries that capture the same intent as the original
            query but use different wording, synonyms, or focus on different aspects.

            Original Query: "{query}"

            Generate {num_queries} alternative queries. Each should be a complete, standalone query.
            Format your response as a numbered list:
            1. First alternative query
            2. Second alternative query
            3. Third alternative query

            Alternative Queries:
        """

        try:
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7  # Higher temperature for diversity
            )
            
            # Parse the numbered list
            queries = self._parse_numbered_list(response)
            
            # Ensure we got valid queries
            queries = [q for q in queries if q and len(q.split()) >= 2]
            
            return queries[:num_queries]
            
        except Exception as e:
            logger.error(f"Multi-query generation error: {e}")
            return []
    
    async def _generate_hyde_query(self, query: str) -> str:
        """
        Generate a Hypothetical Document Embedding (HyDE) query.
        
        Creates a hypothetical answer/document that would satisfy the query,
        then uses that for retrieval.
        
        Args:
            query: The user query
            
        Returns:
            Hypothetical document text
        """
        prompt = f"""
            Given the following question, write a short paragraph (2-3 sentences) that would appear
            in a document that answers this question. Write as if you are the document itself,
            not answering the question directly.

            Question: "{query}"

            Write a hypothetical document passage that would contain the answer to this question:
        """

        try:
            hypothetical_doc = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=150,
                temperature=0.5
            )
            
            # Clean up
            hypothetical_doc = hypothetical_doc.strip()
            
            # Ensure we got something useful
            if not hypothetical_doc or len(hypothetical_doc.split()) < 10:
                return query  # Fallback to original query
            
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"HyDE generation error: {e}")
            return query
    
    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for prompts"""
        formatted = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse a numbered list from LLM output"""
        queries = []
        
        # Split by lines and look for numbered items
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1. query" or "1) query"
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if match:
                query = match.group(1).strip().strip('"').strip("'")
                if query:
                    queries.append(query)
        
        return queries
