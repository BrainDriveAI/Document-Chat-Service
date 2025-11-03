"""
Centralized prompt templates for RAG operations.
These prompts are optimized for smaller LLMs (< 4B parameters).
"""

class PromptTemplates:
    """Collection of prompt templates for various RAG operations"""
    
    # ============================================================================
    # QUERY TRANSFORMATION PROMPTS
    # ============================================================================
    
    CONTEXTUALIZE_QUERY = """Given the conversation history, rewrite the user's latest query as a complete, standalone search query.

Conversation History:
{chat_history}

Latest Query: "{query}"

Instructions:
- Make the query self-contained and clear
- Include relevant context from the conversation
- Keep it concise (one sentence)
- Return ONLY the rewritten query

Standalone Query:"""

    MULTI_QUERY_GENERATION = """Generate {num_queries} alternative search queries for the same topic using different wording.

Original Query: "{query}"

Instructions:
- Use different terminology and synonyms
- Focus on different aspects of the topic
- Each query should be complete and standalone
- Format as numbered list

Alternative Queries:"""

    HYDE_GENERATION = """Write a short passage (2-3 sentences) that would appear in a document answering this question.

Question: "{query}"

Instructions:
- Write as if you are the document itself
- Be factual and informative
- Don't write "The answer is..." - write the actual content

Document Passage:"""

    # ============================================================================
    # INTENT CLASSIFICATION PROMPTS
    # ============================================================================
    
    INTENT_CLASSIFICATION = """Analyze the user's intent based on their query and conversation history.

Conversation History:
{chat_history}

Current Query: "{query}"

Intent Types:
- RETRIEVAL: Search for specific information
- SUMMARY: Summarize document(s)
- CLARIFICATION: Follow-up about previous answer
- COMPARISON: Compare concepts
- LISTING: List items/features
- CHAT: General conversation (no search needed)

Respond in JSON:
{{
    "intent": "INTENT_TYPE",
    "requires_retrieval": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    # ============================================================================
    # COLLECTION SUMMARY PROMPTS
    # ============================================================================
    
    COLLECTION_GENERAL_SUMMARY = """Based on these excerpts from a document collection, provide a comprehensive summary.

Cover:
1. Main topics and themes
2. Key concepts discussed
3. Overall scope and purpose

Be concise but comprehensive (3-5 paragraphs).

Document Excerpts:
{context}

Collection Summary:"""

    COLLECTION_TARGETED_SUMMARY = """Based on these excerpts, answer this question about the collection.

Question: {query}

Document Excerpts:
{context}

Instructions:
- Provide a comprehensive answer
- If excerpts don't fully answer, note what is available
- Be specific and cite information from excerpts

Answer:"""

    # ============================================================================
    # CONTEXTUAL RETRIEVAL PROMPTS
    # ============================================================================
    
    CONTEXTUAL_CHUNK = """<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    # ============================================================================
    # RAG RESPONSE GENERATION PROMPTS
    # ============================================================================
    
    RAG_RESPONSE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer based on the context above
- Be accurate and concise
- If the context doesn't contain the answer, say so
- Don't make up information

Answer:"""

    RAG_RESPONSE_WITH_HISTORY = """You are a helpful assistant. Use the context and conversation history to answer accurately.

Conversation History:
{chat_history}

Context:
{context}

Question: {query}

Instructions:
- Consider previous conversation
- Answer based on the provided context
- Be accurate and concise
- Maintain conversation continuity

Answer:"""

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def format_chat_history(
        chat_history: list,
        max_turns: int = 3,
        max_chars_per_message: int = 200
    ) -> str:
        """Format chat history for prompts"""
        if not chat_history:
            return "No previous conversation."
        
        recent = chat_history[-(max_turns * 2):]
        formatted = []
        
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            
            # Truncate long messages
            if len(content) > max_chars_per_message:
                content = content[:max_chars_per_message] + "..."
            
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_document_excerpts(
        chunks: list,
        max_chars: int = 8000
    ) -> str:
        """Format document chunks as excerpts"""
        excerpts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            excerpt = f"[Excerpt {i}]\n{chunk.content}\n"
            excerpt_len = len(excerpt)
            
            if current_length + excerpt_len > max_chars:
                break
            
            excerpts.append(excerpt)
            current_length += excerpt_len
        
        return "\n".join(excerpts)
