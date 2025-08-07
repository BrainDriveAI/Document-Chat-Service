import time
from typing import List, Optional, AsyncGenerator, Dict, Any

from ...core.ports.orchestrator import ChatOrchestrator
from ...core.domain.entities.chat import ChatMessage
from ...core.ports.embedding_service import EmbeddingService
from ...core.ports.vector_store import VectorStore
from ...core.ports.llm_service import LLMService
from ...core.domain.value_objects.embedding import EmbeddingVector


class LangGraphOrchestrator(ChatOrchestrator):
    """
    A simple RAG orchestrator that:
    1. Embeds the user query.
    2. Retrieves top-k similar chunks from VectorStore.
    3. Builds a prompt including retrieved context and optional chat history.
    4. Calls the LLMService to generate a response (streaming or non-streaming).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        # You may configure parameters like top_k, temperature, etc. here or via settings.
        top_k: int = 5,
        max_context_chars: int = 3000,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def process_query(
        self,
        user_query: str,
        session_id: str,
        collection_id: Optional[str] = None,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> ChatMessage:
        """
        Non-streaming processing: returns a ChatMessage with full response.
        """
        # Record start for response time
        start_time = time.time()

        # 1. Embed the user query
        embedding: EmbeddingVector = await self.embedding_service.generate_embedding(user_query)

        # 2. Retrieve similar chunks
        # Passing filters=None; vector_store.search_similar handles collection_id
        retrieved_chunks = await self.vector_store.search_similar(
            query_embedding=embedding,
            collection_id=collection_id,
            top_k=self.top_k,
            filters=None
        )
        # Extract IDs and contents
        chunk_ids = [chunk.id for chunk in retrieved_chunks]
        # Build context text by concatenating chunk contents, truncated if too long
        context_text = self._build_context_text(retrieved_chunks)

        # 3. Incorporate chat history into prompt if provided
        prompt = self._build_prompt(user_query, chat_history, context_text)

        # 4. Call LLM to generate response
        assistant_response = await self.llm_service.generate_response(
            prompt=prompt,
            context=None,  # our prompt already includes context
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Compute response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # 5. Create ChatMessage entity
        chat_message = ChatMessage.create(
            session_id=session_id,
            user_message=user_query,
            assistant_response=assistant_response,
            retrieved_chunks=chunk_ids,
            response_time_ms=response_time_ms,
            collection_id=collection_id
        )
        return chat_message

    async def process_streaming_query(
        self,
        user_query: str,
        session_id: str,
        collection_id: Optional[str] = None,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming processing: yields dictionaries with chunks of the response.
        The final save of ChatMessage should occur after streaming completes (e.g., in the route or use-case).
        Each yielded dict has at least {"response": <str>} and on first yield also {"retrieved_chunks": [...]}.
        """
        # 1. Embed and retrieve as above
        embedding: EmbeddingVector = await self.embedding_service.generate_embedding(user_query)
        retrieved_chunks = await self.vector_store.search_similar(
            query_embedding=embedding,
            collection_id=collection_id,
            top_k=self.top_k,
            filters=None
        )
        chunk_ids = [chunk.id for chunk in retrieved_chunks]
        context_text = self._build_context_text(retrieved_chunks)
        prompt = self._build_prompt(user_query, chat_history, context_text)

        # 2. Stream from LLM
        # We'll yield the retrieved_chunks on the first chunk
        first = True
        async for chunk in self.llm_service.generate_streaming_response(
            prompt=prompt,
            context=None,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        ):
            data: Dict[str, Any] = {"response": chunk}
            if first:
                data["retrieved_chunks"] = chunk_ids
                first = False
            yield data
        # Note: final ChatMessage creation/saving should be done by the use-case after streaming ends.

    def _build_context_text(self, chunks: List[Any]) -> str:
        """
        Combine chunk contents into a single context string.
        Truncate if exceeding max_context_chars by keeping earlier chunks first.
        """
        texts = []
        total_len = 0
        for chunk in chunks:
            content = chunk.content.strip()
            if not content:
                continue
            # If adding this chunk exceeds limit, break
            if total_len + len(content) > self.max_context_chars:
                # Optionally, append truncated part of content
                remaining = self.max_context_chars - total_len
                if remaining > 0:
                    texts.append(content[:remaining])
                break
            texts.append(content)
            total_len += len(content)
        # Join with separators
        return "\n\n---\n\n".join(texts)

    def _build_prompt(
        self,
        user_query: str,
        chat_history: Optional[List[ChatMessage]],
        context_text: str
    ) -> str:
        """
        Build the prompt for the LLM, including optional chat history and retrieved context.
        """
        prompt_parts = []
        # If there is chat history, include it
        if chat_history:
            # Format as alternating user/assistant lines
            history_texts = []
            for msg in chat_history:
                # Only include up to recent N messages; chat_history passed in should be limited by caller
                history_texts.append(f"User: {msg.user_message}")
                history_texts.append(f"Assistant: {msg.assistant_response}")
            prompt_parts.append("Here is the previous conversation:\n" + "\n".join(history_texts))
        # Include retrieved context
        if context_text:
            prompt_parts.append("Here are some relevant context excerpts from documents:\n" + context_text)
        # Finally, the user question
        prompt_parts.append(f"User question: {user_query}\nAnswer:")
        # Join with double newline
        return "\n\n".join(prompt_parts)
