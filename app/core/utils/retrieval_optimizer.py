"""
Utilities for optimizing retrieval based on model context windows.

Calculates optimal top_k and chunk limits based on available token budget.
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count from text.

    Uses approximation: 1 token ≈ 4 characters for English text.
    This is conservative (actual is often 1 token ≈ 3-4 chars).
    """
    return len(text) // 4


@dataclass
class RetrievalBudget:
    """Token budget calculation for RAG retrieval"""
    context_window: int
    system_prompt_tokens: int
    user_query_tokens: int
    chat_history_tokens: int
    generation_buffer_tokens: int
    safety_margin: float  # fraction of context to use (e.g., 0.75)

    @property
    def available_for_chunks(self) -> int:
        """Calculate tokens available for retrieved chunks"""
        total_overhead = (
            self.system_prompt_tokens
            + self.user_query_tokens
            + self.chat_history_tokens
            + self.generation_buffer_tokens
        )
        available = int(self.context_window * self.safety_margin) - total_overhead
        return max(0, available)


@dataclass
class OptimalRetrievalConfig:
    """Calculated optimal retrieval configuration"""
    top_k: int
    max_chunk_tokens: int
    total_chunk_budget: int
    context_window: int
    reverse_for_ollama: bool = True

    def __str__(self):
        return (
            f"OptimalRetrievalConfig(top_k={self.top_k}, "
            f"max_chunk_tokens={self.max_chunk_tokens}, "
            f"context_window={self.context_window})"
        )


def calculate_optimal_retrieval(
    context_window: int,
    system_prompt_tokens: Optional[int] = None,
    user_query_tokens: Optional[int] = None,
    chat_history_tokens: Optional[int] = None,
    avg_chunk_tokens: int = 200,  # ~800 chars at 1:4 ratio
    safety_margin: float = 0.75,
    min_top_k: int = 2,
    max_top_k: int = 10,
    generation_buffer: int = 512  # Reserve for model's response
) -> OptimalRetrievalConfig:
    """
    Calculate optimal retrieval parameters based on model context window.

    Args:
        context_window: Model's context window size in tokens
        system_prompt_tokens: Estimated tokens in system prompt (auto-estimate if None)
        user_query_tokens: Estimated tokens in user query (auto-estimate if None)
        chat_history_tokens: Tokens in chat history (0 if no history)
        avg_chunk_tokens: Average tokens per chunk
        safety_margin: Use only this fraction of context window (default: 0.75)
        min_top_k: Minimum number of chunks to retrieve
        max_top_k: Maximum number of chunks to retrieve
        generation_buffer: Reserve tokens for model's response

    Returns:
        OptimalRetrievalConfig with calculated parameters
    """
    # Auto-estimate if not provided
    if system_prompt_tokens is None:
        system_prompt_tokens = 150  # Typical RAG system prompt
    if user_query_tokens is None:
        user_query_tokens = 50  # Typical question length
    if chat_history_tokens is None:
        chat_history_tokens = 0

    # Calculate budget
    budget = RetrievalBudget(
        context_window=context_window,
        system_prompt_tokens=system_prompt_tokens,
        user_query_tokens=user_query_tokens,
        chat_history_tokens=chat_history_tokens,
        generation_buffer_tokens=generation_buffer,
        safety_margin=safety_margin
    )

    available_tokens = budget.available_for_chunks

    # Calculate optimal top_k
    if available_tokens <= 0:
        logger.warning(
            f"No tokens available for chunks! Context window too small: {context_window}"
        )
        optimal_top_k = min_top_k
        max_chunk_tokens = avg_chunk_tokens
    else:
        optimal_top_k = max(
            min_top_k,
            min(max_top_k, available_tokens // avg_chunk_tokens)
        )
        max_chunk_tokens = available_tokens // optimal_top_k if optimal_top_k > 0 else avg_chunk_tokens

    config = OptimalRetrievalConfig(
        top_k=optimal_top_k,
        max_chunk_tokens=max_chunk_tokens,
        total_chunk_budget=available_tokens,
        context_window=context_window,
        reverse_for_ollama=True
    )

    logger.debug(
        f"Calculated optimal retrieval: context_window={context_window}, "
        f"available_tokens={available_tokens}, top_k={optimal_top_k}, "
        f"max_chunk_tokens={max_chunk_tokens}"
    )

    return config


def calculate_retrieval_with_history(
    context_window: int,
    chat_history: str,
    user_query: str,
    system_prompt: str = "",
    **kwargs
) -> OptimalRetrievalConfig:
    """
    Calculate optimal retrieval with actual text content.

    Convenience wrapper that estimates tokens from actual text.
    """
    return calculate_optimal_retrieval(
        context_window=context_window,
        system_prompt_tokens=estimate_tokens(system_prompt) if system_prompt else 150,
        user_query_tokens=estimate_tokens(user_query),
        chat_history_tokens=estimate_tokens(chat_history) if chat_history else 0,
        **kwargs
    )
