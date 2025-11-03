from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator


class LLMService(ABC):
    """Port for Large Language Model services"""

    @abstractmethod
    async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.1
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: User question/prompt
            context: Retrieved context for RAG
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        pass

    @abstractmethod
    async def generate_streaming_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the LLM.

        Args:
            prompt: User question/prompt
            context: Retrieved context for RAG
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Response chunks as they're generated
        """
        pass

    @abstractmethod
    async def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the provided text"""
        pass

    @abstractmethod
    async def generate_question(self, text: str) -> str:
        """Generate a question that the provided text chunk answers"""
        pass

    @abstractmethod
    async def generate_multi_queries(self, query: str) -> List[str]:
        """Generate multiple related search queries from the original one"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the LLM model"""
        pass
