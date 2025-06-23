from abc import ABC
from typing import Dict, Any, Optional, AsyncGenerator
from ...core.ports.llm_service import LLMService


class BaseLLMService(LLMService, ABC):
    """Base class for LLM services with common functionality"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model_info = None

    def _clean_response(self, response: str) -> str:
        """Clean up LLM response by removing common artifacts"""
        # Remove common prefixes/suffixes that models might add
        prefixes_to_remove = [
            "Answer:", "Response:", "Assistant:", "AI:",
            "Based on the context", "According to the context"
        ]

        cleaned = response.strip()

        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(':'):
                    cleaned = cleaned[1:].strip()

        return cleaned

    def _validate_input(self, prompt: str, max_tokens: int) -> None:
        """Validate input parameters"""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if len(prompt) > 50000:  # Reasonable limit
            raise ValueError("Prompt too long")
