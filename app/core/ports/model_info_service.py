"""
Port interface for model information service.

Defines abstract interface for retrieving model metadata like context windows.
"""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Model metadata information"""
    name: str
    context_window: int
    parameter_count: Optional[str] = None  # e.g., "7B", "3B"
    quantization: Optional[str] = None     # e.g., "Q4_K_M"
    family: Optional[str] = None           # e.g., "llama", "qwen"


class ModelInfoService(ABC):
    """
    Abstract interface for retrieving model information.

    Implementations should cache results to avoid repeated API calls.
    """

    @abstractmethod
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get model information including context window size.

        Args:
            model_name: Name of the model (e.g., "llama3.2:8b")

        Returns:
            ModelInfo if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_context_window(self, model_name: str) -> int:
        """
        Get context window size for a model.

        Args:
            model_name: Name of the model

        Returns:
            Context window size in tokens (returns default if not found)
        """
        pass

    @abstractmethod
    async def refresh_cache(self) -> None:
        """Refresh the cached model information"""
        pass
