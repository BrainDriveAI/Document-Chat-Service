"""
Ollama implementation of ModelInfoService.

Queries Ollama API to detect model context windows dynamically.
"""
import logging
import re
from typing import Optional, Dict
from pydantic import HttpUrl
import httpx

from ...core.ports.model_info_service import ModelInfoService, ModelInfo

logger = logging.getLogger(__name__)


class OllamaModelInfoAdapter(ModelInfoService):
    """
    Ollama adapter for retrieving model information.

    Queries Ollama API to detect context windows dynamically.
    Caches results to avoid repeated API calls.
    """

    def __init__(
        self,
        base_url: HttpUrl,
        default_context_window: int = 4096,
        cache_ttl_seconds: int = 3600
    ):
        self.base_url = str(base_url).rstrip('/')
        self.default_context_window = default_context_window
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, ModelInfo] = {}
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get model information from Ollama API.

        First checks cache, then queries Ollama if not cached.
        """
        # Check cache first
        if model_name in self._cache:
            logger.debug(f"Using cached model info for: {model_name}")
            return self._cache[model_name]

        # Query Ollama API
        try:
            logger.debug(f"Querying Ollama for model info: {model_name}")
            url = f"{self.base_url}/api/show"
            payload = {"name": model_name}

            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract context window from parameters
            context_window = self._extract_context_window(data)

            # Extract other metadata
            model_info = ModelInfo(
                name=model_name,
                context_window=context_window or self.default_context_window,
                parameter_count=self._extract_parameter_count(model_name),
                quantization=self._extract_quantization(data),
                family=self._extract_family(data)
            )

            # Cache the result
            self._cache[model_name] = model_info
            logger.info(f"Detected context window for {model_name}: {model_info.context_window}")

            return model_info

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Model not found in Ollama: {model_name}")
            else:
                logger.error(f"HTTP error querying Ollama for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None

    async def get_context_window(self, model_name: str) -> int:
        """
        Get context window size for a model.

        Returns default if model not found or detection fails.
        """
        model_info = await self.get_model_info(model_name)
        if model_info:
            return model_info.context_window

        logger.warning(f"Could not detect context window for {model_name}, using default: {self.default_context_window}")
        return self.default_context_window

    async def refresh_cache(self) -> None:
        """Clear cache to force re-detection on next query"""
        logger.info("Refreshing model info cache")
        self._cache.clear()

    async def get_all_local_models(self) -> list[ModelInfo]:
        """
        Get information for all locally available Ollama models.

        Useful for initialization or UI display.
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                model_name = model_data.get("name")
                if model_name:
                    model_info = await self.get_model_info(model_name)
                    if model_info:
                        models.append(model_info)

            return models

        except Exception as e:
            logger.error(f"Error listing local Ollama models: {e}")
            return []

    def _extract_context_window(self, model_data: dict) -> Optional[int]:
        """
        Extract context window from Ollama model parameters.

        Looks for 'num_ctx' parameter in the format: "num_ctx 8192"
        """
        parameters = model_data.get("parameters", [])

        # Parameters can be a list of strings like ["num_ctx 8192", "stop <|im_end|>"]
        if isinstance(parameters, list):
            for param in parameters:
                if isinstance(param, str) and param.startswith("num_ctx"):
                    try:
                        # Parse "num_ctx 8192" -> 8192
                        parts = param.split()
                        if len(parts) >= 2:
                            return int(parts[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse num_ctx from: {param}")

        # Also check model_info if available (highest priority for architecture-specific context)
        if "model_info" in model_data:
            model_info = model_data["model_info"]

            # Check architecture-specific context length (e.g., "gemma3.context_length": 32768)
            for key, value in model_info.items():
                if key.endswith(".context_length") or key.endswith(".context_window"):
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        pass

            # Fallback to generic keys
            for key in ["num_ctx", "context_length", "max_context_length"]:
                if key in model_info:
                    try:
                        return int(model_info[key])
                    except (ValueError, TypeError):
                        pass

        return None

    def _extract_parameter_count(self, model_name: str) -> Optional[str]:
        """
        Extract parameter count from model name.

        Examples: "llama3.2:8b" -> "8B", "qwen2.5:0.5b" -> "0.5B"
        """
        # Match patterns like ":8b", ":0.5b", "-7b"
        match = re.search(r'[:_-](\d+\.?\d*)(b|B)', model_name)
        if match:
            return match.group(1).upper() + "B"
        return None

    def _extract_quantization(self, model_data: dict) -> Optional[str]:
        """
        Extract quantization info if available.

        Common values: Q4_K_M, Q5_K_S, Q8_0, etc.
        """
        # Check in details or modelfile
        details = model_data.get("details", {})
        quantization = details.get("quantization_level")
        if quantization:
            return quantization

        # Try to parse from modelfile
        modelfile = model_data.get("modelfile", "")
        quant_match = re.search(r'(Q\d+_[KF]_[MS]|Q\d+_\d+|F16|F32)', modelfile, re.IGNORECASE)
        if quant_match:
            return quant_match.group(1).upper()

        return None

    def _extract_family(self, model_data: dict) -> Optional[str]:
        """
        Extract model family (llama, qwen, phi, etc.)
        """
        details = model_data.get("details", {})
        family = details.get("family")
        if family:
            return family.lower()

        # Fallback: try to parse from model name in details
        parent_model = details.get("parent_model", "")
        if parent_model:
            # Extract family from parent model name
            for known_family in ["llama", "qwen", "phi", "mistral", "gemma", "mixtral"]:
                if known_family in parent_model.lower():
                    return known_family

        return None

    async def close(self):
        """Close HTTP client"""
        await self._client.aclose()
