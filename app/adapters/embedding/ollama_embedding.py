import httpx
import asyncio
from typing import List, Dict, Any
from ...core.ports.embedding_service import EmbeddingService
from ...core.domain.value_objects.embedding import EmbeddingVector
from ...core.domain.exceptions import EmbeddingGenerationError


class OllamaEmbeddingService(EmbeddingService):
    """Ollama implementation of embedding service"""

    LOCALHOST_URL_PREFIX = "http://localhost"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "mxbai-embed-large",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._model_info = None

    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings_batch([text])
        return embeddings[0]

    def _build_api_url(self) -> str:
        """Return the correct API endpoint based on base_url"""
        if self.base_url.startswith(self.LOCALHOST_URL_PREFIX):
            return f"{self.base_url}/api/embed"
        else:
            return f"{self.base_url}/api/embeddings"

    def _build_api_payload(self, texts: List[str]) -> dict:
        """
        Return the correct payload for local or cloud Ollama embedding API.
        Local supports batch input; cloud only supports single text at a time.
        """
        if self.base_url.startswith(self.LOCALHOST_URL_PREFIX):
            return {
                "model": self.model_name,
                "input": texts  # batch
            }
        else:
            if len(texts) != 1:
                raise ValueError("Cloud API supports one text at a time")
            return {
                "model": self.model_name,
                "prompt": texts[0]
            }

    async def _extract_embeddings(self, response_json: dict, num_texts: int) -> List[List[float]]:
        """Normalize embeddings from local or cloud API into list of lists"""
        if self.base_url.startswith(self.LOCALHOST_URL_PREFIX):
            embeddings = response_json.get("embeddings", [])
            if len(embeddings) != num_texts:
                raise EmbeddingGenerationError(
                    f"Expected {num_texts} embeddings, got {len(embeddings)}"
                )
        else:
            embedding = response_json.get("embedding")
            if not embedding:
                raise EmbeddingGenerationError("No embedding returned from cloud API")
            embeddings = [embedding]
        return embeddings

    async def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts efficiently"""
        try:
            all_embeddings = []
            batch_size = 10
            url = self._build_api_url()

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = self._build_api_payload(batch)

                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                batch_embeddings_values = await self._extract_embeddings(result, len(batch))

                for emb_values in batch_embeddings_values:
                    emb = EmbeddingVector(
                        values=emb_values,
                        model_name=self.model_name,
                        dimensions=len(emb_values)
                    )
                    all_embeddings.append(emb)

                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            return all_embeddings

        except httpx.HTTPError as e:
            raise EmbeddingGenerationError(f"HTTP error generating embeddings: {str(e)}")
        except Exception as e:
            raise EmbeddingGenerationError(f"Error generating embeddings: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the embedding model"""
        if self._model_info is None:
            if "mxbai-embed-large" in self.model_name:
                dimensions = 1024
            elif "nomic-embed-text" in self.model_name:
                dimensions = 768
            elif "all-minilm" in self.model_name:
                dimensions = 384
            else:
                dimensions = 1024

            self._model_info = {
                "model_name": self.model_name,
                "dimensions": dimensions,
                "provider": "ollama",
                "base_url": self.base_url
            }

        return self._model_info

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
