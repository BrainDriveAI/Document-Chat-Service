import httpx
import asyncio
from typing import List, Dict, Any
from ...core.ports.embedding_service import EmbeddingService
from ...core.domain.value_objects.embedding import EmbeddingVector
from ...core.domain.exceptions import EmbeddingGenerationError


class OllamaEmbeddingService(EmbeddingService):
    """Ollama implementation of embedding service"""

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

    async def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts efficiently"""
        try:
            # Ollama embedding endpoint
            url = f"{self.base_url}/api/embeddings"

            # Process texts in batches to avoid overwhelming the server
            batch_size = 10
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []

                # Process each text in the batch
                for text in batch:
                    payload = {
                        "model": self.model_name,
                        "prompt": text
                    }

                    response = await self.client.post(url, json=payload)
                    response.raise_for_status()

                    result = response.json()
                    embedding_values = result.get("embedding", [])

                    if not embedding_values:
                        raise EmbeddingGenerationError(f"No embedding returned for text: {text[:50]}...")

                    embedding = EmbeddingVector(
                        values=embedding_values,
                        model_name=self.model_name,
                        dimensions=len(embedding_values)
                    )
                    batch_embeddings.append(embedding)

                all_embeddings.extend(batch_embeddings)

                # Small delay between batches to be nice to the server
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
            # For mxbai-embed-large, we know the dimensions
            # In a real implementation, you might query Ollama for this info
            if "mxbai-embed-large" in self.model_name:
                dimensions = 1024
            elif "nomic-embed-text" in self.model_name:
                dimensions = 768
            elif "all-minilm" in self.model_name:
                dimensions = 384
            else:
                dimensions = 1024  # Default assumption

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
