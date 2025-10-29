import httpx
import asyncio
import logging
from typing import List, Dict, Any
from ...core.ports.embedding_service import EmbeddingService
from ...core.domain.value_objects.embedding import EmbeddingVector
from ...core.domain.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)

class OllamaEmbeddingService(EmbeddingService):
    """Ollama implementation with retry logic and better error handling"""
    
    LOCAL_URL_KEYWORDS = ["localhost", "127.0.0.1", "host.docker.internal"]
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "mxbai-embed-large",
        timeout: int = 120,
        batch_size: int = 4,
        concurrency_limit: int = 1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Use more conservative limits for local connections
        limits = httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
            keepalive_expiry=30.0
        )
        
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True
        )
        self._model_info = None
        
    def _is_local(self) -> bool:
        """Detect if Ollama endpoint is local"""
        base = self.base_url.lower()
        return any(kw in base for kw in self.LOCAL_URL_KEYWORDS)
    
    def _build_api_url(self) -> str:
        """Return the correct API endpoint"""
        if self._is_local():
            return f"{self.base_url}/api/embed"
        else:
            return f"{self.base_url}/api/embeddings"
    
    def _build_api_payload(self, text: str) -> dict:
        """Build payload for single text (works for both local and cloud)"""
        if self._is_local():
            return {
                "model": self.model_name,
                "input": text
            }
        else:
            return {
                "model": self.model_name,
                "prompt": text
            }
    
    async def _extract_embeddings(self, response_json: dict) -> List[float]:
        """Extract embedding vector from API response"""
        # Both local and cloud return single embedding with "embedding" key
        if self._is_local():
            embeddings = response_json.get("embeddings")
            if not embeddings or len(embeddings) == 0:
                raise EmbeddingGenerationError("No embeddings in response")
            # Return the first (and only) embedding from the array
            return embeddings[0]
        else:
            embedding = response_json.get("embedding")
            embedding = response_json.get("embedding")
            if not embedding:
                raise EmbeddingGenerationError("No embedding in response")
            return embedding
    
    async def _make_request_with_retry(self, url: str, payload: dict, text_preview: str = "") -> dict:
        """Make HTTP request with exponential backoff retry"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Embedding request attempt {attempt + 1}/{self.max_retries} "
                    f"for text: {text_preview[:50]}..."
                )
                
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{self.max_retries} "
                    f"for text: {text_preview[:50]}..."
                )
                
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error (won't retry): {e.response.status_code} - {e.response.text}")
                    raise EmbeddingGenerationError(
                        f"HTTP {e.response.status_code}: {e.response.text}"
                    )
                logger.warning(
                    f"HTTP error {e.response.status_code} on attempt {attempt + 1}/{self.max_retries}"
                )
                
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Request error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        error_msg = f"All {self.max_retries} retry attempts failed"
        if last_error:
            error_msg += f": {str(last_error)}"
        logger.error(error_msg)
        raise EmbeddingGenerationError(error_msg)
    
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for single text"""
        embeddings = await self.generate_embeddings_batch([text])
        return embeddings[0]
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings with improved error handling and retries"""
        if not texts:
            return []
        
        try:
            all_embeddings = []
            url = self._build_api_url()
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            
            total_texts = len(texts)
            logger.info(
                f"Generating embeddings for {total_texts} texts "
                f"(concurrency={self.concurrency_limit})"
            )
            
            async def process_single_text(text: str, index: int):
                """Process a single text with concurrency control"""
                async with semaphore:
                    try:
                        payload = self._build_api_payload(text)
                        result = await self._make_request_with_retry(
                            url, 
                            payload, 
                            text_preview=text
                        )
                        embedding_values = await self._extract_embeddings(result)
                        
                        logger.debug(f"Successfully processed text {index + 1}/{total_texts}")
                        
                        return EmbeddingVector(
                            values=embedding_values,
                            model_name=self.model_name,
                            dimensions=len(embedding_values)
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to process text {index + 1}/{total_texts}: {str(e)}"
                        )
                        raise
            
            # Create tasks for all texts
            tasks = [
                asyncio.create_task(process_single_text(text, i))
                for i, text in enumerate(texts)
            ]
            
            # Process all tasks and collect results in order
            completed = 0
            for task in asyncio.as_completed(tasks):
                try:
                    embedding = await task
                    all_embeddings.append(embedding)
                    completed += 1
                    
                    # Log progress periodically
                    if completed % 10 == 0 or completed == total_texts:
                        logger.info(f"Progress: {completed}/{total_texts} embeddings generated")
                        
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
                    raise
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except EmbeddingGenerationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_embeddings_batch: {str(e)}")
            raise EmbeddingGenerationError(f"Error generating embeddings: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return embedding model information"""
        if self._model_info is None:
            dimension_map = {
                "mxbai-embed-large": 1024,
                "nomic-embed-text": 768,
                "all-minilm": 384,
            }
            
            dimensions = next(
                (dim for model, dim in dimension_map.items() if model in self.model_name),
                1024  # default
            )
            
            self._model_info = {
                "model_name": self.model_name,
                "dimensions": dimensions,
                "provider": "ollama",
                "base_url": self.base_url,
                "batch_size": self.batch_size,
                "concurrency": self.concurrency_limit
            }
        return self._model_info
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
