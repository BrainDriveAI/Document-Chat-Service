import httpx
import json
from typing import Dict, Any, Optional, AsyncGenerator
from ...core.ports.llm_service import LLMService
from ...core.domain.exceptions import DomainException


class OllamaLLMError(DomainException):
    """Ollama-specific LLM error"""
    pass


class OllamaLLMService(LLMService):
    """Ollama implementation of LLM service"""

    def __init__(
            self,
            base_url: str = "http://localhost:11434",
            model_name: str = "llama3.2:3b",
            timeout: int = 120
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._model_info = None

    async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.1
    ) -> str:
        """Generate a response using the LLM"""
        try:
            # Build the full prompt with context if provided
            full_prompt = self._build_prompt(prompt, context)

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                },
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except httpx.HTTPError as e:
            raise OllamaLLMError(f"HTTP error generating response: {str(e)}")
        except Exception as e:
            raise OllamaLLMError(f"Error generating response: {str(e)}")

    async def generate_streaming_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using the LLM"""
        try:
            full_prompt = self._build_prompt(prompt, context)

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                },
                "stream": True
            }

            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]

                            # Check if generation is done
                            if chunk.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPError as e:
            raise OllamaLLMError(f"HTTP error generating streaming response: {str(e)}")
        except Exception as e:
            raise OllamaLLMError(f"Error generating streaming response: {str(e)}")

    async def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the provided text"""
        prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

Text: {text}

Summary:"""

        return await self.generate_response(
            prompt=prompt,
            max_tokens=max_length + 50,
            temperature=0.1
        )

    async def generate_question(self, text: str) -> str:
        """Generate a question that the provided text chunk answers"""
        prompt = f"""Based on the following text, generate a clear question that this text would answer:

Text: {text}

Question:"""

        response = await self.generate_response(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        )

        # Clean up the response to return just the question
        return response.strip().rstrip('?') + '?'

    async def generate_multi_queries(self, query: str) -> list[str]:
        """Generate multiple related search queries from the original one"""
        prompt = f"""
            I have a search query: "{query}".
            Please generate 3 alternative queries that are on the same topic but use different wording or focus on different aspects.
            Return the queries as a numbered list. For example:
            1. First query
            2. Second query
            3. Third query
        """
        response = await self.generate_response(prompt)
        print(f"MULTI QUERIES LLM RESPONSE: {response}")
        queries = [line.strip().split(". ", 1)[1] for line in response.split("\n") if line.strip()]
        queries.append(query)
        return queries

    def _build_prompt(self, user_prompt: str, context: Optional[str] = None) -> str:
        """Build the full prompt with context for RAG"""
        if context:
            prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use the context to answer the user's question accurately and concisely.

Context:
{context}

Question: {user_prompt}

Answer based on the context above:
"""
        else:
            prompt = f"""You are a helpful assistant. Please answer the following question:

Question: {user_prompt}

Answer:
"""

        return prompt

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the LLM model"""
        if self._model_info is None:
            self._model_info = {
                "model_name": self.model_name,
                "provider": "ollama",
                "base_url": self.base_url,
                "capabilities": ["text_generation", "streaming", "summarization"]
            }

        return self._model_info

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
