import re
import logging
from typing import List, Dict, Any
from ...core.ports.token_service import TokenService, TokenInfo
from ...core.domain.exceptions import TokenizationError

logger = logging.getLogger(__name__)


class TikTokenService(TokenService):
    """TikToken-based tokenization service for accurate token counting"""

    def __init__(self, model_name: str = "cl100k_base"):
        """
        Initialize with tiktoken encoding

        Args:
            model_name: Encoding name (cl100k_base for GPT-4, p50k_base for older models)
        """
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(model_name)
            self.model_name = model_name
            self._char_to_token_ratio = 4.0  # Rough estimate: 1 token ≈ 4 characters
        except ImportError:
            raise TokenizationError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )
        except Exception as e:
            raise TokenizationError(f"Failed to initialize tokenizer: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text or not text.strip():
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback to character-based estimation
            return self.estimate_tokens_from_chars(len(text))

    def tokenize(self, text: str) -> TokenInfo:
        """Tokenize text and return detailed information"""
        if not text or not text.strip():
            return TokenInfo(token_count=0, tokens=[], token_ids=[])

        try:
            token_ids = self.encoding.encode(text)
            tokens = [self.encoding.decode([token_id]) for token_id in token_ids]

            return TokenInfo(
                token_count=len(token_ids),
                tokens=tokens,
                token_ids=token_ids
            )
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            raise TokenizationError(f"Failed to tokenize text: {str(e)}")

    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not text or not text.strip():
            return []

        try:
            return self.encoding.encode(text)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise TokenizationError(f"Failed to encode text: {str(e)}")

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        if not token_ids:
            return ""

        try:
            return self.encoding.decode(token_ids)
        except Exception as e:
            logger.error(f"Error decoding tokens: {str(e)}")
            raise TokenizationError(f"Failed to decode tokens: {str(e)}")

    def split_text_by_tokens(
            self,
            text: str,
            max_tokens: int,
            overlap_tokens: int = 0
    ) -> List[str]:
        """Split text into chunks by token count with semantic boundary respect"""
        if not text or not text.strip():
            return []

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")

        try:
            # First, try to split by sentences for better semantic boundaries
            sentences = self._split_into_sentences(text)
            chunks = []
            current_chunk_sentences = []
            current_token_count = 0

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)

                # If single sentence exceeds max_tokens, split it further
                if sentence_tokens > max_tokens:
                    # Save current chunk if exists
                    if current_chunk_sentences:
                        chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []
                        current_token_count = 0

                    # Split long sentence by tokens
                    long_sentence_chunks = self._split_long_text_by_tokens(
                        sentence, max_tokens, overlap_tokens
                    )
                    chunks.extend(long_sentence_chunks)
                    continue

                # Check if adding this sentence would exceed max_tokens
                if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk_sentences))

                    # Start new chunk with overlap
                    if overlap_tokens > 0:
                        overlap_sentences = self._get_overlap_sentences(
                            current_chunk_sentences, overlap_tokens
                        )
                        current_chunk_sentences = overlap_sentences
                        current_token_count = self.count_tokens(" ".join(overlap_sentences))
                    else:
                        current_chunk_sentences = []
                        current_token_count = 0

                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens

            # Add final chunk
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

            return [chunk.strip() for chunk in chunks if chunk.strip()]

        except Exception as e:
            logger.error(f"Error splitting text by tokens: {str(e)}")
            # Fallback to simple token-based splitting
            return self._split_long_text_by_tokens(text, max_tokens, overlap_tokens)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences if cleaned_sentences else [text]

    def _split_long_text_by_tokens(
            self,
            text: str,
            max_tokens: int,
            overlap_tokens: int
    ) -> List[str]:
        """Split long text by tokens without sentence boundary respect"""
        try:
            token_ids = self.encode_text(text)
            if len(token_ids) <= max_tokens:
                return [text]

            chunks = []
            start = 0

            while start < len(token_ids):
                end = min(start + max_tokens, len(token_ids))
                chunk_token_ids = token_ids[start:end]
                chunk_text = self.decode_tokens(chunk_token_ids)

                if chunk_text.strip():
                    chunks.append(chunk_text)

                # Move start position considering overlap
                start = end - overlap_tokens
                if start >= len(token_ids):
                    break

            return chunks

        except Exception as e:
            logger.error(f"Error in token-based splitting: {str(e)}")
            # Last resort: character-based splitting with token estimation
            return self._fallback_char_split(text, max_tokens, overlap_tokens)

    def _get_overlap_sentences(
            self,
            sentences: List[str],
            max_overlap_tokens: int
    ) -> List[str]:
        """Get sentences from the end for overlap"""
        if not sentences or max_overlap_tokens <= 0:
            return []

        overlap_sentences = []
        current_tokens = 0

        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def _fallback_char_split(
            self,
            text: str,
            max_tokens: int,
            overlap_tokens: int
    ) -> List[str]:
        """Fallback character-based splitting with token estimation"""
        max_chars = int(max_tokens * self._char_to_token_ratio)
        overlap_chars = int(overlap_tokens * self._char_to_token_ratio)

        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start = end - overlap_chars
            if start >= len(text):
                break

        return chunks

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the tokenizer model"""
        return {
            "model_name": self.model_name,
            "encoding_name": self.encoding.name,
            "char_to_token_ratio": self._char_to_token_ratio,
            "max_token_value": self.encoding.max_token_value if hasattr(self.encoding, 'max_token_value') else None
        }

    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """Rough estimation of tokens from character count"""
        return max(1, int(char_count / self._char_to_token_ratio))
