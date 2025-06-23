from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TokenInfo:
    """Information about tokenized text"""
    token_count: int
    tokens: List[str]
    token_ids: Optional[List[int]] = None


class TokenService(ABC):
    """Port for tokenization services"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> TokenInfo:
        """Tokenize text and return detailed information"""
        pass

    @abstractmethod
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        pass

    @abstractmethod
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        pass

    @abstractmethod
    def split_text_by_tokens(
            self,
            text: str,
            max_tokens: int,
            overlap_tokens: int = 0
    ) -> List[str]:
        """Split text into chunks by token count"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the tokenizer model"""
        pass

    @abstractmethod
    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """Rough estimation of tokens from character count"""
        pass
