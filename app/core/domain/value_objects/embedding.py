from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass(frozen=True)
class EmbeddingVector:
    """Value object representing an embedding vector"""
    values: List[float]
    model_name: str
    dimensions: int
    
    def __post_init__(self):
        if len(self.values) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(self.values)}")
    
    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector"""
        return sum(x * x for x in self.values) ** 0.5


@dataclass(frozen=True)
class SearchQuery:
    """Value object representing a search query"""
    text: str
    collection_id: Optional[str]
    filters: Dict
    top_k: int = 10
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
