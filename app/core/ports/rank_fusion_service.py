from abc import ABC, abstractmethod
from typing import List
from enum import Enum
from ..domain.entities.document_chunk import DocumentChunk


class FusionMethod(Enum):
    RECIPROCAL_RANK_FUSION = "rrf"
    WEIGHTED_SCORE = "weighted"
    CONVEX_COMBINATION = "convex"
    DISTRIBUTION_BASED = "distribution"


class RankFusionService(ABC):
    """Port for rank fusion services"""

    @abstractmethod
    def fuse_results(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            method: FusionMethod = FusionMethod.RECIPROCAL_RANK_FUSION,
            alpha: float = 0.5,
            top_k: int = 10,
            **kwargs
    ) -> List[DocumentChunk]:
        """Combine search results using specified fusion method"""
        pass
