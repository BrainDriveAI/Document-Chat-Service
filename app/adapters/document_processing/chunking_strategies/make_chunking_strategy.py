from .hierarchical import HierarchicalChunkingStrategy
from .optimized import OptimizedHierarchicalChunkingStrategy
from .semantic import SemanticChunkingStrategy
from .fixed_size import FixedSizeChunkingStrategy
from ....core.ports.chunking_strategy import ChunkingStrategy


# Factory function for easy strategy creation
def create_chunking_strategy(
        strategy_name: str = "optimized_hierarchical",
        **kwargs
) -> ChunkingStrategy:
    """Factory function to create chunking strategies"""

    strategies = {
        "hierarchical": HierarchicalChunkingStrategy,
        "optimized_hierarchical": OptimizedHierarchicalChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "fixed_size": FixedSizeChunkingStrategy
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}. Available: {list(strategies.keys())}")

    return strategies[strategy_name](**kwargs)
