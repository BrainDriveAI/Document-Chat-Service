"""
Tests for Dynamic Context Window Detection feature.

Tests the automatic detection of model context windows from Ollama
and optimization of retrieval based on detected token budgets.

Local test models:
- gemma3:1b (LLM)
- mxbai-embed-large (embeddings)
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from app.adapters.model_info.ollama_model_info import OllamaModelInfoAdapter
from app.core.ports.model_info_service import ModelInfo
from app.core.utils.retrieval_optimizer import (
    calculate_optimal_retrieval,
    calculate_retrieval_with_history,
    estimate_tokens,
    OptimalRetrievalConfig
)


# ==============================================================================
# Unit Tests - Model Info Service
# ==============================================================================

class TestOllamaModelInfoAdapter:
    """Test Ollama model info detection"""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance for testing"""
        return OllamaModelInfoAdapter(
            base_url="http://localhost:11434",
            default_context_window=4096
        )

    @pytest.mark.asyncio
    async def test_detect_gemma3_context_window(self, adapter):
        """Test context window detection for local gemma3:1b model"""
        model_name = "gemma3:1b"

        context_window = await adapter.get_context_window(model_name)

        # gemma3:1b typically has 8192 token context window
        assert context_window > 0, "Context window should be positive"
        assert context_window >= 4096, "Context window should be at least 4K"
        print(f"✓ Detected {model_name} context window: {context_window} tokens")

    @pytest.mark.asyncio
    async def test_get_model_info_full(self, adapter):
        """Test full model info retrieval for gemma3:1b"""
        model_name = "gemma3:1b"

        model_info = await adapter.get_model_info(model_name)

        assert model_info is not None, "Model info should not be None"
        assert model_info.name == model_name
        assert model_info.context_window > 0
        assert model_info.parameter_count == "1B", "Should extract 1B from model name"
        assert model_info.family is not None, "Should detect model family"

        print(f"✓ Model Info:")
        print(f"  Name: {model_info.name}")
        print(f"  Context Window: {model_info.context_window}")
        print(f"  Parameters: {model_info.parameter_count}")
        print(f"  Family: {model_info.family}")
        print(f"  Quantization: {model_info.quantization}")

    @pytest.mark.asyncio
    async def test_cache_works(self, adapter):
        """Test that caching prevents duplicate API calls"""
        model_name = "gemma3:1b"

        # First call - should query API
        context_window_1 = await adapter.get_context_window(model_name)

        # Second call - should use cache (much faster)
        import time
        start = time.time()
        context_window_2 = await adapter.get_context_window(model_name)
        elapsed = time.time() - start

        assert context_window_1 == context_window_2, "Cached value should match"
        assert elapsed < 0.01, "Cached lookup should be near-instant"
        print(f"✓ Cache working: {elapsed*1000:.2f}ms for cached lookup")

    @pytest.mark.asyncio
    async def test_nonexistent_model_fallback(self, adapter):
        """Test fallback to default for non-existent model"""
        model_name = "nonexistent-model-xyz"

        context_window = await adapter.get_context_window(model_name)

        assert context_window == 4096, "Should fallback to default context window"
        print(f"✓ Fallback working: {model_name} → {context_window} (default)")

    @pytest.mark.asyncio
    async def test_list_all_local_models(self, adapter):
        """Test listing all locally available Ollama models"""
        models = await adapter.get_all_local_models()

        assert len(models) > 0, "Should find at least one local model"

        print(f"✓ Found {len(models)} local Ollama models:")
        for model in models:
            print(f"  - {model.name}: {model.context_window} tokens ({model.parameter_count})")

        # Verify gemma3:1b is in the list
        gemma_models = [m for m in models if "gemma" in m.name.lower()]
        assert len(gemma_models) > 0, "Should find gemma3:1b in local models"

    @pytest.mark.asyncio
    async def test_refresh_cache(self, adapter):
        """Test cache refresh functionality"""
        model_name = "gemma3:1b"

        # Populate cache
        await adapter.get_context_window(model_name)
        assert len(adapter._cache) > 0, "Cache should have entries"

        # Refresh cache
        await adapter.refresh_cache()
        assert len(adapter._cache) == 0, "Cache should be empty after refresh"

        print("✓ Cache refresh working")


# ==============================================================================
# Unit Tests - Retrieval Optimizer
# ==============================================================================

class TestRetrievalOptimizer:
    """Test token budget calculation and optimal retrieval config"""

    def test_estimate_tokens(self):
        """Test token estimation from text"""
        # 1 token ≈ 4 characters (conservative estimate)
        text_100_chars = "a" * 100
        text_400_chars = "a" * 400

        assert estimate_tokens(text_100_chars) == 25  # 100 / 4
        assert estimate_tokens(text_400_chars) == 100  # 400 / 4

        print("✓ Token estimation working (1 token ≈ 4 chars)")

    def test_calculate_optimal_retrieval_small_model(self):
        """Test optimal retrieval for small context window model"""
        # Simulate phi3:mini with 4K context window
        config = calculate_optimal_retrieval(
            context_window=4096,
            system_prompt_tokens=150,
            user_query_tokens=50,
            chat_history_tokens=0,
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        assert isinstance(config, OptimalRetrievalConfig)
        assert config.context_window == 4096
        assert config.top_k >= 2, "Should retrieve at least MIN_TOP_K chunks"
        assert config.top_k <= 10, "Should not exceed MAX_TOP_K chunks"
        assert config.total_chunk_budget > 0

        print(f"✓ Small model (4K context): top_k={config.top_k}, "
              f"max_chunk_tokens={config.max_chunk_tokens}")

    def test_calculate_optimal_retrieval_gemma3_1b(self):
        """Test optimal retrieval for gemma3:1b (8K context window)"""
        config = calculate_optimal_retrieval(
            context_window=8192,  # gemma3:1b
            system_prompt_tokens=150,
            user_query_tokens=50,
            chat_history_tokens=500,
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        assert config.top_k >= 4, "8K model should retrieve at least 4 chunks"
        assert config.total_chunk_budget > 2000, "Should have reasonable chunk budget"

        print(f"✓ gemma3:1b (8K context): top_k={config.top_k}, "
              f"budget={config.total_chunk_budget} tokens")

    def test_calculate_optimal_retrieval_large_model(self):
        """Test optimal retrieval for large context window model"""
        # Simulate qwen2.5:7b with 32K context window
        config = calculate_optimal_retrieval(
            context_window=32768,
            system_prompt_tokens=150,
            user_query_tokens=50,
            chat_history_tokens=1000,
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        assert config.top_k == 10, "Large model should hit MAX_TOP_K"
        assert config.total_chunk_budget > 10000, "Should have large chunk budget"

        print(f"✓ Large model (32K context): top_k={config.top_k}, "
              f"budget={config.total_chunk_budget} tokens")

    def test_calculate_retrieval_with_history(self):
        """Test convenience wrapper with actual text"""
        config = calculate_retrieval_with_history(
            context_window=8192,
            chat_history="User: Hello\nAssistant: Hi there!",
            user_query="What is machine learning?",
            system_prompt="You are a helpful assistant.",
            avg_chunk_tokens=200
        )

        assert isinstance(config, OptimalRetrievalConfig)
        assert config.top_k >= 2

        print(f"✓ Retrieval with history: top_k={config.top_k}")

    def test_constrained_budget(self):
        """Test behavior when token budget is very constrained"""
        # Very small context window + lots of overhead
        config = calculate_optimal_retrieval(
            context_window=2048,
            system_prompt_tokens=500,
            user_query_tokens=200,
            chat_history_tokens=1000,
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        # Should still return MIN_TOP_K even with tight budget
        assert config.top_k == 2, "Should return MIN_TOP_K when budget is tight"

        print(f"✓ Constrained budget: top_k={config.top_k} (forced to minimum)")


# ==============================================================================
# Integration Tests - End-to-End
# ==============================================================================

class TestDynamicContextWindowIntegration:
    """
    Integration tests requiring Ollama to be running locally
    with gemma3:1b and mxbai-embed-large models installed.
    """

    @pytest.fixture
    def adapter(self):
        """Create real adapter instance"""
        return OllamaModelInfoAdapter(
            base_url="http://localhost:11434",
            default_context_window=4096
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_gemma3_optimization(self, adapter):
        """
        End-to-end test: Detect gemma3:1b context window and calculate
        optimal retrieval config.
        """
        model_name = "gemma3:1b"

        # Step 1: Detect context window
        context_window = await adapter.get_context_window(model_name)
        print(f"\n1. Detected {model_name} context window: {context_window} tokens")

        # Step 2: Calculate optimal retrieval
        config = calculate_optimal_retrieval(
            context_window=context_window,
            system_prompt_tokens=150,
            user_query_tokens=50,
            chat_history_tokens=500,
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        print(f"2. Calculated optimal retrieval:")
        print(f"   - top_k: {config.top_k} chunks")
        print(f"   - max_chunk_tokens: {config.max_chunk_tokens}")
        print(f"   - total_chunk_budget: {config.total_chunk_budget} tokens")
        print(f"   - reverse_for_ollama: {config.reverse_for_ollama}")

        # Assertions
        assert context_window > 0
        assert config.top_k >= 2
        assert config.total_chunk_budget > 0
        assert config.reverse_for_ollama is True

        print("✓ E2E optimization complete for gemma3:1b!")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_with_long_history(self, adapter):
        """Test optimization with substantial chat history"""
        model_name = "gemma3:1b"

        # Simulate long chat history
        chat_history = "\n".join([
            f"User: Question {i}\nAssistant: Answer {i}" for i in range(5)
        ])
        chat_history_tokens = estimate_tokens(chat_history)

        # Detect context window
        context_window = await adapter.get_context_window(model_name)

        # Calculate with history
        config = calculate_retrieval_with_history(
            context_window=context_window,
            chat_history=chat_history,
            user_query="What are the key points?",
            system_prompt="You are an expert assistant.",
            avg_chunk_tokens=200
        )

        print(f"\n✓ E2E with history:")
        print(f"  - Context window: {context_window}")
        print(f"  - History tokens: {chat_history_tokens}")
        print(f"  - Optimal top_k: {config.top_k}")

        assert config.top_k >= 2


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require running Ollama)"
    )


# ==============================================================================
# Manual Test Runner
# ==============================================================================

async def main():
    """
    Manual test runner for quick verification.

    Run with: python -m pytest tests/test_dynamic_context_window.py -v
    Or: python tests/test_dynamic_context_window.py
    """
    print("=" * 70)
    print("Dynamic Context Window Detection - Manual Test")
    print("=" * 70)

    # Test 1: Model Info Service
    print("\n[Test 1] Testing OllamaModelInfoAdapter...")
    adapter = OllamaModelInfoAdapter(
        base_url="http://localhost:11434",
        default_context_window=4096
    )

    try:
        model_info = await adapter.get_model_info("gemma3:1b")
        print(f"✓ Model: {model_info.name}")
        print(f"✓ Context Window: {model_info.context_window} tokens")
        print(f"✓ Parameters: {model_info.parameter_count}")
        print(f"✓ Family: {model_info.family}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Retrieval Optimizer
    print("\n[Test 2] Testing Retrieval Optimizer...")
    config = calculate_optimal_retrieval(
        context_window=8192,
        system_prompt_tokens=150,
        user_query_tokens=50,
        chat_history_tokens=500,
        avg_chunk_tokens=200
    )
    print(f"✓ Optimal top_k: {config.top_k}")
    print(f"✓ Max chunk tokens: {config.max_chunk_tokens}")
    print(f"✓ Total budget: {config.total_chunk_budget} tokens")

    # Test 3: List all models
    print("\n[Test 3] Listing all local Ollama models...")
    try:
        models = await adapter.get_all_local_models()
        print(f"✓ Found {len(models)} local models:")
        for model in models[:5]:  # Show first 5
            print(f"  - {model.name}: {model.context_window} tokens")
    except Exception as e:
        print(f"✗ Error: {e}")

    await adapter.close()

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
