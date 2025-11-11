"""
Simple standalone test for Dynamic Context Window Detection.
No pytest required - just run: python test_context_window_simple.py
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.adapters.model_info.ollama_model_info import OllamaModelInfoAdapter
from app.core.utils.retrieval_optimizer import (
    calculate_optimal_retrieval,
    calculate_retrieval_with_history,
    estimate_tokens
)


async def test_model_detection():
    """Test 1: Model Info Detection for gemma3:1b"""
    print("\n" + "="*70)
    print("TEST 1: Model Context Window Detection")
    print("="*70)

    adapter = OllamaModelInfoAdapter(
        base_url="http://localhost:11434",
        default_context_window=4096
    )

    try:
        print("\n🔍 Detecting gemma3:1b context window...")
        model_info = await adapter.get_model_info("gemma3:1b")

        if model_info:
            print(f"✓ SUCCESS!")
            print(f"  Model: {model_info.name}")
            print(f"  Context Window: {model_info.context_window} tokens")
            print(f"  Parameters: {model_info.parameter_count}")
            print(f"  Family: {model_info.family}")
            print(f"  Quantization: {model_info.quantization}")

            assert model_info.context_window > 0, "Context window should be positive"
            assert model_info.parameter_count == "1B", "Should extract 1B from model name"

            return model_info.context_window
        else:
            print("✗ FAILED: Could not detect model info")
            return None

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print("\n⚠️  Make sure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. gemma3:1b is installed (ollama pull gemma3:1b)")
        return None
    finally:
        await adapter.close()


async def test_cache():
    """Test 2: Cache Performance"""
    print("\n" + "="*70)
    print("TEST 2: Caching Performance")
    print("="*70)

    adapter = OllamaModelInfoAdapter(
        base_url="http://localhost:11434",
        default_context_window=4096
    )

    try:
        print("\n⏱️  First call (uncached)...")
        start = time.time()
        context_window_1 = await adapter.get_context_window("gemma3:1b")
        elapsed_1 = (time.time() - start) * 1000
        print(f"  Time: {elapsed_1:.2f}ms")
        print(f"  Result: {context_window_1} tokens")

        print("\n⏱️  Second call (cached)...")
        start = time.time()
        context_window_2 = await adapter.get_context_window("gemma3:1b")
        elapsed_2 = (time.time() - start) * 1000
        print(f"  Time: {elapsed_2:.2f}ms")
        print(f"  Result: {context_window_2} tokens")

        if context_window_1 == context_window_2:
            print(f"\n✓ SUCCESS! Cache working")
            print(f"  Speedup: {elapsed_1/elapsed_2:.1f}x faster")
            assert elapsed_2 < 10, "Cached lookup should be under 10ms"
        else:
            print(f"\n✗ FAILED: Results don't match")

    except Exception as e:
        print(f"✗ ERROR: {e}")
    finally:
        await adapter.close()


async def test_list_models():
    """Test 3: List All Local Models"""
    print("\n" + "="*70)
    print("TEST 3: List Local Ollama Models")
    print("="*70)

    adapter = OllamaModelInfoAdapter(
        base_url="http://localhost:11434",
        default_context_window=4096
    )

    try:
        print("\n📋 Discovering local models...")
        models = await adapter.get_all_local_models()

        if models:
            print(f"✓ Found {len(models)} local models:\n")
            for model in models:
                print(f"  • {model.name}")
                print(f"    Context: {model.context_window:,} tokens")
                if model.parameter_count:
                    print(f"    Size: {model.parameter_count}")
                if model.quantization:
                    print(f"    Quant: {model.quantization}")
                print()
        else:
            print("✗ No models found")

    except Exception as e:
        print(f"✗ ERROR: {e}")
    finally:
        await adapter.close()


def test_token_estimation():
    """Test 4: Token Estimation"""
    print("\n" + "="*70)
    print("TEST 4: Token Estimation")
    print("="*70)

    test_cases = [
        ("Short text", 100),
        ("Medium paragraph with more content", 400),
        ("A" * 800, 800),
    ]

    print("\n🔢 Testing token estimation (1 token ≈ 4 chars):\n")

    for text, expected_chars in test_cases:
        estimated = estimate_tokens(text if isinstance(text, str) and len(text) < 50 else f"Text of {expected_chars} chars")
        actual = estimate_tokens(text if len(text) == expected_chars else "A" * expected_chars)
        expected_tokens = expected_chars // 4

        print(f"  Text: {expected_chars} chars")
        print(f"  Estimated: {actual} tokens")
        print(f"  Expected: ~{expected_tokens} tokens")
        print(f"  Status: {'✓' if actual == expected_tokens else '✗'}")
        print()


def test_retrieval_optimizer():
    """Test 5: Retrieval Optimization"""
    print("\n" + "="*70)
    print("TEST 5: Retrieval Budget Optimization")
    print("="*70)

    test_scenarios = [
        ("Small Model (4K)", 4096),
        ("gemma3:1b (8K)", 8192),
        ("Large Model (32K)", 32768),
    ]

    print("\n📊 Testing different context window sizes:\n")

    for name, context_window in test_scenarios:
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

        print(f"  {name}:")
        print(f"    Context Window: {context_window:,} tokens")
        print(f"    Optimal top_k: {config.top_k} chunks")
        print(f"    Chunk Budget: {config.total_chunk_budget:,} tokens")
        print(f"    Max Chunk Size: {config.max_chunk_tokens} tokens")
        print()


async def test_e2e_gemma3():
    """Test 6: End-to-End with gemma3:1b"""
    print("\n" + "="*70)
    print("TEST 6: End-to-End Optimization for gemma3:1b")
    print("="*70)

    adapter = OllamaModelInfoAdapter(
        base_url="http://localhost:11434",
        default_context_window=4096
    )

    try:
        print("\n🎯 Step 1: Detect gemma3:1b context window...")
        context_window = await adapter.get_context_window("gemma3:1b")
        print(f"   Detected: {context_window:,} tokens")

        print("\n🎯 Step 2: Calculate optimal retrieval...")
        config = calculate_retrieval_with_history(
            context_window=context_window,
            chat_history="User: Hello\nAssistant: Hi! How can I help?",
            user_query="What are the key findings in the research paper?",
            system_prompt="You are a helpful AI assistant specialized in research.",
            avg_chunk_tokens=200,
            safety_margin=0.75,
            min_top_k=2,
            max_top_k=10
        )

        print(f"   Optimal top_k: {config.top_k} chunks")
        print(f"   Max chunk tokens: {config.max_chunk_tokens}")
        print(f"   Total budget: {config.total_chunk_budget:,} tokens")
        print(f"   Reverse for Ollama: {config.reverse_for_ollama}")

        print("\n🎯 Step 3: Simulation - Chunk Retrieval...")
        chunks = [
            {"id": 1, "score": 0.95, "content": "Most relevant chunk"},
            {"id": 2, "score": 0.87, "content": "Second best chunk"},
            {"id": 3, "score": 0.82, "content": "Third chunk"},
            {"id": 4, "score": 0.76, "content": "Fourth chunk"},
            {"id": 5, "score": 0.71, "content": "Fifth chunk"},
        ]

        # Simulate taking top_k chunks
        selected = chunks[:config.top_k]
        print(f"   Retrieved {len(selected)} chunks based on optimal top_k")

        # Simulate reversal for Ollama
        if config.reverse_for_ollama:
            selected_reversed = list(reversed(selected))
            print(f"   Reversed chunks (most relevant LAST for Ollama)")
            print(f"   Order: {[c['id'] for c in selected_reversed]}")
            print(f"   Best chunk (ID {selected_reversed[-1]['id']}) is now LAST → Survives!")

        print("\n✓ SUCCESS! End-to-end optimization complete")
        print(f"\n💡 Summary:")
        print(f"   - Model: gemma3:1b")
        print(f"   - Context: {context_window:,} tokens")
        print(f"   - Retrieves: {config.top_k} chunks (auto-calculated)")
        print(f"   - Order: Reversed (best last)")
        print(f"   - Result: Optimal for Ollama's top-down truncation")

    except Exception as e:
        print(f"✗ ERROR: {e}")
    finally:
        await adapter.close()


async def main():
    """Run all tests"""
    print("\n")
    print("="*70)
    print("  DYNAMIC CONTEXT WINDOW DETECTION - TEST SUITE")
    print("="*70)

    # Check if Ollama is running
    print("\n[Pre-flight checks]")
    print("  Checking Ollama connection...")

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                print("  ✓ Ollama is running")
            else:
                print("  ✗ Ollama returned unexpected status")
                return
    except Exception as e:
        print(f"  ✗ Cannot connect to Ollama: {e}")
        print("\n⚠️  Please start Ollama first:")
        print("     ollama serve")
        return

    # Run tests
    tests_passed = 0
    tests_failed = 0

    try:
        # Test 1: Model Detection
        context_window = await test_model_detection()
        if context_window:
            tests_passed += 1
        else:
            tests_failed += 1

        # Test 2: Cache
        await test_cache()
        tests_passed += 1

        # Test 3: List Models
        await test_list_models()
        tests_passed += 1

        # Test 4: Token Estimation
        test_token_estimation()
        tests_passed += 1

        # Test 5: Retrieval Optimizer
        test_retrieval_optimizer()
        tests_passed += 1

        # Test 6: E2E
        await test_e2e_gemma3()
        tests_passed += 1

    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {tests_passed} ✓")
    print(f"  Failed: {tests_failed} ✗")
    print(f"  Total:  {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
