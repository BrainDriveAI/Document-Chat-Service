# Dynamic Context Window Detection

**Advanced RAG optimization that automatically adapts retrieval to any Ollama model's context window.**

## Overview

This feature solves a critical problem in local RAG deployments: **different LLM models have vastly different context windows**, and Ollama truncates tokens **from the top** when the context exceeds the limit. Without adaptation, the most relevant chunks (placed first) get stripped away.

### The Problem

```
Without optimization:
┌─────────────────────────────────────┐
│ System Prompt (150 tokens)          │ ← Stripped first
│ [Best Chunk 1] (200 tokens)         │ ← Lost!
│ [Good Chunk 2] (200 tokens)         │ ← Lost!
│ [Okay Chunk 3] (200 tokens)         │ ← Lost!
│ [Chunk 4] (200 tokens)              │
│ [Chunk 5] (200 tokens)              │
│ User Query (50 tokens)              │
│ Chat History (500 tokens)           │
└─────────────────────────────────────┘
Model with 2K context window → Loses best chunks!
```

### The Solution

```
With dynamic optimization:
┌─────────────────────────────────────┐
│ System Prompt (150 tokens)          │
│ [Chunk 3] (200 tokens)              │ ← Least relevant
│ [Chunk 2] (200 tokens)              │
│ [BEST CHUNK 1] (200 tokens)         │ ← Most relevant LAST
│ User Query (50 tokens)              │
│ Chat History (500 tokens)           │
└─────────────────────────────────────┘
Adapts chunk count + reverses order → Best chunks survive!
```

## How It Works

### 1. **Model Detection** (Automatic)

When a request includes a `model_name`, the system:
- Queries Ollama API: `POST /api/show {"name": "gemma3:1b"}`
- Extracts `num_ctx` parameter from response
- Caches result to avoid repeated API calls

```python
# Example Ollama response
{
  "parameters": [
    "num_ctx 8192",  # ← Extracted automatically
    "stop <|im_end|>"
  ],
  "model_info": {...}
}
```

### 2. **Token Budget Calculation**

```python
# Formula
available_tokens = (context_window × safety_margin)
                   - system_prompt_tokens
                   - user_query_tokens
                   - chat_history_tokens
                   - generation_buffer

optimal_top_k = clamp(
    available_tokens / avg_chunk_tokens,
    min=MIN_TOP_K,
    max=MAX_TOP_K
)
```

**Example for gemma3:1b (8192 tokens):**
```
Context window:     8192 tokens
Safety margin:      0.75 (use 75%)
──────────────────────────────────
Usable:            6144 tokens

Budget allocation:
- System prompt:   ~150 tokens
- User query:      ~50 tokens
- Chat history:    ~500 tokens
- Gen buffer:      ~512 tokens
──────────────────────────────────
Total overhead:    1212 tokens
Available:         4932 tokens

Chunk calculation:
4932 / 200 (avg) = 24.6
Clamped to MAX_TOP_K → 10 chunks
```

### 3. **Chunk Reversal** (Critical!)

**Why reverse?** Ollama strips tokens from the **top** of the context, not the bottom.

```python
# Before reversal (standard ranking)
chunks = [
    DocumentChunk(content="Most relevant!", score=0.95),
    DocumentChunk(content="Pretty good", score=0.85),
    DocumentChunk(content="Okay", score=0.75),
]

# After reversal (for Ollama)
chunks = [
    DocumentChunk(content="Okay", score=0.75),          # First to strip
    DocumentChunk(content="Pretty good", score=0.85),
    DocumentChunk(content="Most relevant!", score=0.95), # Last = survives!
]
```

## Architecture

### Components

```
┌──────────────────────────────────────────────────────────┐
│                    API Request                            │
│              (includes model_name)                        │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│           ContextRetrievalUseCase                         │
│  - Receives model_name parameter                         │
│  - Calls ModelInfoService.get_context_window()           │
│  - Calculates optimal top_k with RetrievalOptimizer      │
│  - Reverses chunks if REVERSE_CONTEXT_FOR_OLLAMA=true    │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│           OllamaModelInfoAdapter                          │
│  - Queries Ollama API (POST /api/show)                   │
│  - Extracts num_ctx from parameters                      │
│  - Caches results (TTL: 1 hour)                          │
│  - Fallback to DEFAULT_CONTEXT_WINDOW on failure         │
└──────────────────────────────────────────────────────────┘
```

### Port Interface

```python
# app/core/ports/model_info_service.py
class ModelInfoService(ABC):
    @abstractmethod
    async def get_context_window(self, model_name: str) -> int:
        """Get context window size in tokens"""
        pass
```

### Adapter Implementation

```python
# app/adapters/model_info/ollama_model_info.py
class OllamaModelInfoAdapter(ModelInfoService):
    async def get_context_window(self, model_name: str) -> int:
        # Check cache first
        if model_name in self._cache:
            return self._cache[model_name].context_window

        # Query Ollama API
        response = await self._client.post(
            f"{self.base_url}/api/show",
            json={"name": model_name}
        )

        # Extract num_ctx from parameters
        context_window = self._extract_context_window(response.json())

        # Cache and return
        self._cache[model_name] = ModelInfo(
            name=model_name,
            context_window=context_window or self.default_context_window
        )
        return context_window
```

## Configuration

### Environment Variables

```bash
# --- RAG Optimization ---

# Default context window when detection fails
DEFAULT_CONTEXT_WINDOW=4096

# Safety margin: use only 75% of context window
# Prevents edge cases and leaves room for generation
CONTEXT_SAFETY_MARGIN=0.75

# Reverse context for Ollama (HIGHLY RECOMMENDED)
# Places most relevant chunks LAST (Ollama strips from top)
REVERSE_CONTEXT_FOR_OLLAMA=true

# Retrieval bounds
MIN_TOP_K=2    # Minimum chunks to retrieve
MAX_TOP_K=10   # Maximum chunks to retrieve

# Expected average tokens per chunk
AVG_CHUNK_TOKENS=200
```

### Configuration in Code

```python
# app/config.py
class AppSettings(BaseSettings):
    DEFAULT_CONTEXT_WINDOW: int = Field(default=4096, ...)
    CONTEXT_SAFETY_MARGIN: float = Field(default=0.75, ...)
    REVERSE_CONTEXT_FOR_OLLAMA: bool = Field(default=True, ...)
    MIN_TOP_K: int = Field(default=2, ...)
    MAX_TOP_K: int = Field(default=10, ...)
    AVG_CHUNK_TOKENS: int = Field(default=200, ...)
```

## Usage

### Option 1: Chat/Search API (Automatic)

When using the chat or search endpoints, pass the `model_name`:

```bash
# Chat endpoint
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key findings?",
    "collection_id": "abc123",
    "model_name": "gemma3:1b",
    "session_id": "session-xyz"
  }'
```

The system automatically:
1. Detects gemma3:1b has 8192 token context window
2. Calculates optimal top_k (e.g., 8 chunks)
3. Reverses chunks (best last)
4. Returns optimized context

### Option 2: Programmatic Usage

```python
# In your use case
context_result = await context_retrieval_use_case.retrieve_context(
    query_text="What are the main topics?",
    collection_id="abc123",
    model_name="llama3.2:8b",  # ← Triggers optimization
    system_prompt="You are a helpful assistant.",  # For token budget
    chat_history=[{"user": "Hi", "assistant": "Hello!"}],
    top_k=10  # ← Overridden if optimization calculates lower
)

# Result includes optimization metadata
print(context_result.metadata["context_window_optimization"])
# {
#   "model_name": "llama3.2:8b",
#   "context_window": 8192,
#   "original_top_k": 10,
#   "calculated_top_k": 8,
#   "max_chunk_tokens": 616,
#   "optimization_enabled": true
# }
```

### Option 3: Evaluation Endpoint

Evaluation automatically uses the model specified in the request:

```bash
POST /evaluation/plugin/start-with-questions
{
  "collection_id": "test-collection",
  "questions": ["What is AI?", "Explain ML"],
  "llm_model": "qwen2.5:7b",  # ← Used for optimization
  "user_id": "abc123..."
}
```

## Model Context Window Reference

**Common Ollama models:**

| Model | Context Window | Optimal top_k (8GB RAM) |
|-------|----------------|-------------------------|
| gemma3:1b | 8,192 | 6-8 |
| llama3.2:3b | 8,192 | 6-8 |
| llama3.2:8b | 8,192 | 6-8 |
| phi3:mini | 4,096 | 3-4 |
| qwen2.5:0.5b | 32,768 | 10 (max) |
| qwen2.5:7b | 32,768 | 10 (max) |
| mistral:7b | 8,192 | 6-8 |
| mixtral:8x7b | 32,768 | 10 (max) |

*Note: These are detected automatically - no manual config needed!*

## Performance Impact

### Caching Strategy

- **First request** for a model: ~50-100ms (API call to Ollama)
- **Subsequent requests**: <1ms (cached in memory)
- **Cache TTL**: 1 hour (configurable via `cache_ttl_seconds`)

### Memory Usage

Minimal - caches only `ModelInfo` objects (~200 bytes each):
```python
@dataclass
class ModelInfo:
    name: str                    # e.g., "gemma3:1b"
    context_window: int          # e.g., 8192
    parameter_count: str | None  # e.g., "1B"
    quantization: str | None     # e.g., "Q4_K_M"
    family: str | None           # e.g., "gemma"
```

## Logging and Debugging

### Enable Debug Logs

```bash
# .env
DEBUG=true
```

### Log Output Examples

```
INFO: RAG optimization: gemma3:1b has 8192 token context window.
      Adjusted top_k from 10 to 7

DEBUG: Calculated optimal retrieval: context_window=8192,
       available_tokens=4932, top_k=7, max_chunk_tokens=704

DEBUG: Reversed 7 chunks (most relevant now LAST for Ollama)

WARNING: Failed to detect context window for unknown-model:
         Model not found. Using default top_k=10
```

### Metadata in Responses

Every retrieval includes optimization metadata:

```json
{
  "chunks": [...],
  "metadata": {
    "context_window_optimization": {
      "model_name": "gemma3:1b",
      "context_window": 8192,
      "original_top_k": 10,
      "calculated_top_k": 7,
      "max_chunk_tokens": 704,
      "optimization_enabled": true
    }
  }
}
```

## Troubleshooting

### Issue: Context window not detected

**Symptoms:**
```
WARNING: Failed to detect context window for gemma3:1b: Model not found
```

**Solutions:**
1. Verify model is pulled: `ollama list`
2. Check Ollama is running: `curl http://localhost:11434/api/tags`
3. Verify model name matches exactly (case-sensitive)
4. Check OLLAMA_LLM_BASE_URL in `.env`

### Issue: Too few chunks retrieved

**Symptoms:**
```
INFO: Adjusted top_k from 10 to 2
```

**Cause:** Small context window + large chat history

**Solutions:**
1. Reduce `CONTEXT_SAFETY_MARGIN` from 0.75 to 0.65
2. Limit chat history turns (`max_history_turns=2`)
3. Reduce `AVG_CHUNK_TOKENS` estimate
4. Use model with larger context window

### Issue: Chunks not reversed

**Check configuration:**
```bash
# .env
REVERSE_CONTEXT_FOR_OLLAMA=true  # Must be lowercase 'true'
```

**Verify in logs:**
```
DEBUG: Reversed 7 chunks (most relevant now LAST for Ollama)
```

## Advanced Features

### Manual Cache Refresh

```python
# Clear cache to force re-detection
await model_info_service.refresh_cache()
```

### Get All Local Models

```python
# List all Ollama models with detected context windows
models = await model_info_service.get_all_local_models()

for model in models:
    print(f"{model.name}: {model.context_window} tokens")
```

### Custom Context Window Override

```python
# For non-Ollama models (future feature)
from app.adapters.model_info.static_model_info import StaticModelInfoAdapter

model_info_service = StaticModelInfoAdapter(
    model_configs={
        "gpt-4": 128000,
        "claude-3-opus": 200000
    },
    default_context_window=4096
)
```

## Future Enhancements

### Phase 2 (Planned)

- [ ] Context compression for small models (use smaller LLM to summarize chunks)
- [ ] Chunk quality scoring (prioritize chunks with key information)
- [ ] Adaptive retrieval based on query complexity
- [ ] Support for OpenAI/Anthropic model detection

### Phase 3 (Future)

- [ ] Semantic chunking strategy with structure preservation
- [ ] Dynamic chunk size based on model (smaller chunks for small models)
- [ ] Multi-model ensemble retrieval
- [ ] Token usage tracking and budget alerts

## Testing

See `tests/test_dynamic_context_window.py` for:
- Unit tests for model info service
- Integration tests for retrieval optimization
- E2E tests with local Ollama models

**Run tests:**
```bash
poetry run pytest tests/test_dynamic_context_window.py -v
```

## Related Documentation

- [OLLAMA_PERFORMANCE_OPTIMIZATION.md](./OLLAMA_PERFORMANCE_OPTIMIZATION.md) - Hardware-specific tuning
- [README.md](../README.md) - General setup and usage
- [FOR-AI-CODING-AGENTS.md](../FOR-AI-CODING-AGENTS.md) - Architecture overview

## License

This feature is part of the Chat with Documents project.
MIT License - See [LICENSE](../LICENSE) for details.
