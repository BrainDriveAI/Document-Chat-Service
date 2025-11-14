# RAG Optimization Guide for BrainDrive - Small Ollama Models

**Document Version:** 1.0
**Date:** 2025-01-10
**Target Models:** llama3.2:3b, qwen2.5:3b, phi3:mini, mistral:7b
**Hardware:** 8GB-16GB RAM average devices

---

## Table of Contents

1. [Context Window Allocation Strategy](#1-context-window-allocation-strategy)
2. [Optimal Chunking Strategy](#2-optimal-chunking-strategy)
3. [Enhanced Chunking with Structured Markdown](#3-enhanced-chunking-strategy-with-structured-markdown)
4. [Contextual Information Strategy](#4-contextual-information-strategy)
5. [Retrieval Strategy Optimization](#5-retrieval-strategy-optimization)
6. [System Prompt Optimization](#6-system-prompt-optimization-for-rag)
7. [Complete RAG Pipeline](#7-complete-rag-pipeline-for-small-models)
8. [Model-Specific Recommendations](#8-model-specific-recommendations)
9. [Critical Fixes for BrainDrive](#9-critical-fixes-needed-in-braindrive)
10. [Recommended Configuration](#10-recommended-rag-configuration-file)
11. [Summary & Action Items](#11-summary-your-optimal-rag-stack)

---

## 1. Context Window Allocation Strategy ⚠️ CRITICAL

This is the **most important** aspect to preserve your persona/system prompts.

### The Problem

With a 4096-token context window on a 3B model:
- **System Prompt (Persona):** 300-800 tokens
- **Retrieved Context (chunks):** 1500-2500 tokens
- **Conversation History:** 500-1000 tokens
- **User Query:** 50-200 tokens
- **XML Formatting Overhead:** ~50-100 tokens

**Result:** System prompt gets truncated! ❌

### Recommended Context Allocation

| Component | % of Context | Tokens (4k) | Tokens (8k) | Priority |
|-----------|--------------|-------------|-------------|----------|
| **System Prompt** | 15-20% | 600-800 | 1200-1600 | 🔴 **HIGHEST** |
| **Retrieved Chunks** | 40-50% | 1600-2000 | 3200-4000 | 🟡 Medium |
| **User Query** | 5-10% | 200-400 | 400-800 | 🟢 Low |
| **Conversation History** | 20-25% | 800-1000 | 1600-2000 | 🟡 Medium |
| **Response Buffer** | 10-15% | 400-600 | 800-1200 | 🟢 Low |

### Implementation

```python
def calculate_context_allocation(num_ctx: int = 4096):
    """Calculate optimal context allocation for RAG"""
    return {
        "system_prompt_max": int(num_ctx * 0.20),      # 20% - PROTECTED
        "retrieved_context_max": int(num_ctx * 0.45),  # 45%
        "conversation_history_max": int(num_ctx * 0.20), # 20%
        "user_query_max": int(num_ctx * 0.10),         # 10%
        "response_buffer": int(num_ctx * 0.05),        # 5% safety margin
    }

# Example for llama3.2:3b with num_ctx=4096
allocations = calculate_context_allocation(4096)
# {
#   "system_prompt_max": 819,
#   "retrieved_context_max": 1843,
#   "conversation_history_max": 819,
#   "user_query_max": 409,
#   "response_buffer": 204
# }
```

---

## 2. Optimal Chunking Strategy

Your current setup (1000 chars, 100 overlap) is good, but here's how to optimize it further:

### Recommended Chunking Parameters

| Model Size | Chunk Size (chars) | Overlap (chars) | Chunks to Retrieve | Total Context |
|------------|-------------------|-----------------|-------------------|---------------|
| **1B** | 400-600 | 50-100 | 2-3 | 800-1800 chars |
| **3B** ⭐ | 600-800 | 100-150 | 3-4 | 1800-3200 chars |
| **7B** | 800-1200 | 150-200 | 4-5 | 3200-6000 chars |

### Recommended for Your Use Case (3B-7B models)

```python
CHUNKING_CONFIG = {
    # For llama3.2:3b, qwen2.5:3b, phi3:mini
    "3b_models": {
        "chunk_size": 700,           # Characters (down from 1000)
        "chunk_overlap": 150,        # Characters (up from 100)
        "max_chunks_to_retrieve": 4, # Top-k results
        "similarity_threshold": 0.65, # Minimum similarity score
        "rerank_top_n": 8,           # Retrieve 8, rerank to 4
    },

    # For mistral:7b or similar
    "7b_models": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_chunks_to_retrieve": 5,
        "similarity_threshold": 0.60,
        "rerank_top_n": 10,
    }
}
```

### Why These Numbers?

**Shorter chunks (700 chars):**
- ~175 tokens per chunk (rough estimate: 4 chars/token)
- 4 chunks = ~700 tokens
- Fits comfortably in 45% allocation (1843 tokens)
- Leaves room for contextual summaries

**Higher overlap (150 chars):**
- Prevents important info from being split across chunks
- Critical for maintaining semantic continuity
- Essential when working with limited context

---

## 3. Enhanced Chunking Strategy with Structured Markdown

Since you're using **docling** (structured markdown), you can leverage document structure!

### Semantic Chunking Strategy

```python
def semantic_chunking_strategy(markdown_doc: str, chunk_size: int = 700):
    """
    Chunk markdown documents by semantic boundaries, not fixed size.
    Preserves document structure from docling output.
    """

    chunks = []

    # Priority 1: Keep headings with their content
    # Priority 2: Keep code blocks together
    # Priority 3: Keep lists together
    # Priority 4: Keep paragraphs together
    # Priority 5: Split at sentence boundaries

    # Pseudo-code for semantic chunking:
    for section in parse_markdown_sections(markdown_doc):
        if section.type == "heading":
            # Keep heading with next N chars of content
            chunk = create_chunk_with_heading(section, max_size=chunk_size)

        elif section.type == "code_block":
            # Keep code blocks together (up to chunk_size)
            if len(section.content) > chunk_size:
                # Split at logical points (function boundaries)
                chunks.extend(split_code_intelligently(section, chunk_size))
            else:
                chunks.append(section.content)

        elif section.type == "list":
            # Keep list items together
            chunk = create_chunk_with_list(section, max_size=chunk_size)

        else:
            # Regular text - split at sentence boundaries
            chunks.extend(split_at_sentences(section, chunk_size))

    return chunks
```

### Recommended Semantic Boundaries (in order)

1. **Section boundaries** (H1, H2 headers) - HIGHEST priority
2. **Code block boundaries** - Keep code together
3. **Paragraph boundaries** - Natural semantic units
4. **Sentence boundaries** - Last resort
5. **Word boundaries** - Absolute last resort

---

## 4. Contextual Information Strategy

Your approach of generating contextual info for each chunk is **excellent**! Here's how to optimize it:

### Chunk Metadata Structure

```python
class EnhancedChunk:
    content: str                    # The actual chunk text
    document_title: str             # Document name
    section_path: str               # e.g., "Chapter 3 > Security > Authentication"
    chunk_position: str             # e.g., "Chunk 5/20 (25% through doc)"
    chunk_summary: str              # 1-2 sentence summary (50-100 chars)
    surrounding_context: str        # Previous + next chunk titles
    semantic_tags: List[str]        # e.g., ["code", "API", "authentication"]
    embedding: List[float]          # Vector representation

    def get_contextual_prompt(self) -> str:
        """Generate contextual information for LLM"""
        return f"""Document: {self.document_title}
Section: {self.section_path}
Position: {self.chunk_position}
Context: {self.surrounding_context}
Summary: {self.chunk_summary}

Content:
{self.content}"""
```

### Critical: Contextual Info Token Budget

**Problem:** Contextual info adds tokens!

- **Chunk content:** ~175 tokens
- **Contextual metadata:** ~50-75 tokens
- **Total per chunk:** ~225-250 tokens

**Solution for 4 chunks:**
- **Without metadata:** 4 × 175 = 700 tokens
- **With metadata:** 4 × 250 = 1000 tokens
- **Extra cost:** +300 tokens

### Tiered Approach

```python
def format_chunks_for_small_models(chunks: List[EnhancedChunk],
                                   available_tokens: int) -> str:
    """
    Format chunks with contextual info, adapting to available tokens.
    """

    if available_tokens > 2000:  # 8k context models
        # Full context (verbose)
        return "\n\n---\n\n".join([
            chunk.get_contextual_prompt() for chunk in chunks
        ])

    elif available_tokens > 1200:  # 4k context models
        # Medium context (condensed)
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"""[Chunk {i}: {chunk.section_path}]
{chunk.content}""")
        return "\n\n".join(formatted)

    else:  # Very limited context
        # Minimal context (bare chunks)
        return "\n\n".join([chunk.content for chunk in chunks])
```

---

## 5. Retrieval Strategy Optimization

### Two-Stage Retrieval (Recommended)

```python
def two_stage_retrieval(query: str,
                       top_k_initial: int = 8,
                       top_k_final: int = 4) -> List[EnhancedChunk]:
    """
    Stage 1: Fast vector search (retrieve 8)
    Stage 2: Rerank by relevance (keep top 4)
    """

    # Stage 1: Vector similarity search
    query_embedding = embed_query(query)
    candidates = vector_db.similarity_search(
        query_embedding,
        top_k=top_k_initial
    )

    # Stage 2: Rerank using cross-encoder or BM25
    # This is done locally, no API calls
    reranked = rerank_by_semantic_similarity(
        query=query,
        candidates=candidates,
        model="lightweight_reranker"  # Small local model
    )

    return reranked[:top_k_final]
```

### Alternative: Query Expansion

```python
def expand_query_for_better_retrieval(user_query: str) -> List[str]:
    """
    Generate multiple query variations to improve retrieval.
    Uses small local model (llama3.2:1b) for speed.
    """

    prompt = f"""Generate 2 alternative phrasings of this question:
Original: {user_query}

Alternative 1:
Alternative 2:"""

    # Use smallest model for query expansion (fast!)
    expanded = call_ollama("llama3.2:1b", prompt, max_tokens=100)

    # Now search with all 3 queries
    all_results = []
    for query_variant in [user_query] + parse_alternatives(expanded):
        results = vector_search(query_variant, top_k=3)
        all_results.extend(results)

    # Deduplicate and return top N
    return deduplicate_by_similarity(all_results)[:4]
```

---

## 6. System Prompt Optimization for RAG

Your persona system prompts need to be **concise yet effective** for small models.

### Optimized System Prompt Structure

```python
def create_rag_optimized_system_prompt(persona_instructions: str,
                                       max_tokens: int = 600) -> str:
    """
    Create a concise system prompt that preserves key instructions.
    """

    # Template optimized for RAG
    template = f"""{persona_instructions}

DOCUMENT CONTEXT:
You will receive relevant excerpts from documents to help answer questions.
Each excerpt is marked with [Chunk N: Section Name].

INSTRUCTIONS:
1. Base your answer primarily on the provided context
2. If context is insufficient, acknowledge this honestly
3. Cite specific sections when referencing information
4. Maintain your persona while being factually accurate

Answer the user's question using the context below:"""

    # Ensure it fits within token budget
    if estimate_tokens(template) > max_tokens:
        # Truncate persona instructions if needed
        truncated_persona = truncate_to_tokens(
            persona_instructions,
            max_tokens=max_tokens - 150  # Reserve 150 for template
        )
        template = template.replace(persona_instructions, truncated_persona)

    return template
```

### Persona Instruction Guidelines

| Persona Length | Model Size | Recommendation |
|----------------|------------|----------------|
| **Long (500+ words)** | 3B | ❌ Compress to 200 words max |
| **Medium (200-300 words)** | 3B | ✅ Optimal |
| **Short (<150 words)** | 3B | ✅ Perfect |
| **Any length** | 7B | ✅ Up to 400 words OK |

---

## 7. Complete RAG Pipeline for Small Models

### Pipeline Architecture

```python
async def rag_pipeline_for_small_models(
    user_query: str,
    conversation_history: List[Dict],
    persona_settings: Dict,
    model_config: Dict
) -> str:
    """
    Optimized RAG pipeline for small local models.
    """

    # 1. Calculate context budget
    num_ctx = persona_settings.get("context_window", 4096)
    allocation = calculate_context_allocation(num_ctx)

    # 2. Prepare system prompt (fits in 20% budget)
    system_prompt = create_rag_optimized_system_prompt(
        persona_settings["system_prompt"],
        max_tokens=allocation["system_prompt_max"]
    )

    # 3. Retrieve relevant chunks (two-stage retrieval)
    relevant_chunks = await two_stage_retrieval(
        query=user_query,
        top_k_initial=8,
        top_k_final=4  # Fits in 45% budget
    )

    # 4. Format context based on available tokens
    context_text = format_chunks_for_small_models(
        relevant_chunks,
        available_tokens=allocation["retrieved_context_max"]
    )

    # 5. Trim conversation history to fit budget
    trimmed_history = trim_conversation_history(
        conversation_history,
        max_tokens=allocation["conversation_history_max"]
    )

    # 6. Build final prompt
    messages = [
        {"role": "system", "content": system_prompt},
        *trimmed_history,
        {"role": "user", "content": f"""Context:
{context_text}

Question: {user_query}"""}
    ]

    # 7. Verify total token count
    total_tokens = estimate_messages_tokens(messages)
    if total_tokens > num_ctx * 0.95:  # 95% safety margin
        # Emergency: Reduce context
        context_text = reduce_context_aggressively(
            context_text,
            target_tokens=allocation["retrieved_context_max"] - 200
        )

    # 8. Call LLM with proper options
    response = await call_ollama_chat(
        model=model_config["model"],
        messages=messages,
        options={
            "num_ctx": num_ctx,  # ← CRITICAL!
            "temperature": persona_settings.get("temperature", 0.7),
            "top_p": persona_settings.get("top_p", 0.9),
        }
    )

    return response
```

---

## 8. Model-Specific Recommendations

### For llama3.2:3b (Primary Target)

```python
LLAMA32_3B_CONFIG = {
    "model": "llama3.2:3b",
    "context_window": 8192,  # Native support
    "chunk_size": 700,
    "chunk_overlap": 150,
    "max_chunks": 4,
    "system_prompt_max_words": 200,
    "rerank_enabled": True,

    # Ollama options
    "ollama_options": {
        "num_ctx": 8192,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
    }
}
```

**Strengths:** Fast, good reasoning, handles 8k context well
**Weaknesses:** May struggle with very technical/specialized content
**Best for:** General Q&A, documentation, summarization

### For qwen2.5:3b (Alternative)

```python
QWEN25_3B_CONFIG = {
    "model": "qwen2.5:3b",
    "context_window": 8192,
    "chunk_size": 800,  # Slightly larger - better at dense content
    "chunk_overlap": 150,
    "max_chunks": 5,    # Can handle more
    "system_prompt_max_words": 250,
    "rerank_enabled": True,

    "ollama_options": {
        "num_ctx": 8192,
        "temperature": 0.7,
        "top_p": 0.9,
    }
}
```

**Strengths:** Better reasoning, multilingual, technical content
**Weaknesses:** Slightly slower than llama3.2:3b
**Best for:** Technical docs, code, complex queries

### For phi3:mini (Code-Focused)

```python
PHI3_MINI_CONFIG = {
    "model": "phi3:mini",
    "context_window": 4096,  # Limited
    "chunk_size": 600,       # Smaller chunks
    "chunk_overlap": 100,
    "max_chunks": 3,         # Fewer chunks
    "system_prompt_max_words": 150,
    "rerank_enabled": True,

    "ollama_options": {
        "num_ctx": 4096,
        "temperature": 0.7,
    }
}
```

**Strengths:** Excellent code understanding, fast
**Weaknesses:** Smaller context window
**Best for:** Technical documentation, API docs, code examples

---

## 9. Critical Fixes Needed in BrainDrive

### Fix 1: Pass Options Correctly to Ollama

**File:** `backend/app/ai_providers/ollama.py`

**Current Issue:** Parameters are passed at the top level, but Ollama expects them in an `options` object.

**❌ Current (Wrong):**
```python
payload = {
    "model": model,
    "prompt": prompt,
    "context_window": 4000,  # Ollama ignores this!
    "temperature": 0.7
}
```

**✅ Fixed (Correct):**
```python
async def _call_ollama_api(self, prompt: str, model: str, params: Dict[str, Any], is_streaming: bool = False) -> Dict[str, Any]:
    # Extract options and map context_window
    options = {}

    for key, value in params.items():
        if key == "context_window":
            options["num_ctx"] = value  # Map to Ollama's parameter
        elif key in ["temperature", "top_p", "top_k", "repeat_penalty",
                     "frequency_penalty", "presence_penalty"]:
            options[key] = value

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options  # ← Put ALL params in options!
    }

    # ... rest of the code
```

**Apply the same fix to `_stream_ollama_api` method!**

### Fix 2: Implement Context Allocation in Chat Endpoint

**File:** `backend/app/api/v1/endpoints/ai_providers.py` (in chat_completion function)

**Add after line 1043 (where enhanced_params is created):**

```python
# Calculate context allocation to prevent system prompt truncation
num_ctx = enhanced_params.get("context_window", 4096)
allocation = {
    "system_prompt_max": int(num_ctx * 0.20),
    "retrieved_context_max": int(num_ctx * 0.45),
    "conversation_history_max": int(num_ctx * 0.20),
    "user_query_max": int(num_ctx * 0.10),
}

# Estimate token counts
system_tokens = estimate_tokens(request.persona_system_prompt) if request.persona_system_prompt else 0
history_tokens = estimate_tokens(str(combined_messages))
current_msg_tokens = estimate_tokens(str(request.messages))

# Warn if system prompt is too large
if system_tokens > allocation["system_prompt_max"]:
    logger.warning(f"System prompt ({system_tokens} tokens) exceeds recommended size ({allocation['system_prompt_max']} tokens). May be truncated by Ollama.")

# TODO: Implement intelligent truncation if needed
```

---

## 10. Recommended RAG Configuration File

Create a configuration file for different model tiers:

```python
# rag_configs.py

RAG_CONFIGS = {
    "llama3.2:3b": {
        "embedding_model": "mxbai-embed-large",
        "chunk_size": 700,
        "chunk_overlap": 150,
        "max_chunks_to_retrieve": 4,
        "rerank_top_n": 8,
        "context_window": 8192,
        "system_prompt_max_tokens": 800,
        "ollama_options": {
            "num_ctx": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }
    },

    "qwen2.5:3b": {
        "embedding_model": "mxbai-embed-large",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "max_chunks_to_retrieve": 5,
        "rerank_top_n": 10,
        "context_window": 8192,
        "system_prompt_max_tokens": 1000,
        "ollama_options": {
            "num_ctx": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    },

    "phi3:mini": {
        "embedding_model": "mxbai-embed-large",
        "chunk_size": 600,
        "chunk_overlap": 100,
        "max_chunks_to_retrieve": 3,
        "rerank_top_n": 6,
        "context_window": 4096,
        "system_prompt_max_tokens": 600,
        "ollama_options": {
            "num_ctx": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    },

    "mistral:7b": {
        "embedding_model": "mxbai-embed-large",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_chunks_to_retrieve": 6,
        "rerank_top_n": 12,
        "context_window": 16384,
        "system_prompt_max_tokens": 1200,
        "ollama_options": {
            "num_ctx": 16384,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
}
```

---

## 11. Summary: Your Optimal RAG Stack

### For 16GB RAM Systems (Recommended Baseline)

**Embedding:**
```bash
ollama pull mxbai-embed-large  # 670MB - You're already using this ✅
```

**LLM Generation:**
```bash
ollama pull qwen2.5:3b         # 2.5GB - Best overall
# OR
ollama pull llama3.2:3b        # 2.0GB - Faster alternative
```

### Configuration

```yaml
chunking:
  chunk_size: 700           # Down from your 1000
  chunk_overlap: 150        # Up from your 100
  semantic_boundaries: true # Use docling structure

retrieval:
  initial_retrieval: 8      # Two-stage approach
  final_chunks: 4           # After reranking
  similarity_threshold: 0.65

context_allocation:
  system_prompt: 20%        # Protected!
  retrieved_context: 45%
  conversation_history: 20%
  user_query: 10%
  buffer: 5%

model_settings:
  num_ctx: 8192            # Use model's native context
  temperature: 0.7
  top_p: 0.9
```

### Expected Performance

- ✅ **System prompts preserved** (never truncated)
- ✅ **4-5 relevant chunks** in context (enough for most queries)
- ✅ **Fast responses** (2-4 seconds)
- ✅ **Runs on 16GB RAM** comfortably
- ✅ **Privacy-first** (100% local)

---

## Action Items

### Immediate Fixes (Critical)

1. **Implement the `num_ctx` fix** in `ollama.py`
   - Move all params to `options` object
   - Map `context_window` → `num_ctx`

2. **Update RAG chunking configuration**
   - Reduce chunk size from 1000 → 700 characters
   - Increase overlap from 100 → 150 characters

3. **Set proper context window**
   - Set `num_ctx: 8192` for llama3.2:3b and qwen2.5:3b
   - Set `num_ctx: 4096` for phi3:mini

### Medium Priority (Recommended)

4. **Implement two-stage retrieval**
   - Retrieve 8 candidates initially
   - Rerank to keep top 4

5. **Add context allocation logic**
   - Prevent system prompt truncation
   - Implement token counting and warnings

6. **Optimize contextual metadata**
   - Use tiered approach based on available tokens
   - Balance between context richness and token budget

### Future Enhancements

7. **Semantic chunking**
   - Leverage docling's structured markdown
   - Chunk at semantic boundaries (headings, code blocks, etc.)

8. **Query expansion**
   - Use llama3.2:1b for fast query reformulation
   - Improve retrieval accuracy

9. **Performance monitoring**
   - Track system prompt preservation rate
   - Monitor chunk relevance scores
   - Measure response quality

---

## References

- **Ollama API Documentation:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **Default Context Window:** 2048 tokens (Ollama default)
- **Recommended Context Windows:**
  - llama3.2:3b → 8192 tokens
  - qwen2.5:3b → 8192 tokens
  - phi3:mini → 4096 tokens
  - mistral:7b → 16384 tokens

---

## Notes

- **Token Estimation:** Rough estimate is 4 characters = 1 token
- **Context Window Must Be Set:** Ollama reverts to 2048 default if not specified
- **System Prompt Protection:** Always allocate 20% of context for system prompt
- **Embedding Model:** mxbai-embed-large (670MB) is excellent for your use case
- **Privacy First:** All processing happens locally, no external API calls

---

**End of Document**
