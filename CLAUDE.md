# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A sophisticated RAG (Retrieval-Augmented Generation) application built on **Clean Architecture** principles. The system enables conversational interaction with documents using advanced retrieval techniques including hybrid search, query transformation, intent classification, and contextual retrieval.

**Core Technologies:**
- FastAPI (async web framework)
- Python 3.12+ with Poetry dependency management
- Ollama (LLM and embeddings)
- ChromaDB (vector store)
- BM25 + Rank Fusion (hybrid search)
- SQLite with async SQLAlchemy (metadata persistence)
- Remote Document Processor API (spaCy Layout-based processing)

## Development Commands

### Local Development
```bash
# Setup
poetry install
poetry shell

# Run application with hot-reload
uvicorn app.main:app --reload

# Access
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
# - Metrics: http://localhost:8000/metrics
```

### Docker Development
```bash
# Build and run
docker-compose up --build

# Stop containers
docker-compose down

# Production deployment (with Nginx + Prometheus)
docker-compose -f docker-compose.prod.yml up --build -d
```

### Testing
Note: Test files are not yet implemented in this repository. When adding tests, use:
```bash
# Run tests with pytest
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_use_cases/test_document_processing.py

# Run with coverage
poetry run pytest --cov=app tests/
```

----------------
- In all interactions and commit messages, be extemely concise and sacrifice grammar for the sake of concision.

## PR Comments

<pr-comment-rule>
When I say to add a comment to a PR with a TODO on it, use the Github 'checkbox' markdown format to add the TODO. For instance:

<example>
- [ ] A description of the todo goes here
</example>
</pr-comment-rule>
-When tagging Claude in Github issues, use '@claude'

## Changesets
To add a changeset, write a new file to the `.changeset` directory

The file should be named `0000-your-change.md`. Decide yourself whether to make it a patch, minor, or major change.

The format of the file should be:
```md
---
"evalite": patch
---

Description of the change.
```

The description of the change should be user-facing, describing which features were added or bugs were fixed.

## Github
- Your primary method for interaction with Github should be the Github CLI (or git as a fallback).

## Git
-When creating branches, prefix them with a category like feat, fix, refactor, chore, docs, etc to promote clarity, consistency, and ease of understanding within a development team.
- Do not include this in git commit messages:
"""
🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
"""

## Plans
- At the end of each plan, give me a list of unresolved questions to answer, if any. Make the questions extemely concise. Sacrifice grammar for the sake of concision.

## Compounding Engineering

**Core principle:** Every session compounds knowledge for future developers/AI agents.

**ALWAYS before implementing, search:**
```bash
grep -ri "keyword" docs/decisions/  # Check if decision exists
grep -ri "keyword" docs/failures/   # Check if already failed
grep -ri "keyword" docs/data-quirks/  # Check for quirks
```

### Auto-Document When:

#### 1. Made Architectural Decision → Create ADR
**Triggers:**
- ✅ Chose between 2+ approaches (REST vs GraphQL, Redux vs Context)
- ✅ Selected library/framework (ChromaDB vs Qdrant)
- ✅ Changed Clean Architecture layer interaction
- ✅ Chose design pattern (Strategy, Factory, Repository)
- ✅ Selected external service provider (Ollama vs OpenAI)

**Action:**
```bash
cp docs/decisions/000-template.md docs/decisions/00X-decision-name.md
# Fill: Context, Problem, Decision, Consequences, Alternatives
```

#### 2. Discovered Data/System Quirk → Create Quirk Doc
**Triggers:**
- ✅ Non-obvious data behavior (retention policies, NULL patterns)
- ✅ API returns unexpected format
- ✅ Timezone/encoding inconsistencies
- ✅ Performance characteristics (batch size limits)
- ✅ Default values different than expected

**Action:**
```bash
touch docs/data-quirks/00X-quirk-name.md
# Document: Behavior, Why it matters, Detection, Correct patterns
```

#### 3. Hit Error/Mistake → Create Failure Log
**Triggers:**
- ✅ Incorrect assumption (wasted >1 hour)
- ✅ Approach failed (later fixed)
- ✅ Anti-pattern discovered
- ✅ Race condition, deadlock, OOM
- ✅ Integration failed unexpectedly

**Action:**
```bash
cp docs/failures/000-template.md docs/failures/00X-failure-name.md
# Document: What happened, Root cause, Impact, Lessons, Prevention
```

#### 4. External Integration → Create Integration Doc
**Triggers:**
- ✅ Connected new API/service
- ✅ Vendor-specific quirks
- ✅ Auth/error patterns established
- ✅ Rate limits encountered

**Action:**
```bash
touch docs/integrations/system-name.md
# Document: Purpose, Auth, Schema, Quirks, Error handling
```

### Templates
- **ADR:** `docs/decisions/000-template.md`
- **Failure:** `docs/failures/000-template.md`
- **Complete guide:** `docs/AI-AGENT-GUIDE.md`

### Before Committing
- [ ] Made architectural decision? → Created ADR
- [ ] Discovered quirk? → Documented it
- [ ] Hit error >1hr? → Created failure log
- [ ] Learned something non-obvious? → Documented it

## Architecture Overview

### Clean Architecture Layers

**1. Domain Layer** (`app/core/domain/`)
- **Pure business logic** with zero external dependencies
- **Entities**: `Document`, `Collection`, `Chat`, `DocumentChunk`, `StructuredElement`
- **Value Objects**: `Embedding`, query transformation/intent types
- **Exceptions**: Custom domain exceptions

**2. Ports Layer** (`app/core/ports/`)
- **Abstract interfaces** defined using `ABC` or `Protocol`
- Key ports: `DocumentProcessor`, `EmbeddingService`, `VectorStore`, `LLMService`, `BM25Service`, `RankFusionService`, repositories
- **Dependency Rule**: All dependencies point inward

**3. Use Cases Layer** (`app/core/use_cases/`)
- **Application-specific business rules**
- Orchestrates domain logic using port interfaces
- Key use cases:
  - `SimplifiedDocumentProcessingUseCase`: Document ingestion with contextual embeddings
  - `ContextRetrievalUseCase`: Main entry point for context retrieval (intent → query transformation → hybrid search)
  - `IntentClassificationUseCase`: Classifies user query intent
  - `QueryTransformationUseCase`: Multi-strategy query transformation
  - `CollectionSummaryUseCase`: Collection-level summarization with clustering
  - `ChatInteractionUseCase`: Chat session management
  - `SearchDocumentsUseCase`: Search orchestration

**4. Adapters Layer** (`app/adapters/`)
- **Concrete implementations** of port interfaces
- Grouped by technology:
  - `embedding/`: Ollama embedding service
  - `llm/`: Ollama LLM service
  - `vector_store/`: ChromaDB adapter
  - `search/`: BM25 and rank fusion adapters
  - `document_processing/`: Remote document processor + chunking strategies
  - `persistence/`: SQLite repositories
  - `clustering/`: Sklearn clustering for summarization

**5. API Layer** (`app/api/`)
- **FastAPI routes** and controllers
- `deps.py`: Dependency injection container (retrieves services from `app.state`)
- Routes: `/documents`, `/collections`, `/chat`, `/search`, `/health`

**6. Infrastructure Layer** (`app/infrastructure/`)
- **Cross-cutting concerns**: logging, metrics (Prometheus), startup handlers

### Dependency Injection Pattern

Services are instantiated **once during startup** in `app/main.py:on_startup()` and stored in `app.state`:

```python
# Startup (app/main.py)
app.state.embedding_service = OllamaEmbeddingService(...)
app.state.vector_store = ChromaVectorStoreAdapter(...)
app.state.document_repo = SQLiteDocumentRepository(...)

# Dependency providers (app/api/deps.py)
def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service

# Route usage
@router.get("/search")
async def search(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    ...
```

All dependency provider functions are in `app/api/deps.py`.

## Key Features & Implementation Details

### 1. Document Processing Pipeline
**File:** `app/core/use_cases/simple_document.py`

1. Upload document to local storage
2. Send to remote Document Processor API (spaCy Layout-based)
3. Receive structured elements (headings, paragraphs, lists, tables)
4. Apply chunking strategy (hierarchical, semantic, optimized)
5. **Contextual Retrieval** (if enabled): Generate context for each chunk using smaller LLM
6. Generate embeddings (batched, with concurrency controls)
7. Store in vector store (Chroma) and BM25 index
8. Persist metadata to SQLite

**Environment variables:**
- `ENABLE_CONTEXTUAL_RETRIEVAL=true` - Use smaller LLM to generate chunk context
- `OLLAMA_CONTEXTUAL_LLM_MODEL=llama3.2:3b` - Smaller model for context generation
- `EMBEDDING_BATCH_SIZE=8` - Batch size for embedding generation
- `EMBEDDING_CONCURRENCY=2` - Concurrent embedding requests

### 2. Advanced Retrieval System
**File:** `app/core/use_cases/context_retrieval.py`

**Main Entry Point:** `ContextRetrievalUseCase.retrieve_context()`

**Pipeline:**
1. **Intent Classification**: Determine query intent (chat, retrieval, summary, comparison, listing)
2. **Query Transformation**: Multi-strategy query expansion/rewriting
3. **Hybrid Search**: Vector (ChromaDB) + BM25 with Reciprocal Rank Fusion
4. **Result Fusion**: Deduplicate and rank results

**Query Transformation Methods** (from `app/core/domain/entities/query_transformation.py`):
- `STEP_BACK`: Generate broader conceptual query
- `SUB_QUERY`: Break into multiple sub-queries
- `CONTEXTUAL`: Incorporate chat history

**Intent Types** (from `app/core/domain/entities/search_intent.py`):
- `CHAT`: Casual conversation, no retrieval needed
- `RETRIEVAL`: Standard document search
- `COLLECTION_SUMMARY`: Summarize entire collection
- `COMPARISON`: Compare multiple concepts
- `LISTING`: Generate structured lists
- `CLARIFICATION`: Follow-up questions

### 3. Chunking Strategies
**Location:** `app/adapters/document_processing/chunking_strategies/`

**Available strategies:**
- `OptimizedHierarchicalChunkingStrategy`: Default - preserves document structure with headings
- `HierarchicalChunkingStrategy`: Standard hierarchical chunking
- `SemanticChunkingStrategy`: Groups semantically similar content
- `RecursiveChunkingStrategy`: Recursive text splitting
- `FixedSizeChunkingStrategy`: Simple fixed-size chunks

**Current default:** `OptimizedHierarchicalChunkingStrategy` (configured in `app/main.py`)

### 4. Hybrid Search Architecture
**Files:**
- `app/adapters/vector_store/chroma_store.py` - Vector similarity search
- `app/adapters/search/bm25_adapter.py` - Keyword search
- `app/adapters/search/rank_fusion_adapter.py` - Reciprocal Rank Fusion

**Search flow:**
1. Generate query embedding (for vector search)
2. Execute vector search (Chroma) and BM25 search **in parallel**
3. Apply Reciprocal Rank Fusion with `alpha` parameter (0.5 default)
4. Return top-k fused results

## Configuration

**All configuration is in:** `app/config.py` using Pydantic Settings

**Key environment variables:**

```bash
# LLM & Embedding Providers
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama

# Ollama Configuration
OLLAMA_LLM_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:8b
OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Contextual Retrieval (Advanced)
ENABLE_CONTEXTUAL_RETRIEVAL=true
OLLAMA_CONTEXTUAL_LLM_BASE_URL=http://localhost:11434
OLLAMA_CONTEXTUAL_LLM_MODEL=llama3.2:3b

# Embedding Optimization (tune for available RAM)
EMBEDDING_BATCH_SIZE=8          # For 16GB RAM
EMBEDDING_CONCURRENCY=2         # Parallel embedding requests
EMBEDDING_TIMEOUT=120
EMBEDDING_MAX_RETRIES=3

# Contextual Retrieval Optimization
CONTEXTUAL_BATCH_SIZE=3
CONTEXTUAL_CHUNK_TIMEOUT=60
CONTEXTUAL_DOC_MAX_LENGTH=4000

# Document Processing
DOCUMENT_PROCESSOR_API_URL=http://host.docker.internal:8080/documents/
DOCUMENT_PROCESSOR_API_KEY=your_api_key
DOCUMENT_PROCESSOR_TIMEOUT=300
DOCUMENT_PROCESSOR_MAX_RETRIES=3

# Storage
UPLOADS_DIR=./data/uploads
CHROMA_PERSIST_DIR=./data/vector_db
BM25_PERSIST_DIR=./data/bm25_index
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Telemetry (disabled by default)
CHROMA_TELEMETRY=0
SCARF_NO_ANALYTICS=true
DO_NOT_TRACK=true
ANONYMIZED_TELEMETRY=false
```

## Code Patterns & Conventions

### Type Safety
- **Always use type hints** for function signatures
- Use Pydantic models for request/response validation
- Use `Optional[T]` for nullable values
- Use `Protocol` or `ABC` for abstract interfaces

### Async Operations
- **Always use `async def`** for I/O operations (database, HTTP, file operations)
- Use `asyncio.gather()` for parallel async operations
- Handle timeouts for external API calls
- Close HTTP clients in shutdown handlers (`app/main.py:on_shutdown()`)

### Error Handling
- Create domain exceptions in `app/core/domain/exceptions.py`
- Use `HTTPException` in routes
- Log errors with context (document IDs, collection IDs)
- Use early returns to avoid deep nesting

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debug information")
logger.info("Initialized service X with config Y")
logger.warning("Recoverable issue occurred")
logger.error(f"Failed to process document {doc_id}: {e}")
```

### Route Pattern
```python
@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    collection_id: Optional[str] = None,
    use_case: DocumentManagementUseCase = Depends(get_document_management_use_case)
):
    """List documents, optionally filtered by collection_id."""
    try:
        documents = await use_case.list_documents(collection_id)
        return [to_response(d) for d in documents]
    except DomainException as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=400, detail=str(e))
```

### Use Case Pattern
```python
class MyUseCase:
    def __init__(
        self,
        repo: Repository,          # Port interface
        service: Service           # Port interface
    ):
        self._repo = repo
        self._service = service

    async def execute(self, input: Input) -> Output:
        # Orchestrate domain logic
        entity = await self._repo.find_by_id(input.id)
        result = await self._service.process(entity)
        return result
```

## Common Workflows

### Adding a New Feature
1. **Define domain entities** (if needed) in `app/core/domain/entities/`
2. **Create port interface** in `app/core/ports/`
3. **Implement use case** in `app/core/use_cases/`
4. **Implement adapter** in `app/adapters/`
5. **Add dependency provider** in `app/api/deps.py`
6. **Create route** in `app/api/routes/`
7. **Register router** in `app/main.py`

### Swapping Providers
To switch from Ollama to OpenAI:
1. Create new adapter implementing the port interface (e.g., `OpenAIEmbeddingService` implementing `EmbeddingService`)
2. Update `app/main.py:on_startup()` to instantiate new adapter
3. Update environment variables
4. **No changes needed** in use cases or routes (dependency injection handles it)

## Performance Considerations

- **Embedding generation** is memory-intensive: Tune `EMBEDDING_BATCH_SIZE` and `EMBEDDING_CONCURRENCY` based on available RAM
- **Contextual retrieval** doubles processing time: Disable via `ENABLE_CONTEXTUAL_RETRIEVAL=false` if not needed
- **Document processing** is CPU-bound: Use background tasks (already implemented in `app/api/routes/documents.py`)
- **Hybrid search** adds latency: Use `use_hybrid=false` for faster vector-only search
- Monitor performance via `/metrics` endpoint (Prometheus)

## Security Notes

- Use `SecretStr` for sensitive config values (`DOCUMENT_PROCESSOR_API_KEY`, etc.)
- Sanitize file paths to prevent directory traversal
- Validate all inputs using Pydantic models
- **Never commit `.env` file** (use `.env.example` as template)

## Debugging Tips

1. **Enable debug logging:** Set `DEBUG=true` in `.env`
2. **Check service health:** Startup includes Ollama connectivity tests (`app/main.py:verify_ollama_services()`)
3. **View logs:** Logs are written to `./data/logs/` (if volume mounted)
4. **Inspect embeddings:** Query ChromaDB directly via `app.state.vector_store`
5. **Test individual adapters:** All adapters implement port interfaces - easy to test in isolation

## Important Notes

- **No tests currently exist** - when adding tests, create in `tests/` with structure matching `app/`
- **Port interfaces** define the contract - always code against interfaces, not implementations
- **Services are singletons** - initialized once in startup, reused across requests
- **Background tasks** are used for long-running document processing
- **Graceful shutdown** handles cleanup of HTTP clients, thread pools, and DB connections
