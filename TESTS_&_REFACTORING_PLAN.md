# Tests & Refactoring Implementation Plan

**Target**: Comprehensive test coverage (80%+), critical refactoring, documentation improvements

**Branch**: `feat/comprehensive-tests-refactoring`

---

## Phase 1: Test Infrastructure Setup

- [x] Add test dependencies to `pyproject.toml` (pytest-asyncio, pytest-cov, pytest-mock)
- [x] Create `pytest.ini` with async support, coverage config
- [x] Create `tests/conftest.py` with shared fixtures
  - [x] Mock repositories (document, collection, chat, evaluation)
  - [x] Mock services (embedding, LLM, vector store, BM25)
  - [x] FastAPI TestClient fixture
  - [x] Async SQLite in-memory database fixture
- [ ] Run `poetry install` to install test dependencies
- [x] Create `.github/workflows/tests.yml` (GitHub Actions CI)

---

## Phase 2: Domain Layer Tests (`tests/unit/domain/`)

- [ ] `test_entities.py` - Test Document, DocumentChunk, Collection, ChatSession, ChatMessage
  - [ ] Test factory methods
  - [ ] Test validation logic
  - [ ] Test entity state transitions
- [ ] `test_value_objects.py` - Test Embedding, SearchQuery, QueryTransformationMethod
  - [ ] Test `__post_init__` validation
  - [ ] Test immutability
- [ ] `test_evaluation_entities.py` - Test EvaluationRun, EvaluationResult
  - [ ] Test progress calculations
  - [ ] Test state management
- [ ] `test_exceptions.py` - Test custom domain exceptions

---

## Phase 3: Use Cases Tests (`tests/unit/use_cases/`)

### Core Document Processing
- [ ] `test_simplified_document_processing.py` (SimplifiedDocumentProcessingUseCase)
  - [ ] Test document processing pipeline (happy path)
  - [ ] Test contextual retrieval enabled/disabled
  - [ ] Test embedding generation with batching
  - [ ] Test error handling (document processor failure, embedding failure)
  - [ ] Test cleanup on failure (document status rollback)

### Retrieval System
- [ ] `test_context_retrieval.py` (ContextRetrievalUseCase)
  - [ ] Test intent classification → query transformation → hybrid search pipeline
  - [ ] Test each intent type (CHAT, RETRIEVAL, COLLECTION_SUMMARY, etc.)
  - [ ] Test query transformation methods (STEP_BACK, SUB_QUERY, CONTEXTUAL)
  - [ ] Test hybrid search with rank fusion
  - [ ] Test dynamic context window detection
  - [ ] Test with/without chat history

- [ ] `test_intent_classification.py` (IntentClassificationUseCase)
  - [ ] Test regex-based classification (chat, summary, comparison, listing)
  - [ ] Test LLM fallback classification
  - [ ] Test clarification detection

- [ ] `test_query_transformation.py` (QueryTransformationUseCase)
  - [ ] Test step-back query generation
  - [ ] Test sub-query decomposition
  - [ ] Test contextual query with chat history
  - [ ] Test HyDE generation

### Collection & Document Management
- [ ] `test_collection_management.py` (CollectionManagementUseCase)
  - [ ] Test create, read, update, delete operations
  - [ ] Test list collections
  - [ ] Test error handling (duplicate names, not found)

- [ ] `test_document_management.py` (DocumentManagementUseCase)
  - [ ] Test list documents (with/without collection filter)
  - [ ] Test delete document (cascade to chunks)
  - [ ] Test error handling

- [ ] `test_collection_summary.py` (CollectionSummaryUseCase)
  - [ ] Test clustering with sklearn
  - [ ] Test summary generation
  - [ ] Test with different sample sizes

### Chat
- [ ] `test_chat_interaction.py` (ChatInteractionUseCase)
  - [ ] Test create/delete chat sessions
  - [ ] Test add/retrieve messages
  - [ ] Test message history limits

### Search
- [ ] `test_search_documents.py` (SearchDocumentsUseCase)
  - [ ] Test search with filters
  - [ ] Test hybrid vs vector-only search

### Evaluation (8 use cases)
- [ ] `test_evaluation_management.py` (Start, Status, List, Results use cases)
  - [ ] Test evaluation lifecycle (start → submit → complete)
  - [ ] Test concurrent submission handling (race condition fix)
  - [ ] Test list evaluations with filters
  - [ ] Test results retrieval

- [ ] `test_evaluation_plugins.py` (Plugin Evaluation use cases)
  - [ ] Test plugin-based evaluation flow
  - [ ] Test different evaluation types (contextual_relevancy, faithfulness, etc.)

---

## Phase 4: Adapters Integration Tests (`tests/integration/adapters/`)

### Embedding & LLM Services
- [ ] `test_ollama_embedding.py` (OllamaEmbeddingService)
  - [ ] Test embedding generation (mock HTTP requests)
  - [ ] Test batch processing
  - [ ] Test retry logic with exponential backoff
  - [ ] Test timeout handling
  - [ ] Test error scenarios (service unavailable)

- [ ] `test_ollama_llm.py` (OllamaLLMService)
  - [ ] Test response generation (mock streaming)
  - [ ] Test multi-query generation
  - [ ] Test chat with history
  - [ ] Test retry logic

### Vector Store & Search
- [ ] `test_chroma_store.py` (ChromaVectorStoreAdapter)
  - [ ] Test add_chunks() with batching (use in-memory Chroma)
  - [ ] Test search() with filters (collection_id, document_id)
  - [ ] Test delete_chunks() and delete_by_document()
  - [ ] Test metadata sanitization
  - [ ] Test persistence (if using temp dir)

- [ ] `test_bm25_adapter.py` (BM25Adapter)
  - [ ] Test index_chunks()
  - [ ] Test search() keyword matching
  - [ ] Test serialization/deserialization
  - [ ] Test persistence to disk

- [ ] `test_rank_fusion.py` (RankFusionAdapter)
  - [ ] Test reciprocal rank fusion algorithm
  - [ ] Test with different alpha values
  - [ ] Test deduplication logic

### Persistence (SQLite Repositories)
- [ ] `test_sqlite_repository.py` (All repository classes)
  - [ ] Test CollectionRepository CRUD (use in-memory SQLite)
  - [ ] Test DocumentRepository CRUD with relationships
  - [ ] Test ChatRepository (sessions + messages)
  - [ ] Test EvaluationRepository (runs + results)
  - [ ] Test async operations
  - [ ] Test transaction handling

### Document Processing
- [ ] `test_remote_document_processor.py` (RemoteDocumentProcessorAdapter)
  - [ ] Test document upload (mock HTTP API)
  - [ ] Test structured element extraction
  - [ ] Test timeout/retry logic
  - [ ] Test error handling

- [ ] `test_chunking_strategies.py` (All 5 strategies)
  - [ ] Test OptimizedHierarchicalChunkingStrategy
  - [ ] Test HierarchicalChunkingStrategy
  - [ ] Test SemanticChunkingStrategy
  - [ ] Test RecursiveChunkingStrategy
  - [ ] Test FixedSizeChunkingStrategy
  - [ ] Test chunk size limits, overlap, metadata preservation

### Other Adapters
- [ ] `test_sklearn_clustering.py` (SklearnClusteringService)
  - [ ] Test clustering with different algorithms
  - [ ] Test optimal cluster detection

- [ ] `test_ollama_model_info.py` (OllamaModelInfoService)
  - [ ] Test model info retrieval
  - [ ] Test context window detection

- [ ] `test_langchain_evaluation.py` (LangChainEvaluationService)
  - [ ] Test evaluation execution (mock LangChain)

- [ ] `test_langgraph_orchestrator.py` (LangGraphOrchestrator)
  - [ ] Test chat orchestration flow

---

## Phase 5: API Endpoint Tests (`tests/integration/api/`)

- [ ] `test_documents_routes.py`
  - [ ] Test POST /documents (upload)
  - [ ] Test GET /documents (list, with/without collection filter)
  - [ ] Test GET /documents/{id}
  - [ ] Test DELETE /documents/{id}
  - [ ] Test error handling (404, validation errors)

- [ ] `test_collections_routes.py`
  - [ ] Test POST /collections (create)
  - [ ] Test GET /collections (list)
  - [ ] Test GET /collections/{id}
  - [ ] Test PUT /collections/{id}
  - [ ] Test DELETE /collections/{id}

- [ ] `test_chat_routes.py`
  - [ ] Test POST /chat/sessions
  - [ ] Test GET /chat/sessions
  - [ ] Test POST /chat/sessions/{id}/messages
  - [ ] Test GET /chat/sessions/{id}/messages
  - [ ] Test DELETE /chat/sessions/{id}

- [ ] `test_search_routes.py`
  - [ ] Test POST /search
  - [ ] Test POST /search_u (unified search)
  - [ ] Test filter parameters

- [ ] `test_evaluation_routes.py` (largest route file)
  - [ ] Test POST /evaluation/start
  - [ ] Test POST /evaluation/submit
  - [ ] Test GET /evaluation/status/{id}
  - [ ] Test GET /evaluation/list
  - [ ] Test GET /evaluation/results/{id}
  - [ ] Test concurrent submissions

- [ ] `test_health_routes.py`
  - [ ] Test GET /health
  - [ ] Test GET /metrics (Prometheus)

---

## Phase 6: Critical Refactoring

### 6.1 Extract Long Methods
- [ ] **SimplifiedDocumentProcessingUseCase** (`app/core/use_cases/simple_document.py`)
  - [ ] Extract `_process_document_structure()` from `process_document()` (lines 45-70)
  - [ ] Extract `_generate_embeddings()` (lines 89-97)
  - [ ] Extract `_index_chunks()` (lines 105-134)
  - [ ] Update tests to reflect new methods

- [ ] **ContextRetrievalUseCase** (`app/core/use_cases/context_retrieval.py`)
  - [ ] Extract `_classify_and_route()` from `retrieve_context()` (lines 58-90)
  - [ ] Extract `_transform_queries()` (lines 95-130)
  - [ ] Extract `_execute_hybrid_search()` (lines 140-180)
  - [ ] Update tests to reflect new methods

### 6.2 Eliminate Code Duplication

- [ ] **Shared Retry Logic** - Create `app/core/utils/retry_helper.py`
  - [ ] Extract `@retry_with_exponential_backoff` decorator
  - [ ] Refactor `OllamaEmbeddingService._make_request_with_retry()`
  - [ ] Refactor `OllamaLLMService` retry logic
  - [ ] Add tests for retry helper

- [ ] **Error Handling Pattern** - Create `app/core/utils/error_handler.py`
  - [ ] Create `@handle_domain_errors` decorator for common try-catch patterns
  - [ ] Apply to use cases (simple_document.py, context_retrieval.py)
  - [ ] Standardize logging format

- [ ] **Repository Base Class** - Create `app/adapters/persistence/base_repository.py`
  - [ ] Extract common CRUD methods (save, find_by_id, delete, find_all)
  - [ ] Extract ORM mapping utilities
  - [ ] Refactor CollectionRepository, DocumentRepository, ChatRepository to inherit
  - [ ] Update tests

- [ ] **Intent Classification Helpers** - Create `app/core/utils/intent_helpers.py`
  - [ ] Extract regex pattern matching from `IntentClassificationUseCase`
  - [ ] Share between intent_classification.py and related use cases
  - [ ] Add tests

### 6.3 Fix Type Hints

- [ ] **Evaluation Routes** (`app/api/routes/evaluation.py`)
  - [ ] Create Pydantic models for all `Dict[str, Any]` return types
  - [ ] Add models: `EvaluationStatusResponse`, `EvaluationListResponse`, etc.

- [ ] **Repository Return Types** (`app/adapters/persistence/sqlite_repository.py`)
  - [ ] Add explicit return type hints to all methods
  - [ ] Ensure consistency with docstrings

### 6.4 Remove Dead Code

- [ ] **Documents Route** (`app/api/routes/documents.py`)
  - [ ] Remove commented code (lines 74-83) - old `determine_document_type()`

---

## Phase 7: Documentation

### 7.1 Add Missing Docstrings

- [ ] **SimplifiedDocumentProcessingUseCase** (`app/core/use_cases/simple_document.py`)
  - [ ] Add docstrings to `_add_contextual_information_batch()`
  - [ ] Document all private helper methods
  - [ ] Add parameters/returns documentation

- [ ] **ContextRetrievalUseCase** (`app/core/use_cases/context_retrieval.py`)
  - [ ] Document all 13 parameters of `retrieve_context()`
  - [ ] Clarify `model_name`, `system_prompt`, `max_history_turns`
  - [ ] Document helper methods

- [ ] **ChromaVectorStoreAdapter** (`app/adapters/vector_store/chroma_store.py`)
  - [ ] Document `search()` parameters (filters, top_k)
  - [ ] Document metadata sanitization logic
  - [ ] Document filter building

- [ ] **SQLiteRepository** (`app/adapters/persistence/sqlite_repository.py`)
  - [ ] Add docstrings to all private methods (`_to_domain_model()`, etc.)
  - [ ] Document ORM mapping strategy

- [ ] **Evaluation Routes** (`app/api/routes/evaluation.py`)
  - [ ] Add comprehensive docstrings to all endpoints
  - [ ] Document state machine flow
  - [ ] Document concurrent submission handling

- [ ] **All Other Public Methods** (50+ methods across codebase)
  - [ ] Add docstrings with parameters, returns, exceptions

### 7.2 Architecture Guides (Inline Documentation)

- [ ] **Contextual Retrieval Pipeline**
  - [ ] Add module-level docstring to `simple_document.py` explaining:
    - When to enable contextual retrieval
    - Performance impact (2x processing time)
    - Accuracy improvements
    - Configuration (`ENABLE_CONTEXTUAL_RETRIEVAL`, `OLLAMA_CONTEXTUAL_LLM_MODEL`)

- [ ] **Hybrid Search Algorithm**
  - [ ] Add module-level docstring to `context_retrieval.py` explaining:
    - Reciprocal rank fusion algorithm
    - `alpha` parameter (0.5 default) - vector vs BM25 weighting
    - When to use hybrid vs vector-only search

- [ ] **Evaluation Framework**
  - [ ] Add module-level docstring to `evaluation.py` route explaining:
    - State machine flow (start → submit → complete)
    - Plugin evaluation protocol
    - Concurrency handling (race condition fixes from recent commits)

- [ ] **Dynamic Context Window Detection**
  - [ ] Add inline documentation in `context_retrieval.py` explaining:
    - Model detection integration points
    - `model_name` and `system_prompt` usage
    - Fallback behavior

### 7.3 Inline Comments for Complex Algorithms

- [ ] **Rank Fusion Algorithm** (`app/adapters/search/rank_fusion_adapter.py`)
  - [ ] Add comments explaining reciprocal rank formula
  - [ ] Explain alpha weighting

- [ ] **Retry Backoff Logic** (`app/adapters/embedding/ollama_embedding.py`)
  - [ ] Comment exponential backoff calculation
  - [ ] Explain jitter

- [ ] **Metadata Sanitization** (`app/adapters/vector_store/chroma_store.py`)
  - [ ] Comment Chroma metadata requirements
  - [ ] Explain sanitization rules

- [ ] **Evaluation State Machine** (`app/api/routes/evaluation.py`)
  - [ ] Add flow diagram in comments
  - [ ] Explain state transitions

---

## Phase 8: CI/CD & Final Validation

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check coverage: `pytest tests/ --cov=app --cov-report=html --cov-report=term`
- [ ] Verify 80%+ coverage target achieved
- [ ] Run linting (if configured)
- [ ] Test GitHub Actions workflow locally (act tool or push to branch)
- [ ] Update CLAUDE.md with testing instructions
- [ ] Create pull request with comprehensive description

---

## Commit Strategy

After each major section completion:
1. ✅ Complete phase/sub-phase
2. ✅ Run tests for that section
3. ✅ Mark checkboxes in this plan
4. ✅ Commit changes with descriptive message
5. ✅ Move to next section

**Example commit messages:**
- `test: add domain layer tests with fixtures`
- `test: add SimplifiedDocumentProcessingUseCase tests`
- `refactor: extract retry logic to shared utility`
- `docs: add docstrings to ContextRetrievalUseCase`

---

## Notes

- **Mock Ollama API**: All tests mock external Ollama HTTP calls (no real service required)
- **In-memory databases**: Use in-memory SQLite and Chroma for adapter tests
- **Target coverage**: 80%+ overall (90%+ domain/use cases, 70%+ adapters)
- **Dead code removal**: Remove commented code in documents.py
