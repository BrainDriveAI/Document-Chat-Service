# Chat with Documents - Project Structure

```
chat-with-docs/
├── README.md
├── pyproject.toml                    # Poetry dependency management
├── docker-compose.yml               # Easy setup with Ollama + Chroma
├── Dockerfile
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration management
│   │
│   ├── core/                        # Clean Architecture Core
│   │   ├── __init__.py
│   │   ├── domain/                  # Domain layer - business entities
│   │   │   ├── __init__.py
│   │   │   ├── entities/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── document.py
│   │   │   │   ├── collection.py
│   │   │   │   ├── chat.py
│   │   │   │   └── chunk.py
│   │   │   ├── value_objects/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── embedding.py
│   │   │   │   └── metadata.py
│   │   │   └── exceptions.py
│   │   │
│   │   ├── ports/                   # Interfaces/Protocols
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py
│   │   │   ├── embedding_service.py
│   │   │   ├── vector_store.py
│   │   │   ├── llm_service.py
│   │   │   ├── orchestrator.py
│   │   │   └── repositories.py
│   │   │
│   │   └── use_cases/               # Application layer
│   │       ├── __init__.py
│   │       ├── collection_management.py
│   │       ├── document_processing.py
│   │       ├── chat_interaction.py
│   │       └── search_documents.py
│   │
│   ├── adapters/                    # External integrations
│   │   ├── __init__.py
│   │   │
│   │   ├── document_processing/
│   │   │   ├── __init__.py
│   │   │   ├── spacy_layout_processor.py
│   │   │   └── chunking_strategies.py
│   │   │
│   │   ├── embedding/
│   │   │   ├── __init__.py
│   │   │   ├── ollama_embedding.py
│   │   │   └── base_embedding.py
│   │   │
│   │   ├── vector_store/
│   │   │   ├── __init__.py
│   │   │   ├── chroma_store.py
│   │   │   └── hybrid_search.py
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── ollama_llm.py
│   │   │   └── base_llm.py
│   │   │
│   │   ├── orchestration/
│   │   │   ├── __init__.py
│   │   │   ├── langraph_orchestrator.py
│   │   │   └── retrieval_strategies.py
│   │   │
│   │   └── persistence/
│   │       ├── __init__.py
│   │       ├── sqlite_repository.py
│   │       └── models.py
│   │
│   ├── api/                         # FastAPI routes and controllers
│   │   ├── __init__.py
│   │   ├── deps.py                  # Dependency injection
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── collections.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── cors.py
│   │       └── error_handlers.py
│   │
│   └── infrastructure/              # Cross-cutting concerns
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── startup.py
│
├── frontend/                        # Simple web interface
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
│       ├── index.html
│       ├── chat.html
│       └── collections.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_use_cases/
│   │   ├── test_domain/
│   │   └── test_adapters/
│   ├── integration/
│   │   ├── test_api/
│   │   └── test_services/
│   └── e2e/
│       └── test_chat_flow.py
│
├── scripts/
│   ├── setup.sh                     # Initial setup script
│   ├── download_models.sh           # Download Ollama models
│   └── dev_setup.sh                 # Development environment setup
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── setup.md
│   └── contributing.md
│
└── data/                           # Runtime data
    ├── uploads/                    # Uploaded documents
    ├── vector_db/                  # Chroma database
    └── logs/
```

## Key Design Principles

### 1. Clean Architecture
- **Domain**: Pure business logic, no external dependencies
- **Ports**: Abstract interfaces for external services
- **Adapters**: Concrete implementations of external services
- **Use Cases**: Application-specific business rules

### 2. Dependency Injection
- FastAPI's dependency system for clean IoC
- Easy to swap implementations (Ollama → OpenAI, Chroma → Qdrant)

### 3. Provider Agnostic
- Abstract base classes for all external services
- Configuration-driven provider selection

### 4. Easy Setup
- Docker Compose for one-command deployment
- Poetry for reproducible dependencies
- Pre-configured Ollama models

### 5. Extensible
- Plugin-like architecture for new document types
- Easy to add new retrieval strategies
- Configurable chunking strategies

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | API endpoints, WebSocket support |
| **Document Processing** | spaCy Layout | PDF/Word structure extraction |
| **Embeddings** | mxbai-embed-large (Ollama) | Vector representations |
| **Vector Store** | Chroma | Document search and storage |
| **LLM** | llama3.2:3b/8b (Ollama) | Chat responses |
| **Orchestration** | LangGraph | RAG pipeline management |
| **Database** | SQLite | Metadata and collections |
| **Frontend** | HTML/JS | Simple chat interface |
| **Container** | Docker + Docker Compose | Easy deployment |

## Next Steps
1. Set up core domain entities and ports
2. Implement Ollama adapters (LLM + Embeddings)
3. Create spaCy Layout document processor
4. Build Chroma vector store adapter
5. Implement LangGraph orchestrator
6. Create FastAPI routes and controllers
7. Add simple frontend interface
