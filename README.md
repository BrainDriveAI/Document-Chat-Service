# Chat with Documents

This project is a sophisticated RAG (Retrieval-Augmented Generation) application built on a Clean Architecture. It provides a flexible and extensible framework for interacting with your documents using a conversational interface.

## Getting Started

### Project Structure

```
chat-with-docs/
├── README.md
├── pyproject.toml                   # Poetry dependency management
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

### 1\. Local Development Setup (Recommended for Contributors)

This method allows you to run the application directly on your machine, which is ideal for debugging, making changes, and developing new features.

#### Prerequisites

  * [Python 3.12](https://www.python.org/downloads/)
  * [Poetry](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation) for dependency management.
  * An [Ollama](https://ollama.com/) server running locally or accessible via a URL.
  * An accessible instance of the [BrainDrive-Document-AI](https://github.com/BrainDriveAI/BrainDrive-Document-AI) service for document processing.

#### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/BrainDriveAI/chat-with-your-documents.git
    cd chat-with-documents-dev
    ```

2.  **Set up the environment file:**
    Copy the example file and fill in your specific configuration, including the URLs for your Ollama and Document Processor services.

    ```bash
    cp .env.example .env
    # Open .env in your editor and modify the variables as needed.
    ```

3.  **Install dependencies with Poetry:**
    This will install all project dependencies and set up a virtual environment.

    ```bash
    poetry install
    ```

4.  **Activate the virtual environment:**

    ```bash
    poetry shell
    ```

5.  **Start the application:**

    ```bash
    uvicorn app.main:app --reload
    ```

      * The `--reload` flag enables live-reloading during development.

6.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8000`. You can also access the interactive API documentation at `http://localhost:8000/docs`.

### 2\. Docker Compose Setup

This is the fastest way to get the application running without worrying about local dependencies. It's ideal for a quick deployment or testing.

#### Prerequisites

  * [Docker](https://docs.docker.com/get-docker/) installed and running.
  * An Ollama server running, with chat and embedding models pulled (e.g., `llama3.2:8b`, `mxbai-embed-large`, and `llama3.2:3b`).

#### Steps

1.  **Clone the repository.**
    ```bash
    git clone https://github.com/BrainDriveAI/chat-with-your-documents.git
    cd chat-with-documents-dev
    ```
2.  **Configure your environment variables.**
    Create your `.env` file from the example and provide the necessary configuration values. This is crucial for the Docker container to pick up your settings.
    ```bash
    cp .env.example .env
    # Open the .env file and set the URLs for your Ollama and Document Processor services.
    ```
3.  **Run the application.**
    The `docker-compose.yml` file will build the image and start the container, mapping the necessary ports and volumes.
    ```bash
    docker-compose up --build
    ```
4.  **Access the application at `http://localhost:8000`.**

### 3\. Advanced Deployment

For a production-ready setup with Nginx and Prometheus, use the provided `docker-compose.prod.yml` file.

#### Steps

1.  **Follow the basic setup steps above.**
2.  **Build and run the production stack.**
    ```bash
    docker-compose -f docker-compose.prod.yml up --build -d
    ```

-----

## Key Design Principles

### 1\. Clean Architecture

  - **Domain**: Pure business logic, no external dependencies.
  - **Ports**: Abstract interfaces for external services.
  - **Adapters**: Concrete implementations of external services.
  - **Use Cases**: Application-specific business rules.

### 2\. Dependency Injection

  - FastAPI's dependency system for clean Inversion of Control.
  - Easy to swap implementations (Ollama → OpenAI, Chroma → Qdrant).

### 3\. Provider Agnostic

  - Abstract base classes for all external services.
  - Configuration-driven provider selection.

### 4\. Easy Setup

  - Docker Compose for one-command deployment.
  - Poetry for reproducible dependencies.

### 5\. Extensible

  - Easy to add new retrieval strategies.
  - Configurable chunking strategies.

-----

## Technology Stack Summary

| Component               | Technology                 | Purpose                          |
|-------------------------|----------------------------|----------------------------------|
| **Web Framework**       | FastAPI                    | API endpoints, WebSocket support |
| **Document Processing** | spaCy Layout               | PDF/Word structure extraction    |
| **Embeddings**          | mxbai-embed-large (Ollama) | Vector representations           |
| **Vector Store**        | Chroma                     | Document search and storage      |
| **LLM**                 | llama3.2:3b/8b (Ollama)    | Chat responses                   |
| **Orchestration**       | LangGraph                  | RAG pipeline management          |
| **Database**            | SQLite                     | Metadata and collections         |
| **Frontend**            | HTML/JS                    | Simple chat interface            |
| **Container**           | Docker + Docker Compose    | Easy deployment                  |

-----

## Next Steps

1.  Set up core domain entities and ports.
2.  Implement Ollama adapters (LLM + Embeddings).
3.  Create spaCy Layout document processor.
4.  Build Chroma vector store adapter.
5.  Implement LangGraph orchestrator.
6.  Create FastAPI routes and controllers.
7.  Add simple frontend interface.
