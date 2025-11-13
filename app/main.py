import os
import logging
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import our middleware
from fastapi.middleware.cors import CORSMiddleware
from .infrastructure.logging import setup_logging, RequestLoggingMiddleware
from .infrastructure.metrics import PrometheusMiddleware, metrics_endpoint

from .config import settings

# Import adapter classes
from .adapters.document_processing.chunking_strategies import OptimizedHierarchicalChunkingStrategy
from .adapters.token_service.tiktoken_service import TikTokenService
from .adapters.storage.local_storage import LocalStorageService
from .adapters.document_processing.remote_document_processor import RemoteDocumentProcessor
from .adapters.embedding.ollama_embedding import OllamaEmbeddingService
from .adapters.llm.ollama_llm import OllamaLLMService
from .adapters.vector_store.chroma_store import ChromaVectorStoreAdapter
from .adapters.search.bm25_adapter import BM25Adapter
from .adapters.search.rank_fusion_adapter import HybridRankFusionAdapter
from .adapters.persistence.sqlite_repository import (
    SQLiteDocumentRepository, SQLiteCollectionRepository, SQLiteChatRepository
)
from .adapters.persistence.evaluation_repository import SQLiteEvaluationRepository
from .adapters.judge.langchain_evaluation_service import LangChainEvaluationService
from .adapters.model_info.ollama_model_info import OllamaModelInfoAdapter

# Imports for routers
from .api.routes.documents import router as documents_router
from .api.routes.collections import router as collections_router
from .api.routes.search_u import router as search_router
from .api.routes.chat import router as chat_router
from .api.routes.health import router as health_router
from .api.routes.web import router as web_router
from .api.routes.evaluation import router as evaluation_router


app = FastAPI(
    title="Chat with Documents",
    debug=settings.DEBUG
)

# Setup logging early
setup_logging()
logger = logging.getLogger(__name__)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)
# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3034",
]

# CORS (if frontend served separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path("web/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web/templates")


# Include metrics endpoint
@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()


# On startup, instantiate and store singleton adapter instances in app.state
@app.on_event("startup")
async def on_startup():
    logger.info("Application startup: instantiating adapters...")
    logger.debug(f"Settings: {settings}")
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

    # Log configuration for debugging
    logger.info(f"Embedding configuration:")
    logger.info(f"  - Provider: {settings.EMBEDDING_PROVIDER}")
    logger.info(f"  - Model: {settings.OLLAMA_EMBEDDING_MODEL}")
    logger.info(f"  - Base URL: {settings.OLLAMA_EMBEDDING_BASE_URL}")
    logger.info(f"  - Batch size: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"  - Concurrency: {settings.EMBEDDING_CONCURRENCY}")
    logger.info(f"  - Timeout: {settings.EMBEDDING_TIMEOUT}s")

    # Storage service
    storage_service = LocalStorageService()
    app.state.storage_service = storage_service

    # Token service
    token_service = TikTokenService()

    # Chunking Strategy
    chunking_strategy = OptimizedHierarchicalChunkingStrategy(token_service)

    # Document processor
    if settings.DOCUMENT_PROCESSOR_API_KEY:
        document_processor_api_key = settings.DOCUMENT_PROCESSOR_API_KEY.get_secret_value()
    else:
        document_processor_api_key = None
    app.state.document_processor = RemoteDocumentProcessor(
        api_base_url=str(settings.DOCUMENT_PROCESSOR_API_URL),
        storage_service=storage_service,
        timeout=settings.DOCUMENT_PROCESSOR_TIMEOUT,
        max_retries=settings.DOCUMENT_PROCESSOR_MAX_RETRIES,
        api_key=document_processor_api_key
    )

    # Embedding service
    if settings.EMBEDDING_PROVIDER.lower() == "ollama":
        app.state.embedding_service = OllamaEmbeddingService(
            base_url=str(settings.OLLAMA_EMBEDDING_BASE_URL),
            model_name=str(settings.OLLAMA_EMBEDDING_MODEL),
            timeout=int(settings.EMBEDDING_TIMEOUT) if settings.EMBEDDING_TIMEOUT else 120,
            batch_size=int(settings.EMBEDDING_BATCH_SIZE) if settings.EMBEDDING_BATCH_SIZE else 4,
            concurrency_limit=int(settings.EMBEDDING_CONCURRENCY) if settings.EMBEDDING_CONCURRENCY else 1,
            max_retries=int(settings.EMBEDDING_MAX_RETRIES) if hasattr(settings, 'EMBEDDING_MAX_RETRIES') else 3,
            retry_delay=float(settings.EMBEDDING_RETRY_DELAY) if hasattr(settings, 'EMBEDDING_RETRY_DELAY') else 2.0,
        )
        logger.info(
            f"Initialized Ollama embedding service: "
            f"batch_size={settings.EMBEDDING_BATCH_SIZE}, "
            f"concurrency={settings.EMBEDDING_CONCURRENCY}"
        )
    else:
        raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")

    # LLM service
    if settings.LLM_PROVIDER.lower() == "ollama":
        app.state.llm_service = OllamaLLMService(
            base_url=str(settings.OLLAMA_LLM_BASE_URL),
            model_name=settings.OLLAMA_LLM_MODEL,
            timeout=settings.LLM_TIMEOUT
        )
    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")

    # Contextual LLM service
    if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
        app.state.contextual_llm_service = OllamaLLMService(
            base_url=str(settings.OLLAMA_CONTEXTUAL_LLM_BASE_URL),
            model_name=settings.OLLAMA_CONTEXTUAL_LLM_MODEL,
            timeout=settings.LLM_TIMEOUT
        )
    else:
        # Create a None placeholder so dependency injection doesn't fail
        app.state.contextual_llm_service = None

    # Model info service (for dynamic context window detection)
    app.state.model_info_service = OllamaModelInfoAdapter(
        base_url=settings.OLLAMA_LLM_BASE_URL,
        default_context_window=settings.DEFAULT_CONTEXT_WINDOW
    )
    logger.info(f"Initialized model info service (default context window: {settings.DEFAULT_CONTEXT_WINDOW})")

    await verify_ollama_services(app)

    # Vector store
    persist_dir = settings.CHROMA_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)
    app.state.vector_store = ChromaVectorStoreAdapter(
        persist_directory=persist_dir,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )

    # BM25 service
    bm25_persist_dir = settings.BM25_PERSIST_DIR
    os.makedirs(bm25_persist_dir, exist_ok=True)
    app.state.bm25_service = BM25Adapter(
        persist_directory=bm25_persist_dir,
        index_name=settings.BM25_INDEX_NAME
    )
    logger.info(f"Initialized BM25 service with index: {settings.BM25_INDEX_NAME}")

    # Rank fusion service (stateless, but keeping for consistency)
    app.state.rank_fusion_service = HybridRankFusionAdapter()

    # Repository adapters
    document_repo = SQLiteDocumentRepository(settings.DATABASE_URL)
    collection_repo = SQLiteCollectionRepository(settings.DATABASE_URL)
    chat_repo = SQLiteChatRepository(settings.DATABASE_URL)

    # Initialize tables
    await document_repo.init_models()
    await collection_repo.init_models()
    await chat_repo.init_models()

    # Store in app.state so dependency functions can retrieve them
    app.state.document_repo = document_repo
    app.state.collection_repo = collection_repo
    app.state.chat_repo = chat_repo

    # Evaluation services initialization (always initialized)
    logger.info("Initializing evaluation services...")

    # Judge service (LangChain with OpenAI)
    if not settings.OPENAI_EVALUATION_API_KEY:
        logger.warning("⚠️  OPENAI_EVALUATION_API_KEY not set. Judge service unavailable.")
        app.state.judge_service = None
    else:
        app.state.judge_service = LangChainEvaluationService(
            api_key=settings.OPENAI_EVALUATION_API_KEY.get_secret_value(),
            model_name=settings.OPENAI_EVALUATION_MODEL,
            timeout=settings.OPENAI_EVALUATION_TIMEOUT
        )
        logger.info(f"✅ Initialized judge service with model: {settings.OPENAI_EVALUATION_MODEL}")

    # Evaluation repository (always initialized)
    evaluation_repo = SQLiteEvaluationRepository(settings.DATABASE_URL)
    await evaluation_repo.init_models()
    app.state.evaluation_repo = evaluation_repo
    logger.info("✅ Initialized evaluation repository")

    # Evaluation state repository (for state persistence)
    from app.adapters.persistence.evaluation_repository import SQLiteEvaluationStateRepository
    evaluation_state_repo = SQLiteEvaluationStateRepository(settings.DATABASE_URL)
    await evaluation_state_repo.init_models()
    app.state.evaluation_state_repo = evaluation_state_repo
    logger.info("✅ Initialized evaluation state repository")

    # Initialize test collection (only if flag is true)
    if settings.INITIALIZE_TEST_COLLECTION:
        logger.info("Initializing test collection...")

        from .core.use_cases.evaluation.initialize_test_collection import InitializeTestCollectionUseCase
        from .core.use_cases.document_management import DocumentManagementUseCase

        # Create document processing use case for test collection initialization
        doc_processing_use_case = DocumentManagementUseCase(
            document_repo=document_repo,
            collection_repo=collection_repo,
            document_processor=app.state.document_processor,
            embedding_service=app.state.embedding_service,
            vector_store=app.state.vector_store,
            llm_service=app.state.llm_service,
            contextual_llm=app.state.contextual_llm_service,
            bm25_service=app.state.bm25_service,
        )

        init_test_collection_use_case = InitializeTestCollectionUseCase(
            collection_repo=collection_repo,
            document_repo=document_repo,
            document_processing_use_case=doc_processing_use_case,
            test_collection_id=settings.EVALUATION_TEST_COLLECTION_ID,
            test_collection_name=settings.EVALUATION_TEST_COLLECTION_NAME,
            test_docs_dir=settings.EVALUATION_TEST_DOCS_DIR
        )

        # Initialize test collection if needed
        try:
            initialized = await init_test_collection_use_case.initialize_if_needed()
            if initialized:
                logger.info("✅ Evaluation test collection initialized successfully")
            else:
                logger.info("✅ Evaluation test collection already exists")
        except Exception as e:
            logger.error(f"⚠️  Failed to initialize evaluation test collection: {str(e)}")
            # Don't fail startup if test collection initialization fails
    else:
        logger.info("⚠️  Test collection initialization disabled (INITIALIZE_TEST_COLLECTION=false)")

    logger.info("Startup complete: adapters instantiated")


async def verify_ollama_services(app):
    """Sanity test Ollama LLM and Embedding endpoints to fail fast if not reachable"""
    logger.info("🔍 Running Ollama service connectivity checks...")

    try:
        # Test embedding API
        embedding_service = app.state.embedding_service
        test_text = "health check test"
        emb = await embedding_service.generate_embedding(test_text)
        if not emb or not emb.values:
            raise RuntimeError("Embedding test failed: empty result.")
        logger.info(f"✅ Embedding service OK ({embedding_service.model_name})")

        # Test LLM API
        llm_service = app.state.llm_service
        test_prompt = "Hello, are you online?"
        resp = await llm_service.generate_response(test_prompt)
        if not resp:
            raise RuntimeError("LLM test failed: no response.")
        logger.info(f"✅ LLM service OK ({llm_service.model_name})")

        # Test contextual LLM (if enabled)
        contextual_llm_service = getattr(app.state, "contextual_llm_service", None)
        if contextual_llm_service:
            resp2 = await contextual_llm_service.generate_response("This is a contextual test")
            if not resp2:
                raise RuntimeError("Contextual LLM test failed: no response.")
            logger.info(f"✅ Contextual LLM OK ({contextual_llm_service.model_name})")

        logger.info("🚀 Ollama service checks complete. All good.")

    except Exception as e:
        logger.error(f"❌ Ollama service health check failed: {e}")
        raise  # Stop startup if core services aren't healthy


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application shutdown: closing resources...")
    # Close BM25 service (cleanup thread pool)
    bm25_service = getattr(app.state, "bm25_service", None)
    if bm25_service:
        try:
            # The BM25Adapter has a __del__ method that handles thread pool cleanup
            # but we can also explicitly shut it down if needed
            if hasattr(bm25_service, '_executor'):
                bm25_service._executor.shutdown(wait=True)
                logger.info("Closed BM25 service thread pool")
        except Exception as e:
            logger.warning(f"Error closing BM25 service: {e}")

    # Close embedding service client if exists
    embedding_svc = getattr(app.state, "embedding_service", None)
    if embedding_svc and hasattr(embedding_svc, "client"):
        try:
            await embedding_svc.client.aclose()
            logger.info("Closed embedding_service HTTP client")
        except Exception as e:
            logger.warning(f"Error closing embedding_service client: {e}")

    llm_svc = getattr(app.state, "llm_service", None)
    if llm_svc and hasattr(llm_svc, "client"):
        try:
            await llm_svc.client.aclose()
            logger.info("Closed llm_service HTTP client")
        except Exception as e:
            logger.warning(f"Error closing llm_service client: {e}")

    c_llm_svc = getattr(app.state, "contextual_llm_service", None)
    if c_llm_svc and hasattr(c_llm_svc, "client"):
        try:
            await c_llm_svc.client.aclose()
            logger.info("Closed contextual_llm_service HTTP client")
        except Exception as e:
            logger.warning(f"Error closing contextual_llm_service client: {e}")

    # Close DB engine
    document_repo = getattr(app.state, "document_repo", None)
    if document_repo and hasattr(document_repo, "_engine"):
        try:
            await document_repo._engine.dispose()
            logger.info("Closed document_repo engine")
        except Exception as e:
            logger.warning(f"Error disposing document_repo engine: {e}")

    collection_repo = getattr(app.state, "collection_repo", None)
    if collection_repo and hasattr(collection_repo, "_engine"):
        try:
            await collection_repo._engine.dispose()
            logger.info("Closed collection_repo engine")
        except Exception as e:
            logger.warning(f"Error disposing collection_repo engine: {e}")

    chat_repo = getattr(app.state, "chat_repo", None)
    if chat_repo and hasattr(chat_repo, "_engine"):
        try:
            await chat_repo._engine.dispose()
            logger.info("Closed chat_repo engine")
        except Exception as e:
            logger.warning(f"Error disposing chat_repo engine: {e}")

    logger.info("Shutdown complete.")


app.include_router(
    documents_router,
    prefix="/documents",
    tags=["documents"]
)

app.include_router(
    collections_router,
    prefix="/collections",
    tags=["collections"]
)

app.include_router(
    search_router,
    prefix="/search",
    tags=["search"]
)

app.include_router(
    chat_router,
    prefix="/chat",
    tags=["chat"]
)

app.include_router(
    health_router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    web_router,
    prefix="",
    tags=["web"]
)

app.include_router(
    evaluation_router,
    prefix="/api/evaluation",
    tags=["evaluation"]
)
