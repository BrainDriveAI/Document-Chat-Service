import os
import logging
import asyncio
import uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import our middleware
from fastapi.middleware.cors import CORSMiddleware
from .infrastructure.multipart_limit_middleware import MultipartLimitMiddleware
from .infrastructure.logging import setup_logging, RequestLoggingMiddleware
from .infrastructure.metrics import PrometheusMiddleware, metrics_endpoint

from .config import settings

# Import adapter classes
from .adapters.document_processing.chunking_strategies import OptimizedHierarchicalChunkingStrategy
from .adapters.token_service.tiktoken_service import TikTokenService
from .adapters.document_processing.spacy_layout_processor import SpacyLayoutProcessor
from .adapters.embedding.ollama_embedding import OllamaEmbeddingService
from .adapters.llm.ollama_llm import OllamaLLMService
from .adapters.vector_store.chroma_store import ChromaVectorStoreAdapter
from .adapters.search.bm25_adapter import BM25Adapter
from .adapters.search.rank_fusion_adapter import HybridRankFusionAdapter
from .adapters.persistence.sqlite_repository import (
    SQLiteDocumentRepository, SQLiteCollectionRepository, SQLiteChatRepository
)

# Imports for routers
from .api.routes.documents import router as documents_router
from .api.routes.collections import router as collections_router
from .api.routes.search import router as search_router
from .api.routes.chat import router as chat_router
from .api.routes.health import router as health_router
from .api.routes.web import router as web_router


app = FastAPI(
    title="Chat with Documents",
    debug=settings.DEBUG
)

# Setup logging early
setup_logging()
logger = logging.getLogger(__name__)

# Add Multipart limit middleware with, e.g., 50 MiB limit. Adjust as needed.
# app.add_middleware(
#     MultipartLimitMiddleware,
#     max_part_size=settings.UPLOAD_MAX_PART_SIZE,
#     max_fields=settings.UPLOAD_MAX_FIELDS,
#     max_file_size=settings.UPLOAD_MAX_FILE_SIZE
# )

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)
# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# CORS (if frontend served separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
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
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

    # Token service
    token_service = TikTokenService()

    # Chunking Strategy
    chunking_strategy = OptimizedHierarchicalChunkingStrategy(token_service)

    # Document processor
    app.state.document_processor = SpacyLayoutProcessor(
        spacy_model=settings.SPACY_MODEL,
    )

    # Embedding service
    if settings.EMBEDDING_PROVIDER.lower() == "ollama":
        app.state.embedding_service = OllamaEmbeddingService(
            base_url=str(settings.OLLAMA_EMBEDDING_BASE_URL),
            model_name=settings.OLLAMA_EMBEDDING_MODEL,
            timeout=settings.EMBEDDING_TIMEOUT
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

    logger.info("Startup complete: adapters instantiated")


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
