import os
from fastapi import Request, HTTPException, Depends
from pathlib import Path

from ..config import settings

# Adapter classes
from ..adapters.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from ..adapters.logging.python_logger import PythonLogger

# Port interfaces
from ..core.ports.logger import Logger
from ..core.ports.document_processor import DocumentProcessor
from ..core.ports.storage_service import StorageService
from ..core.ports.embedding_service import EmbeddingService
from ..core.ports.vector_store import VectorStore
from ..core.ports.llm_service import LLMService
from ..core.ports.bm25_service import BM25Service
from ..core.ports.rank_fusion_service import RankFusionService
from ..core.ports.orchestrator import ChatOrchestrator
from ..core.ports.repositories import (
    DocumentRepository, CollectionRepository, ChatRepository
)

# Use-case classes
# from ..core.use_cases.document_use_case import DocumentProcessingUseCase
from ..core.use_cases.simple_document import SimplifiedDocumentProcessingUseCase
from ..core.use_cases.collection_management import CollectionManagementUseCase
from ..core.use_cases.document_management import DocumentManagementUseCase
from ..core.use_cases.search_documents import SearchDocumentsUseCase
from ..core.use_cases.chat_interaction import ChatInteractionUseCase


# Dependency provider functions
def get_document_processor(request: Request) -> DocumentProcessor:
    dp = getattr(request.app.state, "document_processor", None)
    if dp is None:
        raise HTTPException(status_code=500, detail="DocumentProcessor not initialized")
    return dp

def get_storage_service(request: Request) -> StorageService:
    """
    Get storage service instance.
    Currently returns LocalStorageService, but can be easily swapped
    for cloud storage implementations.
    """
    storage_service = getattr(request.app.state, "storage_service", None)
    if storage_service is None:
        raise HTTPException(status_code=500, detail="Storage service not initialized")
    return storage_service

def get_embedding_service(request: Request) -> EmbeddingService:
    embedder = getattr(request.app.state, "embedding_service", None)
    if embedder is None:
        raise HTTPException(status_code=500, detail="EmbeddingService not initialized")
    return embedder


def get_llm_service(request: Request) -> LLMService:
    llm = getattr(request.app.state, "llm_service", None)
    if llm is None:
        raise HTTPException(status_code=500, detail="LLMService not initialized")
    return llm


def get_contextual_llm_service(request: Request) -> LLMService:
    llm = getattr(request.app.state, "contextual_llm_service", None)
    if settings.ENABLE_CONTEXTUAL_RETRIEVAL and llm is None:
        raise HTTPException(status_code=500, detail="Contextual LLMService not initialized")
    return llm


def get_vector_store(request: Request) -> VectorStore:
    vs = getattr(request.app.state, "vector_store", None)
    if vs is None:
        raise HTTPException(status_code=500, detail="VectorStore not initialized")
    return vs


def get_document_repository(request: Request) -> DocumentRepository:
    repo = getattr(request.app.state, "document_repo", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="DocumentRepository not initialized")
    return repo


def get_collection_repository(request: Request) -> CollectionRepository:
    repo = getattr(request.app.state, "collection_repo", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="CollectionRepository not initialized")
    return repo


def get_chat_repository(request: Request) -> ChatRepository:
    repo = getattr(request.app.state, "chat_repo", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="ChatRepository not initialized")
    return repo


def get_bm25_service(request: Request) -> BM25Service:
    """Get BM25 service instance from app state"""
    bm25 = getattr(request.app.state, "bm25_service", None)
    if bm25 is None:
        raise HTTPException(status_code=500, detail="BM25Service not initialized")
    return bm25


def get_rank_fusion_service(request: Request) -> RankFusionService:
    """Get Rank Fusion service instance from app state"""
    rank_f = getattr(request.app.state, "rank_fusion_service", None)
    if rank_f is None:
        raise HTTPException(status_code=500, detail="RankFusionService not initialized")
    return rank_f


def get_chat_orchestrator(
        embedding_service: EmbeddingService = Depends(get_embedding_service),
        vector_store: VectorStore = Depends(get_vector_store),
        llm_service: LLMService = Depends(get_llm_service),
) -> ChatOrchestrator:
    return LangGraphOrchestrator(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        top_k=5,
        max_context_chars=3000,
        temperature=0.1,
        max_tokens=2000
    )

def get_document_logger() -> Logger:
    return PythonLogger(__name__)

# Dependency provider for DocumentProcessingUseCase
def get_document_processing_use_case(
        document_repo: DocumentRepository = Depends(get_document_repository),
        collection_repo: CollectionRepository = Depends(get_collection_repository),
        document_processor: DocumentProcessor = Depends(get_document_processor),
        embedding_service: EmbeddingService = Depends(get_embedding_service),
        vector_store: VectorStore = Depends(get_vector_store),
        llm_service: LLMService = Depends(get_llm_service),
        contextual_llm: LLMService = Depends(get_contextual_llm_service),
        bm25_service: BM25Service = Depends(get_bm25_service),
) -> SimplifiedDocumentProcessingUseCase:
    return SimplifiedDocumentProcessingUseCase(
        document_repo=document_repo,
        collection_repo=collection_repo,
        document_processor=document_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        contextual_llm=contextual_llm,
        bm25_service=bm25_service,
    )


def get_collection_management_use_case(
        collection_repo: CollectionRepository = Depends(get_collection_repository)
) -> CollectionManagementUseCase:
    return CollectionManagementUseCase(collection_repo=collection_repo)

def get_document_management_use_case(
        document_repo: DocumentRepository = Depends(get_document_repository),
        vector_store: VectorStore = Depends(get_vector_store),
        storage_service: StorageService = Depends(get_storage_service),
        logger: Logger = Depends(get_document_logger),
        collection_management_use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case),
) -> DocumentManagementUseCase:
    return DocumentManagementUseCase(
        document_repo=document_repo,
        vector_store=vector_store,
        storage_service=storage_service,
        logger=logger,
        collection_management_use_case=collection_management_use_case,
    )


def get_search_documents_use_case(
        embedding_service: EmbeddingService = Depends(get_embedding_service),
        vector_store: VectorStore = Depends(get_vector_store),
        bm25_service: BM25Service = Depends(get_bm25_service),
        rank_fusion_service: RankFusionService = Depends(get_rank_fusion_service),
) -> SearchDocumentsUseCase:
    return SearchDocumentsUseCase(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        rank_fusion_service=rank_fusion_service,
    )


def get_chat_interaction_use_case(
        chat_repo: ChatRepository = Depends(get_chat_repository),
        orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> ChatInteractionUseCase:
    return ChatInteractionUseCase(chat_repo=chat_repo, orchestrator=orchestrator)
