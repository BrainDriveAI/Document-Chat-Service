import os
from fastapi import Request, HTTPException, Depends
from pathlib import Path

from ..config import settings

# Adapter classes
from ..adapters.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from ..adapters.logging.python_logger import PythonLogger
from ..adapters.clustering.sklearn_clustering import SklearnClusteringAdapter

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
from ..core.ports.clustering_service import ClusteringService
from ..core.ports.judge_service import JudgeService
from ..core.ports.evaluation_repository import EvaluationRepository
from ..core.ports.repositories import (
    DocumentRepository, CollectionRepository, ChatRepository
)
from ..core.ports.model_info_service import ModelInfoService

# Use-case classes
from ..core.use_cases.document_management import DocumentManagementUseCase
from ..core.use_cases.collection_management import CollectionManagementUseCase
from ..core.use_cases.search_documents_u import SearchDocumentsUseCase
from ..core.use_cases.chat_interaction import ChatInteractionUseCase
from ..core.use_cases.query_transformation import QueryTransformationUseCase
from ..core.use_cases.intent_classification import IntentClassificationUseCase
from ..core.use_cases.collection_summary import CollectionSummaryUseCase
from ..core.use_cases.context_retrieval import ContextRetrievalUseCase
from ..core.use_cases.evaluation.initialize_test_collection import InitializeTestCollectionUseCase
from ..core.use_cases.evaluation.run_evaluation import RunEvaluationUseCase
from ..core.use_cases.evaluation.get_results import GetEvaluationResultsUseCase
from ..core.use_cases.evaluation.start_plugin_evaluation import StartPluginEvaluationUseCase
from ..core.use_cases.evaluation.submit_plugin_evaluation import SubmitPluginEvaluationUseCase


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


def get_model_info_service(request: Request) -> ModelInfoService:
    """Get model info service instance for dynamic context window detection"""
    model_info = getattr(request.app.state, "model_info_service", None)
    if model_info is None:
        raise HTTPException(status_code=500, detail="ModelInfoService not initialized")
    return model_info


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

def get_clustering_service() -> ClusteringService:
    """Get Clustering service instance"""
    return SklearnClusteringAdapter()


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
) -> DocumentManagementUseCase:
    return DocumentManagementUseCase(
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


def get_query_transformation_use_case(
        llm_service: LLMService = Depends(get_llm_service),
) -> QueryTransformationUseCase:
    return QueryTransformationUseCase(llm_service=llm_service)

def get_intent_classification_use_case(
    llm_service: LLMService = Depends(get_llm_service),
) -> IntentClassificationUseCase:
    """Get intent classification use case"""
    return IntentClassificationUseCase(llm_service=llm_service)

def get_collection_summary_use_case(
    vector_store: VectorStore = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> CollectionSummaryUseCase:
    """Get collection summary use case"""
    return CollectionSummaryUseCase(
        vector_store=vector_store,
        llm_service=llm_service,
        clustering_service=clustering_service,
    )


def get_search_documents_use_case(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_service: BM25Service = Depends(get_bm25_service),
    rank_fusion_service: RankFusionService = Depends(get_rank_fusion_service),
    query_transformation_use_case: QueryTransformationUseCase = Depends(get_query_transformation_use_case),
    intent_classification_use_case: IntentClassificationUseCase = Depends(get_intent_classification_use_case),
    collection_summary_use_case: CollectionSummaryUseCase = Depends(get_collection_summary_use_case),
) -> SearchDocumentsUseCase:
    return SearchDocumentsUseCase(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        rank_fusion_service=rank_fusion_service,
        query_transformation_use_case=query_transformation_use_case,
        intent_classification_use_case=intent_classification_use_case,
        collection_summary_use_case=collection_summary_use_case,
    )

def get_context_retrieval_use_case(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_service: BM25Service = Depends(get_bm25_service),
    rank_fusion_service: RankFusionService = Depends(get_rank_fusion_service),
    query_transformation_use_case: QueryTransformationUseCase = Depends(get_query_transformation_use_case),
    intent_classification_use_case: IntentClassificationUseCase = Depends(get_intent_classification_use_case),
    collection_summary_use_case: CollectionSummaryUseCase = Depends(get_collection_summary_use_case),
    model_info_service: ModelInfoService = Depends(get_model_info_service),
) -> ContextRetrievalUseCase:
    """Get context retrieval use case - the main entry point for context retrieval"""
    return ContextRetrievalUseCase(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        rank_fusion_service=rank_fusion_service,
        query_transformation_use_case=query_transformation_use_case,
        intent_classification_use_case=intent_classification_use_case,
        collection_summary_use_case=collection_summary_use_case,
        model_info_service=model_info_service,
    )


def get_chat_interaction_use_case(
        chat_repo: ChatRepository = Depends(get_chat_repository),
        orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> ChatInteractionUseCase:
    return ChatInteractionUseCase(chat_repo=chat_repo, orchestrator=orchestrator)


# Evaluation dependency providers
def get_evaluation_repository(request: Request) -> EvaluationRepository:
    """Get evaluation repository instance from app state"""
    repo = getattr(request.app.state, "evaluation_repo", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="EvaluationRepository not initialized")
    return repo


def get_evaluation_state_repository(request: Request):
    """Get evaluation state repository instance from app state"""
    from ..core.ports.evaluation_state_repository import EvaluationStateRepository
    repo = getattr(request.app.state, "evaluation_state_repo", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="EvaluationStateRepository not initialized")
    return repo


def get_judge_service(request: Request) -> JudgeService:
    """Get judge service instance from app state"""
    judge = getattr(request.app.state, "judge_service", None)
    if judge is None:
        raise HTTPException(status_code=500, detail="JudgeService not initialized")
    return judge


def get_initialize_test_collection_use_case(
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    document_repo: DocumentRepository = Depends(get_document_repository),
    document_processing_use_case: DocumentManagementUseCase = Depends(get_document_processing_use_case),
) -> InitializeTestCollectionUseCase:
    """Get initialize test collection use case"""
    return InitializeTestCollectionUseCase(
        collection_repo=collection_repo,
        document_repo=document_repo,
        document_processing_use_case=document_processing_use_case,
        test_collection_id=settings.EVALUATION_TEST_COLLECTION_ID,
        test_collection_name=settings.EVALUATION_TEST_COLLECTION_NAME,
        test_docs_dir=settings.EVALUATION_TEST_DOCS_DIR
    )


def get_run_evaluation_use_case(
    evaluation_repo: EvaluationRepository = Depends(get_evaluation_repository),
    judge_service: JudgeService = Depends(get_judge_service),
    llm_service: LLMService = Depends(get_llm_service),
    context_retrieval_use_case: ContextRetrievalUseCase = Depends(get_context_retrieval_use_case),
    initialize_test_collection_use_case: InitializeTestCollectionUseCase = Depends(get_initialize_test_collection_use_case),
) -> RunEvaluationUseCase:
    """Get run evaluation use case"""
    return RunEvaluationUseCase(
        evaluation_repo=evaluation_repo,
        judge_service=judge_service,
        llm_service=llm_service,
        context_retrieval_use_case=context_retrieval_use_case,
        initialize_test_collection_use_case=initialize_test_collection_use_case,
        test_collection_id=settings.EVALUATION_TEST_COLLECTION_ID
    )


def get_get_evaluation_results_use_case(
    evaluation_repo: EvaluationRepository = Depends(get_evaluation_repository),
) -> GetEvaluationResultsUseCase:
    """Get evaluation results use case"""
    return GetEvaluationResultsUseCase(evaluation_repo=evaluation_repo)


def get_start_plugin_evaluation_use_case(
    evaluation_repo: EvaluationRepository = Depends(get_evaluation_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    context_retrieval_use_case: ContextRetrievalUseCase = Depends(get_context_retrieval_use_case),
) -> StartPluginEvaluationUseCase:
    """Get start plugin evaluation use case"""
    return StartPluginEvaluationUseCase(
        evaluation_repo=evaluation_repo,
        collection_repo=collection_repo,
        context_retrieval=context_retrieval_use_case,
        test_collection_id=settings.EVALUATION_TEST_COLLECTION_ID,
        test_cases_path=str(Path(settings.EVALUATION_TEST_DOCS_DIR) / "test_cases.json"),
        concurrency=settings.EVALUATION_CONCURRENCY
    )


def get_submit_plugin_evaluation_use_case(
    evaluation_repo: EvaluationRepository = Depends(get_evaluation_repository),
    judge_service: JudgeService = Depends(get_judge_service),
) -> SubmitPluginEvaluationUseCase:
    """Get submit plugin evaluation use case"""
    return SubmitPluginEvaluationUseCase(
        evaluation_repo=evaluation_repo,
        judge_service=judge_service,
        test_cases_path=str(Path(settings.EVALUATION_TEST_DOCS_DIR) / "test_cases.json")
    )


# Evaluation State dependency providers
def get_save_evaluation_state_use_case(
    state_repo = Depends(get_evaluation_state_repository),
    eval_repo: EvaluationRepository = Depends(get_evaluation_repository),
):
    """Get save evaluation state use case"""
    from ..core.use_cases.evaluation.save_evaluation_state import SaveEvaluationStateUseCase
    return SaveEvaluationStateUseCase(
        state_repository=state_repo,
        evaluation_repository=eval_repo
    )


def get_load_evaluation_state_use_case(
    state_repo = Depends(get_evaluation_state_repository),
    eval_repo: EvaluationRepository = Depends(get_evaluation_repository),
):
    """Get load evaluation state use case"""
    from ..core.use_cases.evaluation.load_evaluation_state import LoadEvaluationStateUseCase
    return LoadEvaluationStateUseCase(
        state_repository=state_repo,
        evaluation_repository=eval_repo
    )


def get_delete_evaluation_state_use_case(
    state_repo = Depends(get_evaluation_state_repository),
):
    """Get delete evaluation state use case"""
    from ..core.use_cases.evaluation.delete_evaluation_state import DeleteEvaluationStateUseCase
    return DeleteEvaluationStateUseCase(state_repository=state_repo)


def get_list_evaluation_states_use_case(
    state_repo = Depends(get_evaluation_state_repository),
):
    """Get list evaluation states use case"""
    from ..core.use_cases.evaluation.list_evaluation_states import ListEvaluationStatesUseCase
    return ListEvaluationStatesUseCase(state_repository=state_repo)
