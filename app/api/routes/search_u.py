# app/api/routes/search.py

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...api.deps import get_search_documents_use_case
from ...core.use_cases.search_documents import SearchDocumentsUseCase
from ...core.domain.entities.document_chunk import DocumentChunk
from ...core.domain.entities.query_transformation import QueryTransformationMethod
from ...core.use_cases.intent_classification import Intent, IntentType

router = APIRouter()


# Pydantic models for request/response
class SearchRequestQueryTransformation(BaseModel):
    """Configuration for query transformation"""
    enabled: bool = True
    methods: List[QueryTransformationMethod] = Field(
        default=[QueryTransformationMethod.CONTEXTUALIZE],
        description="List of transformation methods to apply"
    )


class SearchRequestConfig(BaseModel):
    """Configuration for search request"""
    use_chat_history: bool = Field(
        default=True,
        description="Whether to use chat history for contextualization"
    )
    max_history_turns: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum conversation turns to consider"
    )
    top_k: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Number of results to return"
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (vector + BM25)"
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search (0=BM25 only, 1=vector only)"
    )
    use_intent_classification: bool = Field(
        default=True,
        description="Enable intent classification"
    )
    query_transformation: Optional[SearchRequestQueryTransformation] = Field(
        default_factory=SearchRequestQueryTransformation
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for search"
    )


class ChatMessage(BaseModel):
    """Chat message format"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class SearchRequest(BaseModel):
    """Search request payload"""
    query_text: str = Field(..., min_length=1, description="Search query")
    collection_id: Optional[str] = Field(None, description="Collection to search in")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Conversation history for contextualization"
    )
    config: SearchRequestConfig = Field(default_factory=SearchRequestConfig)


class IntentResponse(BaseModel):
    """Intent classification response"""
    type: str
    requires_retrieval: bool
    requires_collection_scan: bool
    confidence: float
    reasoning: str


class DocumentChunkResponse(BaseModel):
    """Document chunk response"""
    id: str
    document_id: str
    collection_id: Optional[str]
    content: str
    chunk_index: Optional[int]
    chunk_type: Optional[str]
    parent_chunk_id: Optional[str]
    metadata: Optional[Dict[str, Any]]


class SearchResponse(BaseModel):
    """Search response with metadata"""
    chunks: List[DocumentChunkResponse]
    intent: Optional[IntentResponse] = None
    transformed_queries: Optional[List[str]] = None
    summary: Optional[str] = None  # For collection summary requests
    message: Optional[str] = None  # For chat intents
    metadata: Dict[str, Any] = Field(default_factory=dict)


def to_chunk_response(chunk: DocumentChunk) -> DocumentChunkResponse:
    """Convert domain entity to API response"""
    return DocumentChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        collection_id=chunk.collection_id,
        content=chunk.content,
        chunk_index=getattr(chunk, "chunk_index", None),
        chunk_type=getattr(chunk, "chunk_type", None),
        parent_chunk_id=getattr(chunk, "parent_chunk_id", None),
        metadata=chunk.metadata,
    )


def to_intent_response(intent: Intent) -> IntentResponse:
    """Convert Intent to API response"""
    return IntentResponse(
        type=intent.type.value,
        requires_retrieval=intent.requires_retrieval,
        requires_collection_scan=intent.requires_collection_scan,
        confidence=intent.confidence,
        reasoning=intent.reasoning
    )


@router.post("/", response_model=SearchResponse)
async def search_documents(
    req: SearchRequest,
    use_case: SearchDocumentsUseCase = Depends(get_search_documents_use_case)
):
    """
    Search for relevant document chunks with intelligent query processing.
    
    Features:
    - Intent classification (detects chat, retrieval, summary requests)
    - Query transformation (contextualization, multi-query, HyDE)
    - Hybrid search (vector + BM25 with rank fusion)
    - Collection-level summaries
    
    Returns:
        SearchResponse with chunks, intent, and metadata
    """
    try:
        # Convert chat history to dict format
        chat_history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in req.chat_history
        ] if req.chat_history else []
        
        # Prepare query transformation methods
        transformation_methods = []
        if req.config.query_transformation and req.config.query_transformation.enabled:
            transformation_methods = req.config.query_transformation.methods
        
        # Execute search
        result = await use_case.search_documents(
            query_text=req.query_text,
            collection_id=req.collection_id,
            chat_history=chat_history_dicts if req.config.use_chat_history else None,
            top_k=req.config.top_k,
            filters=req.config.filters,
            use_hybrid=req.config.use_hybrid,
            alpha=req.config.alpha,
            use_intent_classification=req.config.use_intent_classification,
            query_transformation_enabled=req.config.query_transformation.enabled if req.config.query_transformation else False,
            query_transformation_methods=transformation_methods,
            max_history_turns=req.config.max_history_turns,
        )
        
        # Build response
        response = SearchResponse(
            chunks=[to_chunk_response(chunk) for chunk in result.get("chunks", [])],
            intent=to_intent_response(result["intent"]) if result.get("intent") else None,
            transformed_queries=result.get("transformed_queries"),
            summary=result.get("summary"),
            message=result.get("message"),
            metadata=result.get("metadata", {})
        )
        
        return response
        
    except Exception as e:
        # Log the error
        print(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health")
async def search_health_check():
    """Health check endpoint for search service"""
    return {"status": "healthy", "service": "search"}
