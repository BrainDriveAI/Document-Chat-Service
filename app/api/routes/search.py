from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from enum import Enum

from ...api.deps import get_search_documents_use_case
from ...core.use_cases.search_documents import SearchDocumentsUseCase
from ...core.domain.entities.document_chunk import DocumentChunk
from app.core.domain.entities.query_transformation import QueryTransformationMethod

router = APIRouter()


# Pydantic models for request/response
class SearchRequestQueryTransformation(BaseModel):
    enabled: Optional[bool] = True
    methods: Optional[QueryTransformationMethod] = QueryTransformationMethod.CONTEXTUALIZE

class SearchRequestConfig(BaseModel):
    use_chat_history: Optional[bool] = True
    max_history_turns: Optional[int] = 3
    top_k: Optional[int] = Field(10, gt=0)
    use_hybrid: Optional[bool] = True
    query_transformation: Optional[SearchRequestQueryTransformation]
    filters: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    collection_id: Optional[str] = None
    chat_history: Optional[List[Dict]] = []
    config: SearchRequestConfig


class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    collection_id: Optional[str]
    content: str
    chunk_index: Optional[int]
    chunk_type: Optional[str]
    parent_chunk_id: Optional[str]
    metadata: Optional[Dict[str, Any]]


def to_chunk_response(chunk: DocumentChunk) -> DocumentChunkResponse:
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


@router.post("/", response_model=List[DocumentChunkResponse])
async def search_documents(
    req: SearchRequest,
    use_case: SearchDocumentsUseCase = Depends(get_search_documents_use_case)
):
    """
    Search for relevant document chunks given a query.
    """
    try:
        results: List[DocumentChunk] = await use_case.search_documents(
            query_text=req.query_text,
            collection_id=req.collection_id,
            top_k=req.top_k,
            filters=req.filters,
            use_hybrid=req.use_hybrid,
            use_query_transformation=req.use_query_transformation
        )
    except Exception as e:
        # If hybrid not implemented, fallback to vector-only?
        # For now, propagate error:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    # Convert to response models
    return [to_chunk_response(chunk) for chunk in results]
