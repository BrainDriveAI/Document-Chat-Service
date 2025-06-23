from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from datetime import datetime

from ...api.deps import get_collection_repository, get_collection_management_use_case
from ...adapters.persistence.sqlite_repository import SQLiteCollectionRepository
from ...core.domain.entities.collection import Collection as DomainCollection
from ...core.domain.exceptions import CollectionNotFoundError
from ...core.use_cases.collection_management import CollectionManagementUseCase
from pydantic import BaseModel, Field

router = APIRouter()


# Pydantic schemas for request/response

class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    color: Optional[str] = Field(None, pattern=r"^#(?:[0-9a-fA-F]{3}){1,2}$")  # e.g., "#3B82F6"


class CollectionUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = Field(None, min_length=1)
    color: Optional[str] = Field(None, pattern=r"^#(?:[0-9a-fA-F]{3}){1,2}$")


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    color: str
    created_at: datetime
    updated_at: datetime
    document_count: int


def to_response_model(domain: DomainCollection) -> CollectionResponse:
    return CollectionResponse(
        id=domain.id,
        name=domain.name,
        description=domain.description,
        color=domain.color,
        created_at=domain.created_at,
        updated_at=domain.updated_at,
        document_count=domain.document_count
    )


@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CollectionCreateRequest,
    use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case)
):
    """
    Create a new collection.
    """
    try:
        collection = await use_case.create_collection(
            name=body.name,
            description=body.description,
            color=body.color or "#3B82F6"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")
    return to_response_model(collection)


@router.get("/", response_model=List[CollectionResponse])
async def list_collections(
    use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case)
):
    """
    List all collections.
    """
    try:
        collections = await use_case.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
    return [to_response_model(c) for c in collections]


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case)
):
    """
    Get a collection by ID.
    """
    try:
        collection = await use_case.get_collection(collection_id)
    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")
    return to_response_model(collection)


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    body: CollectionUpdateRequest,
    use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case)
):
    """
    Update an existing collection.
    """
    try:
        updated = await use_case.update_collection(
            collection_id=collection_id,
            name=body.name,
            description=body.description,
            color=body.color
        )
    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update collection: {str(e)}")
    return to_response_model(updated)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: str,
    use_case: CollectionManagementUseCase = Depends(get_collection_management_use_case),
    collection_repo: SQLiteCollectionRepository = Depends(get_collection_repository)
):
    """
    Delete a collection. Note: documents in it should also be deleted or handled separately.
    """
    # Optionally: before deletion, you may check and delete all documents in this collection.
    # For now, just delete the collection record.
    try:
        # Confirm it exists
        _ = await use_case.get_collection(collection_id)
    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collection: {str(e)}")

    try:
        success = await use_case.delete_collection(collection_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")
    if not success:
        raise HTTPException(status_code=500, detail="Collection deletion reported failure")
    # Optionally: cascade delete documents or return info
    return
