import shutil
import logging
import traceback
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ...api.deps import (
    get_document_repository,
    get_collection_repository,
    get_document_processing_use_case,
    get_vector_store,
    get_storage_service,
)

from ...core.use_cases.simple_document import SimplifiedDocumentProcessingUseCase
from ...core.ports.vector_store import VectorStore
from ...core.ports.storage_service import StorageService
from ...core.domain.entities.document import Document as DomainDocument, DocumentType
from ...core.domain.exceptions import InvalidDocumentTypeError

from ...config import settings

from ...adapters.persistence.sqlite_repository import SQLiteDocumentRepository, SQLiteCollectionRepository

router = APIRouter()

# Set up logger for this module
logger = logging.getLogger(__name__)


# Pydantic models for responses
class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    document_type: str
    collection_id: str
    status: str
    created_at: datetime
    processed_at: Optional[datetime]
    metadata: dict
    chunk_count: int


def to_document_response(doc: DomainDocument) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_path=doc.file_path,
        file_size=doc.file_size,
        document_type=doc.document_type.value,
        collection_id=doc.collection_id,
        status=doc.status.value,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
        metadata=doc.metadata or {},
        chunk_count=doc.chunk_count
    )


def determine_document_type(filename: str) -> DocumentType:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "pdf":
        return DocumentType.PDF
    elif ext == "docx":
        return DocumentType.DOCX
    elif ext == "doc":
        return DocumentType.DOC
    else:
        raise InvalidDocumentTypeError(f"Unsupported file extension: {ext}")


async def process_document_background(doc_use_case: SimplifiedDocumentProcessingUseCase, document: DomainDocument):
    """
    Background task to process document.
    """
    try:
        logger.info(f"Starting background processing for document {document.id}")
        await doc_use_case.process_document(document)
        logger.info(f"Successfully processed document {document.id}")
    except Exception as e:
        logger.error(f"Background document processing failed for {document.id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # The use case should already mark the document as failed, but let's ensure it
        try:
            document.mark_failed()
            # We need to access the repository to save the failed state
            # This is a limitation of the current design - we might need to pass the repo
            logger.error(f"Document {document.id} marked as failed due to processing error")
        except Exception as save_error:
            logger.error(f"Failed to mark document {document.id} as failed: {save_error}")


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        collection_id: str = Form(...),
        document_repo: SQLiteDocumentRepository = Depends(get_document_repository),
        collection_repo: SQLiteCollectionRepository = Depends(get_collection_repository),
        doc_use_case: SimplifiedDocumentProcessingUseCase = Depends(get_document_processing_use_case),
        vector_store=Depends(get_vector_store),
        storage_service: StorageService = Depends(get_storage_service),
):
    """
    Upload a document file to a collection, save it, and trigger background processing.
    """
    logger.info(f"[Upload Debug] Received file: {file.filename}, size: {file.size}")
    logger.info(f"[Upload Debug] Collection ID: {collection_id}")

    # 1. Validate collection exists
    coll = await collection_repo.find_by_id(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")

    # 2. Determine type
    try:
        doc_type = determine_document_type(file.filename)
    except InvalidDocumentTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. Save file to disk
    # uploads_base = Path(settings.UPLOADS_DIR)
    # collection_dir = uploads_base / collection_id
    # collection_dir.mkdir(parents=True, exist_ok=True)

    # import uuid
    # new_id = str(uuid.uuid4())
    # ext = Path(file.filename).suffix.lower()
    # saved_filename = f"{new_id}{ext}"
    # saved_path = collection_dir / saved_filename

    # try:
    #     # Write file
    #     with open(saved_path, "wb") as out_file:
    #         shutil.copyfileobj(file.file, out_file)
    #     logger.info(f"File saved to: {saved_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save file: {e}")
    #     raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    # finally:
    #     file.file.close()

    # 3. Save file using storage service
    try:
        saved_path = await storage_service.save_file(
            file_content=file.file,
            collection_id=collection_id,
            filename=file.filename
        )
        logger.info(f"File saved to: {saved_path}")
        
        # Get file size from storage service
        file_size = await storage_service.get_file_size(saved_path)
        if file_size is None:
            raise HTTPException(status_code=500, detail="Failed to get file size after saving")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        file.file.close()

    # 4. Create DomainDocument entity
    saved_filename = Path(saved_path).name
    # file_stat = saved_path.stat()
    domain_doc = DomainDocument.create(
        filename=saved_filename,
        original_filename=file.filename,
        file_path=str(saved_path),
        file_size=file_size,
        document_type=doc_type,
        collection_id=collection_id,
        metadata={}
    )

    logger.info(f"Created domain document: {domain_doc.id}")

    # 5. Save to repository (status = UPLOADED initially)
    try:
        saved = await document_repo.save(domain_doc)
        logger.info(f"Document saved to database: {saved.id} with status: {saved.status}")
    except Exception as e:
        logger.error(f"Failed to save document to database: {e}")
        # Cleanup file if DB save fails
        try:
            await storage_service.delete_file(saved_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save document record: {e}")

    # 6. Trigger background processing
    logger.info(f"Adding background task for document {saved.id}")
    background_tasks.add_task(process_document_background, doc_use_case, saved)

    return to_document_response(saved)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
        document_id: str,
        document_repo: SQLiteDocumentRepository = Depends(get_document_repository)
):
    """
    Get document info and status.
    """
    doc = await document_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return to_document_response(doc)


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
        collection_id: Optional[str] = None,
        document_repo: SQLiteDocumentRepository = Depends(get_document_repository),
        vector_store: VectorStore = Depends(get_vector_store)
):
    """
    List documents, optionally filtered by collection_id.
    """
    if collection_id:
        docs = await document_repo.find_by_collection_id(collection_id)
    else:
        docs = await document_repo.find_all()
        # raise HTTPException(status_code=400, detail="collection_id query parameter is required")
    return [to_document_response(d) for d in docs]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
        document_id: str,
        document_repo: SQLiteDocumentRepository = Depends(get_document_repository),
        vector_store: VectorStore = Depends(get_vector_store),
        storage_service: StorageService = Depends(get_storage_service),
):
    """
    Delete a document: remove from vector store and delete DB record and file.
    Only returns success if ALL components are deleted successfully.
    """
    # 1. Fetch document
    doc = await document_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    # Track deletion results
    deletion_results = {
        "vector_store": {"success": False, "error": None},
        "file": {"success": False, "error": None},
        "database": {"success": False, "error": None}
    }
    
    # 2. Delete from vector store
    try:
        await vector_store.delete_by_document_id(document_id)
        deletion_results["vector_store"]["success"] = True
    except Exception as e:
        deletion_results["vector_store"]["error"] = str(e)
        logger.error(f"Failed to delete chunks from vector store for {document_id}: {e}")
    
    # 3. Delete file on disk
    try:
        saved_path = Path(doc.file_path)
        if saved_path.exists():
            await storage_service.delete_file(saved_path)
            deletion_results["file"]["success"] = True
        else:
            # File doesn't exist, consider it "successfully deleted"
            deletion_results["file"]["success"] = True
            logger.info(f"File {doc.file_path} already doesn't exist")
    except Exception as e:
        deletion_results["file"]["error"] = str(e)
        logger.error(f"Failed to delete file {doc.file_path}: {e}")
    
    # 4. Delete from repository
    try:
        success = await document_repo.delete(document_id)
        if success:
            deletion_results["database"]["success"] = True
        else:
            deletion_results["database"]["error"] = "Repository delete returned False"
    except Exception as e:
        deletion_results["database"]["error"] = str(e)
        logger.error(f"Failed to delete document record {document_id}: {e}")
    
    # Check if ALL deletions were successful
    all_successful = all(result["success"] for result in deletion_results.values())
    
    if not all_successful:
        # Collect all errors
        errors = []
        for component, result in deletion_results.items():
            if not result["success"]:
                errors.append(f"{component}: {result['error']}")
        
        error_message = f"Failed to delete document completely. Errors: {'; '.join(errors)}"
        raise HTTPException(status_code=500, detail=error_message)
    
    # All deletions successful
    return {
        "message": "Document deleted successfully from all systems",
        "document_id": document_id,
        "document_name": doc.filename,
        "deleted_from": ["vector_store", "local_disk", "database"]
    }

