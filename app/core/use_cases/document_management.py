from typing import Optional

from .collection_management import CollectionManagementUseCase
from ..ports.repositories import DocumentRepository
from ..ports.vector_store import VectorStore
from ..ports.storage_service import StorageService
from ..ports.logger import Logger
from ..domain.exceptions import (
    DocumentNotFoundError,
    PartialDocumentDeletionError,
)

class DocumentManagementUseCase:
    """Use case for managing documents"""

    def __init__(
            self,
            document_repo: DocumentRepository,
            vector_store: VectorStore,
            storage_service: StorageService,
            logger: Logger,
            collection_management_use_case: CollectionManagementUseCase = None,
        ):
        self.document_repo = document_repo
        self.vector_store = vector_store
        self.storage_service = storage_service
        self.logger = logger
        self._collection_management_use_case = collection_management_use_case

    def set_collection_management_use_case(self, collection_management_use_case: CollectionManagementUseCase):
        """Setter to avoid circular dependency issues"""
        self._collection_management_use_case = collection_management_use_case

    async def get_document_by_id(self,document_id: str):
        document = await self.document_repo.find_by_id(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return document
    
    async def list_documents(self, collection_id: Optional[str]):
        if collection_id:
            documents = await self.document_repo.find_by_collection_id(collection_id)
        else:
            documents = await self.document_repo.find_all()

        return documents
    
    async def delete_document(self, document_id: str):
        """
        Delete a document: remove from vector store and delete DB record and file.
        Only returns success if ALL components are deleted successfully.
        """
        # 1. Fetch document
        document = await self.document_repo.find_by_id(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        
        collection_id = document.collection_id
        
        # Track deletion results
        deletion_results = {
            "vector_store": {"success": False, "error": None},
            "file": {"success": False, "error": None},
            "database": {"success": False, "error": None},
            "collection_count": {"success": False, "error": None}
        }

        # 2. Delete from vector store
        try:
            await self.vector_store.delete_by_document_id(document_id)
            deletion_results["vector_store"]["success"] = True
        except Exception as e:
            deletion_results["vector_store"]["error"] = str(e)
            self.logger.error(f"Failed to delete chunks from vector store for {document_id}: {e}")


        # 3. Delete file on disk
        try:
            file_path = document.file_path
            file_exists = await self.storage_service.file_exists(file_path)
            if file_exists:
                await self.storage_service.delete_file(file_path)
                deletion_results["file"]["success"] = True
            else:
                # File doesn't exist, consider it "successfully deleted"
                deletion_results["file"]["success"] = True
                self.logger.info(f"File {document.file_path} already doesn't exist")
        except Exception as e:
            deletion_results["file"]["error"] = str(e)
            self.logger.error(f"Failed to delete file {document.file_path}: {e}")

        # 4. Delete from repository
        try:
            success = await self.document_repo.delete(document_id)
            if success:
                deletion_results["database"]["success"] = True
            else:
                deletion_results["database"]["error"] = "Repository delete returned False"
        except Exception as e:
            deletion_results["database"]["error"] = str(e)
            self.logger.error(f"Failed to delete document record {document_id}: {e}")


        # 5. Decrement collection document count
        try:
            if self._collection_management_use_case:
                await self._collection_management_use_case.decrement_document_count(collection_id)
                deletion_results["collection_count"]["success"] = True
            else:
                self.logger.warning("Collection management use case not available, skipping count decrement")
                deletion_results["collection_count"]["success"] = True  # Don't fail for this
        except Exception as e:
            deletion_results["collection_count"]["error"] = str(e)
            self.logger.error(f"Failed to decrement document count for collection {collection_id}: {e}")
        
        # Check if ALL deletions were successful
        all_successful = all(result["success"] for result in deletion_results.values())
        
        if not all_successful:
            # Collect all errors
            errors = []
            failed_operations = []
            for component, result in deletion_results.items():
                if not result["success"]:
                    errors.append(f"{component}: {result['error']}")
                    failed_operations.append(component)
            
            error_message = f"Failed to delete document completely. Errors: {'; '.join(errors)}"
            raise PartialDocumentDeletionError(error_message, deletion_results, failed_operations)
        
        # All deletions successful
        return {
            "message": "Document deleted successfully from all systems",
            "document_id": document_id,
            "document_name": document.filename,
            "deleted_from": ["vector_store", "local_disk", "database"]
        }
