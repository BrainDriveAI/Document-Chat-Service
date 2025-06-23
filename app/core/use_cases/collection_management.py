from typing import List
from datetime import datetime, UTC
from ..domain.entities.collection import Collection
from ..domain.exceptions import CollectionNotFoundError
from ..ports.repositories import CollectionRepository


class CollectionManagementUseCase:
    """Use case for managing document collections"""

    def __init__(self, collection_repo: CollectionRepository):
        self.collection_repo = collection_repo

    async def create_collection(
            self,
            name: str,
            description: str,
            color: str = "#3B82F6"
    ) -> Collection:
        """Create a new document collection"""
        collection = Collection.create(name=name, description=description, color=color)
        return await self.collection_repo.save(collection)

    async def get_collection(self, collection_id: str) -> Collection:
        """Get a collection by ID"""
        collection = await self.collection_repo.find_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")
        return collection

    async def get_collection_with_documents(self, collection_id: str) -> Collection:
        """Get a collection by ID along with all its associated documents"""
        collection = await self.collection_repo.find_by_id_with_documents(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")
        return collection

    async def list_collections(self) -> List[Collection]:
        """List all collections"""
        return await self.collection_repo.find_all()

    async def update_collection(
            self,
            collection_id: str,
            name: str = None,
            description: str = None,
            color: str = None
    ) -> Collection:
        """Update an existing collection"""
        collection = await self.get_collection(collection_id)
        if name:
            collection.name = name
        if description:
            collection.description = description
        if color:
            collection.color = color
        collection.updated_at = datetime.now(UTC)
        return await self.collection_repo.save(collection)

    async def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        collection = await self.get_collection(collection_id)
        return await self.collection_repo.delete(collection_id)

    async def increment_document_count(self, collection_id: str) -> None:
        """Increment the document count for a collection"""
        collection = await self.get_collection(collection_id)
        collection.increment_document_count()
        await self.collection_repo.save(collection)

    async def decrement_document_count(self, collection_id: str) -> None:
        """Decrement the document count for a collection"""
        collection = await self.get_collection(collection_id)
        collection.decrement_document_count()
        await self.collection_repo.save(collection)
