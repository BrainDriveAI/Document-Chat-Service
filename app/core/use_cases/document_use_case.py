import asyncio
import logging
import traceback
from typing import List, Optional

from ...config import settings
from ..domain.entities.document import Document, DocumentType
from ..domain.entities.document_chunk import DocumentChunk
from ..domain.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    InvalidDocumentTypeError
)
from ..ports.document_processor import DocumentProcessor
from ..ports.embedding_service import EmbeddingService
from ..ports.vector_store import VectorStore
from ..ports.repositories import DocumentRepository, CollectionRepository
from ..ports.llm_service import LLMService
from ..ports.bm25_service import BM25Service


class DocumentProcessingUseCase:
    """Use case for processing and indexing documents"""

    def __init__(
            self,
            document_repo: DocumentRepository,
            document_processor: DocumentProcessor,
            embedding_service: EmbeddingService,
            vector_store: VectorStore,
            collection_repo: CollectionRepository,
            llm_service: LLMService,
            contextual_llm: Optional[LLMService],
            bm25_service: BM25Service,
    ):
        self.document_repo = document_repo
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.collection_repo = collection_repo
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service
        self.contextual_llm = contextual_llm
        self.bm25_service = bm25_service

    async def process_document(self, document: Document) -> Document:
        """
        Process a document: extract structure, generate embeddings, store in vector DB and BM25 index
        """
        try:
            self.logger.info(f"Starting processing for document {document.id} ({document.original_filename})")
            # Mark document as processing
            document.mark_processing()
            await self.document_repo.save(document)
            self.logger.info(f"Document {document.id} marked as processing")

            # Process document structure
            self.logger.info(f"Processing document structure for {document.id}")
            try:
                doc_chunks, complete_text = await self.document_processor.process_document(document)
                self.logger.info(f"Extracted {len(doc_chunks) if doc_chunks else 0} chunks from document {document.id}")
            except Exception as e:
                self.logger.error(f"Failed to process document structure for {document.id}: {str(e)}")
                self.logger.error(f"Document processor error traceback: {traceback.format_exc()}")
                raise DocumentProcessingError(f"Document structure processing failed: {str(e)}")

            if not doc_chunks:
                error_msg = "No chunks extracted from document"
                self.logger.error(f"{error_msg} for document {document.id}")
                raise DocumentProcessingError(error_msg)

            # Generate contextual information
            if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
                contextual_chunks = await self._add_contextual_information(doc_chunks, complete_text)
            else:
                contextual_chunks = doc_chunks

            # Generate embeddings for chunks
            self.logger.info(f"Generating embeddings for {len(contextual_chunks)} chunks from document {document.id}")
            try:
                chunk_texts = [chunk.content for chunk in contextual_chunks]
                self.logger.debug(f"First chunk preview for {document.id}: {chunk_texts[0][:100]}...")
                embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)
                self.logger.info(f"Generated {len(embeddings)} embeddings for document {document.id}")
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for {document.id}: {str(e)}")
                self.logger.error(f"Embedding error traceback: {traceback.format_exc()}")
                raise DocumentProcessingError(f"Embedding generation failed: {str(e)}")

            # Add embeddings to chunks
            self.logger.info(f"Adding embeddings to chunks for document {document.id}")
            try:
                for i, (chunk, embedding) in enumerate(zip(contextual_chunks, embeddings)):
                    chunk.embedding_vector = embedding.values
                    self.logger.debug(
                        f"Added embedding to chunk {i + 1}/{len(contextual_chunks)} for document {document.id}")
            except Exception as e:
                self.logger.error(f"Failed to add embeddings to chunks for {document.id}: {str(e)}")
                raise DocumentProcessingError(f"Failed to add embeddings to chunks: {str(e)}")

            # Store chunks in both vector database and BM25 index
            await self._index_chunks_in_stores(contextual_chunks, document.id)

            # Mark document as processed
            self.logger.info(f"Marking document {document.id} as processed with {len(contextual_chunks)} chunks")
            document.mark_processed(len(contextual_chunks))
            result = await self.document_repo.save(document)

            # Update collection document count after successful processing
            try:
                await self._increment_collection_document_count(document.collection_id)
                self.logger.info(f"Incremented document count for collection {document.collection_id}")
            except Exception as e:
                self.logger.error(f"Failed to update collection document count for {document.collection_id}: {str(e)}")
                # Don't fail the entire operation for count update failure

            self.logger.info(f"Successfully processed document {document.id} with {len(contextual_chunks)} chunks")
            return result

        except DocumentProcessingError:
            # Re-raise DocumentProcessingError as-is
            raise
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected error processing document {document.id}: {str(e)}")
            self.logger.error(f"Unexpected error traceback: {traceback.format_exc()}")
            raise DocumentProcessingError(f"Unexpected processing error: {str(e)}")
        finally:
            # Ensure document is marked as failed if we get here due to an exception
            # This will only execute if an exception occurred above
            try:
                if document.status.value == "processing":
                    self.logger.warning(f"Document {document.id} still in processing state, marking as failed")
                    document.mark_failed()
                    await self.document_repo.save(document)
            except Exception as cleanup_error:
                self.logger.error(f"Failed to cleanup document {document.id} state: {cleanup_error}")

    async def _index_chunks_in_stores(self, chunks: List[DocumentChunk], document_id: str) -> None:
        """Index chunks in both vector store and BM25 index"""
        # Store in vector database
        self.logger.info(f"Storing {len(chunks)} chunks in vector database for document {document_id}")
        try:
            await self.vector_store.add_chunks(chunks)
            self.logger.info(f"Successfully stored chunks in vector database for document {document_id}")
        except Exception as e:
            self.logger.error(f"Failed to store chunks in vector database for {document_id}: {str(e)}")
            self.logger.error(f"Vector store error traceback: {traceback.format_exc()}")
            raise DocumentProcessingError(f"Vector database storage failed: {str(e)}")

        # Index in BM25
        self.logger.info(f"Indexing {len(chunks)} chunks in BM25 for document {document_id}")
        try:
            success = await self.bm25_service.index_chunks(chunks)
            if not success:
                raise DocumentProcessingError("BM25 indexing returned failure status")
            self.logger.info(f"Successfully indexed chunks in BM25 for document {document_id}")
        except Exception as e:
            self.logger.error(f"Failed to index chunks in BM25 for {document_id}: {str(e)}")
            self.logger.error(f"BM25 indexing error traceback: {traceback.format_exc()}")
            # Consider whether BM25 failure should fail the entire operation
            # For now, let's make it fail since we want both indexes to be in sync
            raise DocumentProcessingError(f"BM25 indexing failed: {str(e)}")

    async def _add_contextual_information(
            self,
            chunks: List[DocumentChunk],
            full_document_text: str
    ) -> List[DocumentChunk]:
        """Add contextual information to chunks using LLM"""

        print(f"[[[[[[[[[[ FULL DOCUMENT: {full_document_text}]]]]]]]]]]")

        contextualized_chunks: List[DocumentChunk] = []

        for chunk in chunks:
            # Generate contextual information
            prompt = f"""
            {full_document_text}

            Here is the chunk we want to situate within the whole document:
            {chunk.content}

            Please give a short succinct context to situate this chunk within the overall document
            for the purposes of improving search retrieval of the chunk.
            Answer only with the succinct context and nothing else.
            """

            try:
                context = await self.contextual_llm.generate_response(prompt)
                contextualized_content = f"{context.strip()} {chunk.content}"

                print(f">>>>>>>>Contextual content: {contextualized_content}")

                chunk.content = contextualized_content

                # Update chunk metadata with contextual content
                chunk.metadata = {
                    **chunk.metadata,
                    'context_prefix': context.strip()
                }

            except Exception as e:
                # Fall back to original content if context generation fails
                chunk.metadata = {
                    **chunk.metadata,
                    'context_prefix': ''
                }

            contextualized_chunks.append(chunk)

        return contextualized_chunks

    async def get_document(self, document_id: str) -> Document:
        """Get a document by ID"""
        document = await self.document_repo.find_by_id(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return document

    async def list_documents_by_collection(self, collection_id: str) -> List[Document]:
        """List all documents in a collection"""
        return await self.document_repo.find_by_collection_id(collection_id)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from all indexes"""
        document = await self.get_document(document_id)
        collection_id = document.collection_id

        # Get chunk IDs before deletion (needed for BM25 removal)
        try:
            # Assuming we can get chunks by document_id from vector store
            # You might need to add this method to your vector store port if not available
            chunks = await self.vector_store.get_chunks_by_document_id(document_id)
            chunk_ids = [chunk.id for chunk in chunks] if chunks else []
        except Exception as e:
            self.logger.warning(f"Failed to get chunk IDs for document {document_id}: {e}")
            chunk_ids = []

        # Delete from vector store
        try:
            await self.vector_store.delete_by_document_id(document_id)
            self.logger.info(f"Deleted chunks from vector store for document {document_id}")
        except Exception as e:
            self.logger.warning(f"Failed to delete chunks from vector store for {document_id}: {e}")

        # Delete from BM25 index
        if chunk_ids:
            try:
                success = await self.bm25_service.remove_chunks(chunk_ids)
                if success:
                    self.logger.info(f"Deleted {len(chunk_ids)} chunks from BM25 for document {document_id}")
                else:
                    self.logger.warning(f"BM25 removal returned false for document {document_id}")
            except Exception as e:
                self.logger.warning(f"Failed to delete chunks from BM25 for {document_id}: {e}")

        # Delete from repository
        success = await self.document_repo.delete(document_id)
        if success:
            self.logger.info(f"Successfully deleted document {document_id}")

            # Decrement collection document count after successful deletion
            try:
                await self._decrement_collection_document_count(collection_id)
                self.logger.info(f"Decremented document count for collection {collection_id}")
            except Exception as e:
                self.logger.error(f"Failed to update collection document count for {collection_id}: {str(e)}")
                # Don't fail the entire operation for count update failure
        else:
            self.logger.error(f"Failed to delete document {document_id} from repository")

        return success

    async def _increment_collection_document_count(self, collection_id: str) -> None:
        """Helper method to increment collection document count"""
        collection = await self.collection_repo.find_by_id(collection_id)
        if collection:
            collection.increment_document_count()
            await self.collection_repo.save(collection)

    async def _decrement_collection_document_count(self, collection_id: str) -> None:
        """Helper method to decrement collection document count"""
        collection = await self.collection_repo.find_by_id(collection_id)
        if collection:
            collection.decrement_document_count()
            await self.collection_repo.save(collection)
