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


class SimplifiedDocumentProcessingUseCase:
    """Simplified use case for processing and indexing documents"""

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
        Process a document: extract text, create chunks, optionally add context, generate embeddings, and index
        """
        try:
            self.logger.info(
                f"Starting simplified processing for document {document.id} ({document.original_filename})")

            # Mark document as processing
            document.mark_processing()
            await self.document_repo.save(document)
            self.logger.info(f"Document {document.id} marked as processing")

            # Extract text and create basic chunks
            self.logger.info(f"Extracting text and creating chunks for document {document.id}")
            try:
                doc_chunks, complete_text = await self.document_processor.process_document(document)
                self.logger.info(f"Created {len(doc_chunks)} initial chunks from document {document.id}")
            except Exception as e:
                self.logger.error(f"Failed to process document for {document.id}: {str(e)}")
                self.logger.error(f"Document processor error traceback: {traceback.format_exc()}")
                raise DocumentProcessingError(f"Document processing failed: {str(e)}")

            if not doc_chunks:
                error_msg = "No chunks created from document"
                self.logger.error(f"{error_msg} for document {document.id}")
                raise DocumentProcessingError(error_msg)

            # Add contextual information if enabled
            if settings.ENABLE_CONTEXTUAL_RETRIEVAL and self.contextual_llm:
                self.logger.info(f"Adding contextual information to {len(doc_chunks)} chunks")
                try:
                    contextual_chunks = await self._add_contextual_information_batch(
                        doc_chunks, complete_text, document.id
                    )
                    self.logger.info(f"Successfully added context to chunks for document {document.id}")
                except Exception as e:
                    self.logger.warning(f"Failed to add contextual information for {document.id}: {str(e)}")
                    # Continue with original chunks if contextual processing fails
                    contextual_chunks = doc_chunks
            else:
                contextual_chunks = doc_chunks
                self.logger.info("Contextual retrieval disabled, using original chunks")

            # Generate embeddings for all chunks in batch
            self.logger.info(f"Generating embeddings for {len(contextual_chunks)} chunks")
            try:
                contextual_chunks = await self._generate_embeddings_batch(contextual_chunks, document.id)
                self.logger.info(f"Successfully generated embeddings for document {document.id}")
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for {document.id}: {str(e)}")
                self.logger.error(f"Embedding error traceback: {traceback.format_exc()}")
                raise DocumentProcessingError(f"Embedding generation failed: {str(e)}")

            # Index chunks in both stores
            await self._index_chunks_in_stores(contextual_chunks, document.id)

            # Mark document as processed
            self.logger.info(f"Marking document {document.id} as processed with {len(contextual_chunks)} chunks")
            document.mark_processed(len(contextual_chunks))
            result = await self.document_repo.save(document)

            # Update collection document count
            try:
                await self._increment_collection_document_count(document.collection_id)
                self.logger.info(f"Incremented document count for collection {document.collection_id}")
            except Exception as e:
                self.logger.error(f"Failed to update collection document count: {str(e)}")
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
            try:
                if document.status.value == "processing":
                    self.logger.warning(f"Document {document.id} still in processing state, marking as failed")
                    document.mark_failed()
                    await self.document_repo.save(document)
            except Exception as cleanup_error:
                self.logger.error(f"Failed to cleanup document {document.id} state: {cleanup_error}")

    async def _generate_embeddings_batch(
            self,
            chunks: List[DocumentChunk],
            document_id: str
    ) -> List[DocumentChunk]:
        """Generate embeddings for all chunks in batch for efficiency"""
        try:
            # Extract texts for batch processing
            chunk_texts = [chunk.content for chunk in chunks]

            # Generate embeddings in batch
            embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)

            if len(embeddings) != len(chunks):
                raise DocumentProcessingError(
                    f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
                )

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_vector = embedding.values

            self.logger.debug(f"Assigned embeddings to {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings batch for {document_id}: {str(e)}")
            raise DocumentProcessingError(f"Batch embedding generation failed: {str(e)}")

    async def _add_contextual_information_batch(
            self,
            chunks: List[DocumentChunk],
            full_document_text: str,
            document_id: str
    ) -> List[DocumentChunk]:
        """Add contextual information to chunks with better error handling and batching"""
        try:
            contextualized_chunks: List[DocumentChunk] = []

            # Process chunks in smaller batches to avoid overwhelming the LLM
            batch_size = getattr(settings, 'CONTEXTUAL_BATCH_SIZE', 5)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.logger.debug(f"Processing contextual batch {i // batch_size + 1} for document {document_id}")

                batch_tasks = []
                for chunk in batch:
                    task = self._add_context_to_chunk(chunk, full_document_text)
                    batch_tasks.append(task)

                try:
                    # Process batch with timeout
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=getattr(settings, 'CONTEXTUAL_TIMEOUT', 60)
                    )

                    for chunk, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            self.logger.warning(f"Failed to add context to chunk {chunk.id}: {result}")
                            # Use original chunk if context generation fails
                            contextualized_chunks.append(chunk)
                        else:
                            contextualized_chunks.append(result)

                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout processing contextual batch for document {document_id}")
                    # Add original chunks without context
                    contextualized_chunks.extend(batch)
                except Exception as e:
                    self.logger.warning(f"Error processing contextual batch for document {document_id}: {e}")
                    # Add original chunks without context
                    contextualized_chunks.extend(batch)

                # Add small delay between batches to avoid overwhelming the LLM
                await asyncio.sleep(0.1)

            return contextualized_chunks

        except Exception as e:
            self.logger.error(f"Failed to add contextual information for document {document_id}: {str(e)}")
            # Return original chunks if contextual processing completely fails
            return chunks

    async def _add_context_to_chunk(
            self,
            chunk: DocumentChunk,
            full_document_text: str
    ) -> DocumentChunk:
        """Add contextual information to a single chunk"""
        try:
            # Create a more efficient context prompt
            context_prompt = self._create_context_prompt(chunk.content, full_document_text)

            # Generate context with timeout
            context = await asyncio.wait_for(
                self.contextual_llm.generate_response(context_prompt),
                timeout=getattr(settings, 'CONTEXTUAL_CHUNK_TIMEOUT', 30)
            )

            if context and context.strip():
                # Prepend context to chunk content
                contextualized_content = f"{context.strip()}\n\n{chunk.content}"
                chunk.content = contextualized_content

                # Update metadata
                chunk.metadata = {
                    **chunk.metadata,
                    'context_prefix': context.strip(),
                    'has_context': True
                }
            else:
                # Mark that context generation was attempted but empty
                chunk.metadata = {
                    **chunk.metadata,
                    'context_prefix': '',
                    'has_context': False
                }

            return chunk

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout generating context for chunk {chunk.id}")
            chunk.metadata = {**chunk.metadata, 'context_prefix': '', 'has_context': False}
            return chunk
        except Exception as e:
            self.logger.warning(f"Failed to generate context for chunk {chunk.id}: {e}")
            chunk.metadata = {**chunk.metadata, 'context_prefix': '', 'has_context': False}
            return chunk

    def _create_context_prompt(self, chunk_content: str, full_document: str) -> str:
        """Create an optimized prompt for context generation"""
        # Truncate full document if it's too long to avoid token limits
        max_doc_length = getattr(settings, 'CONTEXTUAL_DOC_MAX_LENGTH', 8000)

        if len(full_document) > max_doc_length:
            # Take beginning and end of document for context
            half_length = max_doc_length // 2
            truncated_doc = full_document[:half_length] + "\n\n[...document continues...]\n\n" + full_document[
                                                                                                 -half_length:]
        else:
            truncated_doc = full_document

        prompt = f"""Document content:
{truncated_doc}

Chunk to contextualize:
{chunk_content}

Provide a brief context (1-2 sentences) that situates this chunk within the overall document for better search retrieval. Focus on what this section is about and how it relates to the document's main topics.

Context:"""

        return prompt

    async def _index_chunks_in_stores(self, chunks: List[DocumentChunk], document_id: str) -> None:
        """Index chunks in both vector store and BM25 index with better error handling"""
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
            raise DocumentProcessingError(f"BM25 indexing failed: {str(e)}")

    # Keep existing methods for document management
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

        # Get chunk IDs before deletion
        try:
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
            try:
                await self._decrement_collection_document_count(collection_id)
                self.logger.info(f"Decremented document count for collection {collection_id}")
            except Exception as e:
                self.logger.error(f"Failed to update collection document count: {str(e)}")

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
