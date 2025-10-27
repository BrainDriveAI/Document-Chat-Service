import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union
import chromadb
from chromadb.config import Settings
from ...core.ports.vector_store import VectorStore
from ...core.domain.entities.document_chunk import DocumentChunk
from ...core.domain.value_objects.embedding import EmbeddingVector, SearchQuery


class ChromaVectorStoreAdapter(VectorStore):
    """
    Chroma-based implementation of the VectorStore port using the new PersistentClient API.
    """

    def __init__(
            self,
            persist_directory: str,
            collection_name: str = "documents",
    ):
        """
        Args:
            persist_directory: Path to persist Chroma DB files (local folder).
            collection_name: Name of the Chroma collection to use.
        """
        # Thread pool executor for wrapping sync calls
        self._executor = ThreadPoolExecutor()

        # Instantiate PersistentClient for local storage
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get the named collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 100,
                    "ef_construction": 100,
                    "max_neighbors": 16,
                }
            }
        )

    def _sanitize_metadata_value(self, value: Any) -> Union[str, int, float, bool]:
        """
        Sanitize metadata values to be compatible with ChromaDB.
        ChromaDB only accepts str, int, float, bool (NO None values).
        """
        if value is None:
            return ""
        elif isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        elif isinstance(value, str):
            return value
        elif hasattr(value, '__dict__'):  # Handle spaCy objects
            return str(value)
        elif isinstance(value, (list, tuple)):
            # Convert lists to comma-separated strings, filtering None values
            items = [str(item) for item in value if item is not None]
            return ",".join(items)
        elif isinstance(value, dict):
            # Convert dicts to string, ensuring nested values are also serializable
            return str({k: self._sanitize_metadata_value(v) for k, v in value.items()})
        else:
            return str(value)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool, None]]:
        """
        Sanitize all metadata values to be compatible with ChromaDB.
        """
        sanitized = {}
        for key, value in metadata.items():
            sanitized[key] = self._sanitize_metadata_value(value)
        return sanitized

    def _build_chroma_where_filter(self, filters: Optional[Dict] = None, collection_id: Optional[str] = None) -> \
            Optional[Dict[str, Any]]:
        """
        Build ChromaDB-compatible where filter, excluding similarity-based filters.

        Args:
            filters: Raw filters from the request
            collection_id: Collection ID to filter by

        Returns:
            ChromaDB-compatible where filter or None
        """
        where: Dict[str, Any] = {}

        # Add collection_id filter
        if collection_id:
            where["collection_id"] = collection_id

        if filters:
            for key, value in filters.items():
                # Skip similarity-based filters as they're handled separately
                if key in ["min_similarity", "max_similarity", "similarity_threshold"]:
                    continue

                # Handle file_type filter (array of file types)
                if key == "file_type" and isinstance(value, list):
                    # ChromaDB uses $in operator for array matching
                    where["file_type"] = {"$in": value}
                else:
                    where[key] = value

        # Return None if empty to avoid ChromaDB errors
        return where if where else None

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks (with embedding_vector already set) to Chroma.
        Wrap synchronous calls in executor.
        """
        if not chunks:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._add_chunks_sync, chunks)

    def _add_chunks_sync(self, chunks: List[DocumentChunk]) -> bool:
        """
        Synchronous helper to add chunks to Chroma.
        """
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for chunk in chunks:
            if not chunk.embedding_vector:
                raise ValueError(f"Chunk {chunk.id} has no embedding_vector set")

            ids.append(chunk.id)

            if isinstance(chunk.embedding_vector, EmbeddingVector):
                vec = chunk.embedding_vector.values
            else:
                vec = chunk.embedding_vector  # assume List[float]
            embeddings.append(vec)
            documents.append(chunk.content)

            # Build metadata with sanitization
            meta = {}
            if chunk.metadata:
                meta.update(chunk.metadata)

            # Add chunk-specific metadata
            meta.update({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "collection_id": chunk.collection_id,
                "chunk_index": getattr(chunk, "chunk_index", None),
                "chunk_type": getattr(chunk, "chunk_type", None),
                "parent_chunk_id": getattr(chunk, "parent_chunk_id", None),
            })

            # Sanitize metadata to ensure ChromaDB compatibility
            sanitized_meta = self._sanitize_metadata(meta)
            metadatas.append(sanitized_meta)

        # Use collection.add; new Chroma persists automatically
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        return True

    async def search_similar(
            self,
            query_embedding: EmbeddingVector,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """
        Search for similar chunks by vector similarity, with optional metadata filters.
        """
        # Build ChromaDB-compatible where filter
        where = self._build_chroma_where_filter(filters, collection_id)

        # Extract similarity threshold if provided
        min_similarity = None
        if filters and "min_similarity" in filters:
            min_similarity = float(filters["min_similarity"])

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._search_similar_sync,
            query_embedding,
            where,
            top_k,
            min_similarity
        )

    def _search_similar_sync(
            self,
            query_embedding: EmbeddingVector,
            where: Optional[Dict[str, Any]],
            top_k: int,
            min_similarity: Optional[float] = None
    ) -> List[DocumentChunk]:
        """
        Synchronous helper for search_similar with cosine similarity.
        """
        query_vec = query_embedding.values

        # Query ChromaDB - increase top_k if we need to filter by similarity
        query_top_k = top_k * 2 if min_similarity else top_k

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=query_top_k,
            where=where,
            include=['metadatas', 'documents', 'distances']
        )

        ids_list = results.get("ids", [[]])
        metadatas_list = results.get("metadatas", [[]])
        documents_list = results.get("documents", [[]])
        distances_list = results.get("distances", [[]])

        chunks: List[DocumentChunk] = []

        for idx, chunk_id in enumerate(ids_list[0]):
            # For cosine distance, similarity = 1 - distance
            # Calculate similarity from distance (ChromaDB returns distances, not similarities)
            distance = distances_list[0][idx] if distances_list and distances_list[0] else 0
            similarity = 1 - distance  # Convert to similarity (1 = identical, -1 = opposite)

            # Apply similarity threshold filter
            if min_similarity and similarity < min_similarity:
                continue

            metadata = metadatas_list[0][idx] if metadatas_list and metadatas_list[0] else {}
            content = documents_list[0][idx] if documents_list and documents_list[0] else ""

            document_id = metadata.get("document_id")
            coll_id = metadata.get("collection_id")
            chunk_index = metadata.get("chunk_index", 0)
            chunk_type = metadata.get("chunk_type", "paragraph")
            parent_chunk_id = metadata.get("parent_chunk_id", None)

            # Convert chunk_index back to int if it was stored as string
            if isinstance(chunk_index, str) and chunk_index.isdigit():
                chunk_index = int(chunk_index)

            # Add similarity to metadata for frontend display
            metadata["similarity"] = similarity

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=coll_id,
                content=content,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                parent_chunk_id=parent_chunk_id,
                metadata=metadata,
                embedding_vector=None  # vector not returned by Chroma query
            )
            chunks.append(chunk)

            # Stop if we have enough results after filtering
            if len(chunks) >= top_k:
                break

        return chunks

    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        where = {
            "document_id": document_id,
        }
        results = self._collection.get(
            where=where,
            include=['metadatas', 'documents']
        )
        ids_list = results.get("ids", [])
        metadatas_list = results.get("metadatas", [])
        documents_list = results.get("documents", [])

        chunks: List[DocumentChunk] = []

        for idx, chunk_id in enumerate(ids_list):
            metadata = metadatas_list[idx] if metadatas_list and metadatas_list else {}
            content = documents_list[idx] if documents_list and documents_list else ""

            document_id = metadata.get("document_id")
            coll_id = metadata.get("collection_id")
            chunk_index = metadata.get("chunk_index", 0)
            chunk_type = metadata.get("chunk_type", "paragraph")
            parent_chunk_id = metadata.get("parent_chunk_id", None)

            # Convert chunk_index back to int if it was stored as string
            if isinstance(chunk_index, str) and chunk_index.isdigit():
                chunk_index = int(chunk_index)

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=coll_id,
                content=content,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                parent_chunk_id=parent_chunk_id,
                metadata=metadata,
                embedding_vector=None  # vector not returned by Chroma query
            )
            chunks.append(chunk)

        return chunks

    async def hybrid_search(
            self,
            query: SearchQuery,
            alpha: float = 0.5
    ) -> List[DocumentChunk]:
        """
        Hybrid search combining vector and keyword search.
        Currently not implemented; caller can fallback to vector-only search.
        """
        raise NotImplementedError("Hybrid search not implemented yet. Use pure vector search via search_similar.")

    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks with given document_id.
        """
        where = {"document_id": document_id}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._delete_sync, where)

    async def delete_by_collection_id(self, collection_id: str) -> bool:
        """
        Delete all chunks with given collection_id.
        """
        where = {"collection_id": collection_id}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._delete_sync, where)

    def _delete_sync(self, where: Dict[str, Any]) -> bool:
        """
        Synchronous helper for deletions.
        """
        try:
            self._collection.delete(where=where)
            return True
        except Exception:
            return False

    async def get_all_chunks_in_collection(self, collection_id, limit = None, include_embeddings = True) -> List[DocumentChunk]:
        where = {
            "collection_id": collection_id,
        }
        include = ['metadatas', 'documents']
        if include_embeddings:
            include.append('embeddings')
        
        # Execute the synchronous get operation in the thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_all_chunks_in_collection_sync, where, include, limit)
    
    def _get_all_chunks_in_collection_sync(self, where: Dict[str, str], include: List[str], limit: Optional[int]) -> List[DocumentChunk]:
        """
        Synchronous helper to retrieve all chunks in a collection.
        """
        results = self._collection.get(
            where=where,
            include=include,
            limit=limit,
        )
        
        ids_list = results.get("ids", [])
        metadatas_list = results.get("metadatas", [])
        documents_list = results.get("documents", [])
        embeddings_list = results.get("embeddings", [])

        chunks: List[DocumentChunk] = []

        # Iterate over results; ChromaDB returns a flat list for documents and metadata
        for idx, chunk_id in enumerate(ids_list):
            metadata = metadatas_list[idx] if metadatas_list and metadatas_list[idx] else {}
            content = documents_list[idx] if documents_list and documents_list[idx] else ""
            embeddings = embeddings_list[idx] if embeddings_list and embeddings_list[idx] else None

            document_id = metadata.get("document_id")
            coll_id = metadata.get("collection_id")
            chunk_index = metadata.get("chunk_index", 0)
            chunk_type = metadata.get("chunk_type", "paragraph")
            parent_chunk_id = metadata.get("parent_chunk_id", None)

            # Convert chunk_index back to int if it was stored as string
            if isinstance(chunk_index, str) and chunk_index.isdigit():
                chunk_index = int(chunk_index)

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=coll_id,
                content=content,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                parent_chunk_id=parent_chunk_id,
                metadata=metadata,
                embedding_vector=embeddings
            )
            chunks.append(chunk)

        return chunks

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a single document chunk by its unique ID.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_chunk_by_id_sync, chunk_id)

    def _get_chunk_by_id_sync(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Synchronous helper to retrieve a single chunk by its unique ID (ChromaDB internal ID).
        """
        # Efficiently query ChromaDB by its internal ID using the 'ids' parameter.
        results = self._collection.get(
            ids=[chunk_id],
            include=['metadatas', 'documents']
        )

        ids_list = results.get("ids", [])
        
        # If the query returns no IDs, the chunk was not found.
        if not ids_list:
            return None
        
        # Extract the single result (the lists are flat when using .get(ids=[...]))
        found_id = ids_list[0]
        # ChromaDB returns a flat list of results for `metadatas` and `documents` when ids are specified.
        metadata = results.get("metadatas", [None])[0] or {}
        content = results.get("documents", [None])[0] or ""

        # --- Rebuilding DocumentChunk object ---
        
        document_id = metadata.get("document_id")
        coll_id = metadata.get("collection_id")
        chunk_index = metadata.get("chunk_index", 0)
        chunk_type = metadata.get("chunk_type", "paragraph")
        parent_chunk_id = metadata.get("parent_chunk_id", None)
        
        # Convert chunk_index back to int if it was stored as string
        if isinstance(chunk_index, str) and chunk_index.isdigit():
            chunk_index = int(chunk_index)

        return DocumentChunk(
            id=found_id,
            document_id=document_id,
            collection_id=coll_id,
            content=content,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            parent_chunk_id=parent_chunk_id,
            metadata=metadata,
            embedding_vector=None
        )

