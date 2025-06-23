# app/adapters/search/bm25_adapter.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
import pickle
import os
import logging
from ...core.ports.bm25_service import BM25Service
from ...core.domain.entities.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Adapter(BM25Service):
    """BM25 implementation using rank-bm25 library"""

    def __init__(self, persist_directory: str, index_name: str = "documents_bm25"):
        self.persist_directory = persist_directory
        self.index_name = index_name

        # Create file paths using index_name for better organization
        self.index_path = os.path.join(persist_directory, f"{index_name}_index.pkl")
        self.chunks_path = os.path.join(persist_directory, f"{index_name}_chunks.pkl")
        self.metadata_path = os.path.join(persist_directory, f"{index_name}_metadata.pkl")

        self._executor = ThreadPoolExecutor(max_workers=2)

        # Initialize data structures
        self.bm25_index = None
        self.indexed_chunks = {}  # chunk_id -> DocumentChunk
        self.chunk_order = []  # Maintains order for BM25 index alignment

        # Load existing index if available
        self._load_index()

    def _load_index(self):
        """Load BM25 index from disk if exists"""
        try:
            if (os.path.exists(self.index_path) and
                    os.path.exists(self.chunks_path) and
                    os.path.exists(self.metadata_path)):
                with open(self.index_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)

                with open(self.chunks_path, 'rb') as f:
                    self.indexed_chunks = pickle.load(f)

                with open(self.metadata_path, 'rb') as f:
                    self.chunk_order = pickle.load(f)

                logger.info(f"Loaded BM25 index with {len(self.indexed_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}. Starting fresh.")
            self.bm25_index = None
            self.indexed_chunks = {}
            self.chunk_order = []

    def _save_index(self):
        """Save BM25 index to disk"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)

            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.indexed_chunks, f)

            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunk_order, f)

            logger.info(f"Saved BM25 index with {len(self.indexed_chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def _tokenize_content(self, content: str) -> List[str]:
        """Simple tokenization - can be enhanced with spaCy if needed"""
        # Basic preprocessing: lowercase, split on whitespace and common punctuation
        import re
        # Remove special characters but keep alphanumeric and spaces
        content = re.sub(r'[^\w\s]', ' ', content.lower())
        # Split and filter empty tokens
        tokens = [token for token in content.split() if token.strip()]
        return tokens

    async def index_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Index chunks for BM25 search"""
        if not chunks:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._index_chunks_sync, chunks)

    def _index_chunks_sync(self, chunks: List[DocumentChunk]) -> bool:
        """Synchronous BM25 indexing"""
        try:
            # Add new chunks to indexed_chunks
            for chunk in chunks:
                if chunk.id not in self.indexed_chunks:
                    self.indexed_chunks[chunk.id] = chunk
                    self.chunk_order.append(chunk.id)
                else:
                    # Update existing chunk
                    self.indexed_chunks[chunk.id] = chunk

            # Rebuild the entire BM25 index (BM25Okapi doesn't support incremental updates)
            self._rebuild_index()

            # Save to disk
            self._save_index()

            logger.info(f"Indexed {len(chunks)} chunks. Total chunks: {len(self.indexed_chunks)}")
            return True

        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            return False

    def _rebuild_index(self):
        """Rebuild the entire BM25 index from current chunks"""
        if not self.indexed_chunks:
            self.bm25_index = None
            return

        tokenized_docs = []

        # Process chunks in order
        for chunk_id in self.chunk_order:
            if chunk_id in self.indexed_chunks:
                chunk = self.indexed_chunks[chunk_id]
                # Use contextual content if available, otherwise use original content
                content = chunk.metadata.get('contextual_content', chunk.content)
                tokens = self._tokenize_content(content)
                tokenized_docs.append(tokens)

        # Create new BM25 index
        if tokenized_docs:
            self.bm25_index = BM25Okapi(tokenized_docs)
        else:
            self.bm25_index = None

    async def search(
            self,
            query_text: str,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """Search using BM25"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._search_sync,
            query_text,
            collection_id,
            top_k,
            filters
        )

    def _search_sync(
            self,
            query_text: str,
            collection_id: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """Synchronous BM25 search"""
        if not self.bm25_index or not self.indexed_chunks:
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize_content(query_text)
            if not query_tokens:
                return []

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top results with scores (get more for filtering)
            top_indices = scores.argsort()[-top_k * 3:][::-1]

            results = []

            for idx in top_indices:
                if idx >= len(self.chunk_order):
                    continue

                chunk_id = self.chunk_order[idx]
                if chunk_id not in self.indexed_chunks:
                    continue

                chunk = self.indexed_chunks[chunk_id]
                score = scores[idx]

                # Skip chunks with very low scores (BM25 can return 0 scores)
                if score <= 0:
                    continue

                # Apply collection filter
                if collection_id and chunk.collection_id != collection_id:
                    continue

                # Apply metadata filters
                if filters:
                    skip_chunk = False
                    for key, value in filters.items():
                        # Skip similarity-related filters (not applicable to BM25)
                        if key in ['min_similarity', 'max_similarity', 'similarity_threshold']:
                            continue
                        if key not in chunk.metadata or chunk.metadata[key] != value:
                            skip_chunk = True
                            break
                    if skip_chunk:
                        continue

                # Create chunk copy with BM25 score
                chunk_copy = DocumentChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    collection_id=chunk.collection_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    chunk_type=chunk.chunk_type,
                    parent_chunk_id=chunk.parent_chunk_id,
                    metadata={**chunk.metadata, 'bm25_score': float(score)},
                    embedding_vector=chunk.embedding_vector
                )

                results.append(chunk_copy)

                if len(results) >= top_k:
                    break

            logger.debug(f"BM25 search returned {len(results)} results for query: {query_text[:50]}...")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def remove_chunks(self, chunk_ids: List[str]) -> bool:
        """Remove chunks from BM25 index"""
        if not chunk_ids:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._remove_chunks_sync, chunk_ids)

    def _remove_chunks_sync(self, chunk_ids: List[str]) -> bool:
        """Synchronous chunk removal"""
        try:
            removed_count = 0

            # Remove chunks from data structures
            for chunk_id in chunk_ids:
                if chunk_id in self.indexed_chunks:
                    del self.indexed_chunks[chunk_id]
                    removed_count += 1

                # Remove from order list
                if chunk_id in self.chunk_order:
                    self.chunk_order.remove(chunk_id)

            if removed_count > 0:
                # Rebuild the BM25 index after removal
                self._rebuild_index()

                # Save updated index
                self._save_index()

                logger.info(f"Removed {removed_count} chunks from BM25 index")

            return True

        except Exception as e:
            logger.error(f"Failed to remove chunks from BM25 index: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        return {
            "total_chunks": len(self.indexed_chunks),
            "index_name": self.index_name,
            "persist_directory": self.persist_directory,
            "has_index": self.bm25_index is not None
        }

    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
