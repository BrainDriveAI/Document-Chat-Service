from sklearn.cluster import KMeans
import numpy as np
import random
import logging
from typing import List
from app.core.ports.clustering_service import ClusteringService
from app.core.domain.entities.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


class SklearnClusteringAdapter(ClusteringService):
    """
    Sklearn-based clustering service for getting diverse chunk representatives.
    Handles fallback strategies when embeddings are not available.
    """
    
    async def get_diverse_representatives(
        self,
        chunks_with_embeddings: List[DocumentChunk],
        k: int
    ) -> List[DocumentChunk]:
        """
        Get k diverse representative chunks using K-means clustering.
        Falls back to random sampling if embeddings are not available.
        
        Args:
            chunks_with_embeddings: List of document chunks (may or may not have embeddings)
            k: Number of representatives to return
            
        Returns:
            List of representative chunks
        """
        try:
            # Empty check
            if not chunks_with_embeddings or len(chunks_with_embeddings) == 0:
                return []
            
            # If we have fewer chunks than k, return all
            if len(chunks_with_embeddings) <= k:
                return chunks_with_embeddings
            
            # Separate chunks with valid embeddings
            embeddings = []
            valid_chunks = []
            
            for chunk in chunks_with_embeddings:
                # Properly check if embedding exists and is not None/empty
                if chunk.embedding_vector is not None and len(chunk.embedding_vector) > 0:
                    embeddings.append(chunk.embedding_vector)
                    valid_chunks.append(chunk)
            
            # Strategy 1: Use clustering if we have embeddings
            if embeddings and len(embeddings) > 0:
                return self._cluster_and_sample(embeddings, valid_chunks, k)
            
            # Strategy 2: Fallback to random sampling if no embeddings
            return self._random_sample(chunks_with_embeddings, k)
            
        except Exception as e:
            # Ultimate fallback: random sample
            logger.warning(f"Clustering failed, using random sampling: {e}")
            return self._random_sample(chunks_with_embeddings, k)
    
    def _cluster_and_sample(
        self,
        embeddings: List,
        valid_chunks: List[DocumentChunk],
        k: int
    ) -> List[DocumentChunk]:
        """
        Cluster embeddings and sample one chunk from each cluster.
        """
        try:
            n_clusters = min(k, len(embeddings))
            embeddings_array = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Sample one chunk from each cluster (closest to cluster center)
            sampled_chunks = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = np.linalg.norm(
                        embeddings_array[cluster_indices] - cluster_center,
                        axis=1
                    )
                    closest_idx = cluster_indices[np.argmin(distances)]
                    sampled_chunks.append(valid_chunks[closest_idx])
            
            return sampled_chunks
            
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            # Fallback to random from valid chunks
            return self._random_sample(valid_chunks, k)
    
    def _random_sample(
        self,
        chunks: List[DocumentChunk],
        k: int
    ) -> List[DocumentChunk]:
        """
        Randomly sample k chunks from the list.
        This is the ultimate fallback strategy.
        """
        if not chunks or len(chunks) == 0:
            return []
        
        sample_size = min(k, len(chunks))
        return random.sample(chunks, sample_size)
