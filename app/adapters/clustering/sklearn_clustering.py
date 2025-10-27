from sklearn.cluster import KMeans
import numpy as np
from typing import List

from app.core.ports.clustering_service import ClusteringService
from app.core.domain.entities.document_chunk import DocumentChunk


class SklearnClusteringAdapter(ClusteringService):
    
    async def get_diverse_representatives(
        self,
        chunks_with_embeddings: List[DocumentChunk],
        k: int
    ) -> List[DocumentChunk]:
        
        if not chunks_with_embeddings:
            return []

        if len(chunks_with_embeddings) <= k:
            return chunks_with_embeddings

        embeddings = []
        valid_chunks = []
        for chunk in chunks_with_embeddings:
            if chunk.embedding_vector:
                embeddings.append(chunk.embedding_vector)
                valid_chunks.append(chunk)

        if not embeddings:
            import random
            return random.sample(chunks_with_embeddings, min(k, len(chunks_with_embeddings)))

        n_clusters = min(k, len(embeddings))
        embeddings_array = np.array(embeddings)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

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
