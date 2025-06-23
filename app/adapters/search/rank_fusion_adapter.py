import math
from typing import List, Dict
from ...core.ports.rank_fusion_service import RankFusionService, FusionMethod
from ...core.domain.entities.document_chunk import DocumentChunk


class HybridRankFusionAdapter(RankFusionService):
    """
    Implements various rank fusion techniques for combining
    BM25 (keyword) and vector (semantic) search results
    """
    def fuse_results(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            method: FusionMethod = FusionMethod.RECIPROCAL_RANK_FUSION,
            alpha: float = 0.5,
            top_k: int = 10,
            rrf_k: int = 60,
            **kwargs
    ) -> List[DocumentChunk]:
        """
        Main fusion method that delegates to specific fusion strategies

        Args:
            vector_results: Results from semantic/vector search
            bm25_results: Results from BM25/keyword search
            method: Fusion method to use
            alpha: Weight parameter (0.0 = only BM25, 1.0 = only vector)
            top_k: Number of final results to return
            rrf_k: RRF constant (higher values = less emphasis on rank differences)

        Returns:
            Fused and ranked list of document chunks
        """
        if method == FusionMethod.RECIPROCAL_RANK_FUSION:
            return self._reciprocal_rank_fusion(vector_results, bm25_results, top_k, rrf_k)
        elif method == FusionMethod.WEIGHTED_SCORE:
            return self._weighted_score_fusion(vector_results, bm25_results, alpha, top_k)
        elif method == FusionMethod.CONVEX_COMBINATION:
            return self._convex_combination_fusion(vector_results, bm25_results, alpha, top_k)
        elif method == FusionMethod.DISTRIBUTION_BASED:
            return self._distribution_based_fusion(vector_results, bm25_results, alpha, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _reciprocal_rank_fusion(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            top_k: int,
            rrf_k: int = 60
    ) -> List[DocumentChunk]:
        """
        Reciprocal Rank Fusion (RRF) - Most robust and commonly used method

        RRF Score = Σ(1 / (k + rank_i))
        where rank_i is the rank in each result list, k is a constant (typically 60)

        This method is rank-based and doesn't depend on the actual scores,
        making it more robust to score distribution differences.
        """
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Process vector search results
        for rank, chunk in enumerate(vector_results, 1):
            chunk_id = chunk.id
            rrf_score = 1.0 / (rrf_k + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_map[chunk_id] = chunk

        # Process BM25 search results
        for rank, chunk in enumerate(bm25_results, 1):
            chunk_id = chunk.id
            rrf_score = 1.0 / (rrf_k + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_map[chunk_id] = chunk

        # Sort by RRF score and return top-k
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Create result chunks with fusion metadata
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            chunk = chunk_map[chunk_id]

            # Add fusion metadata
            chunk.metadata = {
                **chunk.metadata,
                'rrf_score': rrf_score,
                'fusion_method': 'reciprocal_rank_fusion',
                'found_in_vector': chunk_id in [c.id for c in vector_results],
                'found_in_bm25': chunk_id in [c.id for c in bm25_results]
            }
            results.append(chunk)

        return results

    def _weighted_score_fusion(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            alpha: float,
            top_k: int
    ) -> List[DocumentChunk]:
        """
        Weighted Score Fusion - Combines normalized scores

        Final Score = α * normalized_vector_score + (1-α) * normalized_bm25_score
        """
        # Extract and normalize scores
        vector_scores = self._extract_and_normalize_scores(vector_results, 'similarity')
        bm25_scores = self._extract_and_normalize_scores(bm25_results, 'bm25_score')

        # Combine all chunks
        all_chunks = self._get_all_unique_chunks(vector_results, bm25_results)

        chunk_scores: Dict[str, float] = {}

        for chunk in all_chunks.values():
            chunk_id = chunk.id

            # Get normalized scores (0 if not found in that search)
            vec_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)

            # Weighted combination
            final_score = alpha * vec_score + (1 - alpha) * bm25_score
            chunk_scores[chunk_id] = final_score

        # Sort and return top-k
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for chunk_id, final_score in sorted_chunks:
            chunk = all_chunks[chunk_id]
            chunk.metadata = {
                **chunk.metadata,
                'weighted_score': final_score,
                'fusion_method': 'weighted_score',
                'alpha': alpha,
                'vector_score': vector_scores.get(chunk_id, 0.0),
                'bm25_score': bm25_scores.get(chunk_id, 0.0)
            }
            results.append(chunk)

        return results

    def _convex_combination_fusion(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            alpha: float,
            top_k: int
    ) -> List[DocumentChunk]:
        """
        Convex Combination with Rank-Score Hybrid

        Combines both rank information and score information
        """
        chunk_scores: Dict[str, float] = {}
        all_chunks = self._get_all_unique_chunks(vector_results, bm25_results)

        # Get normalized scores
        vector_scores = self._extract_and_normalize_scores(vector_results, 'similarity')
        bm25_scores = self._extract_and_normalize_scores(bm25_results, 'bm25_score')

        # Create rank maps
        vector_ranks = {chunk.id: rank for rank, chunk in enumerate(vector_results, 1)}
        bm25_ranks = {chunk.id: rank for rank, chunk in enumerate(bm25_results, 1)}

        max_rank = max(len(vector_results), len(bm25_results))

        for chunk in all_chunks.values():
            chunk_id = chunk.id

            # Score component
            vec_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            score_component = alpha * vec_score + (1 - alpha) * bm25_score

            # Rank component (normalized inverse rank)
            vec_rank = vector_ranks.get(chunk_id, max_rank + 1)
            bm25_rank = bm25_ranks.get(chunk_id, max_rank + 1)

            vec_rank_score = 1.0 - (vec_rank - 1) / max_rank
            bm25_rank_score = 1.0 - (bm25_rank - 1) / max_rank
            rank_component = alpha * vec_rank_score + (1 - alpha) * bm25_rank_score

            # Combine score and rank components (equal weight)
            final_score = 0.5 * score_component + 0.5 * rank_component
            chunk_scores[chunk_id] = final_score

        # Sort and return top-k
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for chunk_id, final_score in sorted_chunks:
            chunk = all_chunks[chunk_id]
            chunk.metadata = {
                **chunk.metadata,
                'convex_score': final_score,
                'fusion_method': 'convex_combination'
            }
            results.append(chunk)

        return results

    def _distribution_based_fusion(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk],
            alpha: float,
            top_k: int
    ) -> List[DocumentChunk]:
        """
        Distribution-based fusion using z-score normalization

        Normalizes scores using mean and standard deviation to handle
        different score distributions between vector and BM25 searches
        """
        # Extract raw scores
        vector_raw_scores = [
            chunk.metadata.get('similarity', 0.0) for chunk in vector_results
        ]
        bm25_raw_scores = [
            chunk.metadata.get('bm25_score', 0.0) for chunk in bm25_results
        ]

        # Calculate z-scores (standardization)
        vector_zscores = self._calculate_z_scores(vector_raw_scores)
        bm25_zscores = self._calculate_z_scores(bm25_raw_scores)

        # Create score maps
        vector_scores = {
            chunk.id: zscore for chunk, zscore in zip(vector_results, vector_zscores)
        }
        bm25_scores = {
            chunk.id: zscore for chunk, zscore in zip(bm25_results, bm25_zscores)
        }

        # Combine all chunks
        all_chunks = self._get_all_unique_chunks(vector_results, bm25_results)

        chunk_scores: Dict[str, float] = {}

        for chunk in all_chunks.values():
            chunk_id = chunk.id

            # Get z-scores (0 if not found)
            vec_zscore = vector_scores.get(chunk_id, 0.0)
            bm25_zscore = bm25_scores.get(chunk_id, 0.0)

            # Weighted combination of z-scores
            final_score = alpha * vec_zscore + (1 - alpha) * bm25_zscore
            chunk_scores[chunk_id] = final_score

        # Sort and return top-k
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for chunk_id, final_score in sorted_chunks:
            chunk = all_chunks[chunk_id]
            chunk.metadata = {
                **chunk.metadata,
                'distribution_score': final_score,
                'fusion_method': 'distribution_based'
            }
            results.append(chunk)

        return results

    # Helper methods

    def _extract_and_normalize_scores(
            self,
            chunks: List[DocumentChunk],
            score_key: str
    ) -> Dict[str, float]:
        """Extract and min-max normalize scores"""
        if not chunks:
            return {}

        scores = [chunk.metadata.get(score_key, 0.0) for chunk in chunks]

        if not scores or all(s == 0 for s in scores):
            return {chunk.id: 0.0 for chunk in chunks}

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return {chunk.id: 1.0 for chunk in chunks}

        normalized_scores = {}
        for chunk, score in zip(chunks, scores):
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores[chunk.id] = normalized_score

        return normalized_scores

    def _calculate_z_scores(self, scores: List[float]) -> List[float]:
        """Calculate z-scores (standardization)"""
        if not scores or len(scores) < 2:
            return [0.0] * len(scores)

        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / (len(scores) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        return [(score - mean_score) / std_dev for score in scores]

    def _get_all_unique_chunks(
            self,
            vector_results: List[DocumentChunk],
            bm25_results: List[DocumentChunk]
    ) -> Dict[str, DocumentChunk]:
        """Get all unique chunks from both result sets"""
        all_chunks = {}

        # Add vector results
        for chunk in vector_results:
            all_chunks[chunk.id] = chunk

        # Add BM25 results (will overwrite if same ID, but that's fine)
        for chunk in bm25_results:
            all_chunks[chunk.id] = chunk

        return all_chunks
