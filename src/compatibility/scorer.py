"""
Compatibility Scorer Module

Score compatibility between garment pairs and full outfits
using the trained Siamese network.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .model import SiameseCompatibilityNet, get_device


@dataclass
class CompatibilityScore:
    """Compatibility score with breakdown."""
    score: float
    confidence: float
    details: Optional[Dict] = None


@dataclass
class OutfitScore:
    """Full outfit compatibility score."""
    overall_score: float
    pairwise_scores: Dict[Tuple[str, str], float]
    weakest_pair: Optional[Tuple[str, str]] = None
    strongest_pair: Optional[Tuple[str, str]] = None


class CompatibilityScorer:
    """
    Score outfit compatibility using a trained Siamese network.

    Provides pairwise and full outfit scoring with explanations.
    """

    def __init__(
        self,
        model: SiameseCompatibilityNet,
        classifier=None,
        color_extractor=None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the scorer.

        Args:
            model: Trained SiameseCompatibilityNet
            classifier: GarmentClassifier for extracting visual features
            color_extractor: ColorExtractor for extracting color features
            device: Device to run inference on
        """
        self.model = model
        self.classifier = classifier
        self.color_extractor = color_extractor
        self.device = device or get_device()

        self.model.to(self.device)
        self.model.eval()

        # Cache for embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def _extract_features(
        self,
        image_path: Union[str, Path],
        category_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from an image.

        Returns:
            Tuple of (visual_features, color_features, type_tensor)
        """
        # Visual features
        vis_feat = self.classifier.get_features(image_path)
        vis_tensor = torch.tensor(vis_feat, dtype=torch.float32).unsqueeze(0)

        # Color features
        col_feat = self.color_extractor.get_color_vector(image_path)
        col_tensor = torch.tensor(col_feat, dtype=torch.float32).unsqueeze(0)

        # Type
        type_tensor = torch.tensor([category_id], dtype=torch.long)

        return vis_tensor, col_tensor, type_tensor

    def _get_embedding(
        self,
        visual_features: torch.Tensor,
        color_features: torch.Tensor,
        type_idx: torch.Tensor,
        cache_key: Optional[str] = None
    ) -> torch.Tensor:
        """Get embedding, using cache if available."""
        if cache_key and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        with torch.no_grad():
            embedding = self.model.encode(
                visual_features.to(self.device),
                color_features.to(self.device),
                type_idx.to(self.device)
            )

        if cache_key:
            self._embedding_cache[cache_key] = embedding

        return embedding

    @torch.no_grad()
    def score_pair(
        self,
        item1_features: Tuple[np.ndarray, np.ndarray, int],
        item2_features: Tuple[np.ndarray, np.ndarray, int],
        item1_id: Optional[str] = None,
        item2_id: Optional[str] = None
    ) -> CompatibilityScore:
        """
        Score compatibility between two items.

        Args:
            item1_features: (visual_features, color_features, category_id) for item 1
            item2_features: (visual_features, color_features, category_id) for item 2
            item1_id: Optional ID for caching
            item2_id: Optional ID for caching

        Returns:
            CompatibilityScore with score in [0, 1]
        """
        vis1, col1, type1 = item1_features
        vis2, col2, type2 = item2_features

        # Convert to tensors
        vis1_t = torch.tensor(vis1, dtype=torch.float32).unsqueeze(0)
        col1_t = torch.tensor(col1, dtype=torch.float32).unsqueeze(0)
        type1_t = torch.tensor([type1], dtype=torch.long)

        vis2_t = torch.tensor(vis2, dtype=torch.float32).unsqueeze(0)
        col2_t = torch.tensor(col2, dtype=torch.float32).unsqueeze(0)
        type2_t = torch.tensor([type2], dtype=torch.long)

        # Get embeddings
        emb1 = self._get_embedding(vis1_t, col1_t, type1_t, item1_id)
        emb2 = self._get_embedding(vis2_t, col2_t, type2_t, item2_id)

        # Compute similarity (cosine similarity for L2-normalized vectors)
        similarity = self.model.compute_similarity(emb1, emb2).item()

        # Convert from [-1, 1] to [0, 1]
        score = (similarity + 1) / 2

        # Estimate confidence based on distance from decision boundary
        confidence = min(1.0, abs(similarity) * 2)

        return CompatibilityScore(
            score=score,
            confidence=confidence
        )

    @torch.no_grad()
    def score_pair_from_images(
        self,
        image1_path: Union[str, Path],
        category1_id: int,
        image2_path: Union[str, Path],
        category2_id: int
    ) -> CompatibilityScore:
        """
        Score compatibility between two items from their images.

        Args:
            image1_path: Path to first item's image
            category1_id: Category ID of first item
            image2_path: Path to second item's image
            category2_id: Category ID of second item

        Returns:
            CompatibilityScore
        """
        if self.classifier is None or self.color_extractor is None:
            raise ValueError("Classifier and color_extractor required for image scoring")

        # Extract features
        vis1, col1, type1 = self._extract_features(image1_path, category1_id)
        vis2, col2, type2 = self._extract_features(image2_path, category2_id)

        # Get embeddings
        emb1 = self._get_embedding(vis1, col1, type1, str(image1_path))
        emb2 = self._get_embedding(vis2, col2, type2, str(image2_path))

        # Compute similarity
        similarity = self.model.compute_similarity(emb1, emb2).item()
        score = (similarity + 1) / 2
        confidence = min(1.0, abs(similarity) * 2)

        return CompatibilityScore(score=score, confidence=confidence)

    def score_outfit(
        self,
        items: List[Tuple[np.ndarray, np.ndarray, int]],
        item_ids: Optional[List[str]] = None
    ) -> OutfitScore:
        """
        Score a complete outfit.

        Computes pairwise compatibility scores and aggregates them.

        Args:
            items: List of (visual_features, color_features, category_id) tuples
            item_ids: Optional list of item IDs for caching

        Returns:
            OutfitScore with overall and pairwise scores
        """
        if len(items) < 2:
            return OutfitScore(
                overall_score=1.0,
                pairwise_scores={}
            )

        if item_ids is None:
            item_ids = [str(i) for i in range(len(items))]

        # Compute all pairwise scores
        pairwise_scores = {}
        all_scores = []

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score = self.score_pair(
                    items[i], items[j],
                    item_ids[i], item_ids[j]
                )
                pair_key = (item_ids[i], item_ids[j])
                pairwise_scores[pair_key] = score.score
                all_scores.append((pair_key, score.score))

        # Aggregate scores
        scores_array = np.array([s for _, s in all_scores])
        overall_score = np.mean(scores_array)

        # Find weakest and strongest pairs
        all_scores.sort(key=lambda x: x[1])
        weakest_pair = all_scores[0][0] if all_scores else None
        strongest_pair = all_scores[-1][0] if all_scores else None

        return OutfitScore(
            overall_score=float(overall_score),
            pairwise_scores=pairwise_scores,
            weakest_pair=weakest_pair,
            strongest_pair=strongest_pair
        )

    def rank_candidates(
        self,
        anchor_features: Tuple[np.ndarray, np.ndarray, int],
        candidates: List[Tuple[np.ndarray, np.ndarray, int]],
        anchor_id: Optional[str] = None,
        candidate_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate items by compatibility with an anchor item.

        Args:
            anchor_features: Features of the anchor item
            candidates: List of candidate item features
            anchor_id: Optional ID for caching
            candidate_ids: Optional list of candidate IDs
            top_k: Return only top k results

        Returns:
            List of (candidate_index, score) sorted by score descending
        """
        if candidate_ids is None:
            candidate_ids = [str(i) for i in range(len(candidates))]

        scores = []
        for i, (cand_features, cand_id) in enumerate(zip(candidates, candidate_ids)):
            score = self.score_pair(
                anchor_features,
                cand_features,
                anchor_id,
                cand_id
            )
            scores.append((i, score.score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scores = scores[:top_k]

        return scores


class EnsembleScorer:
    """
    Ensemble compatibility scorer combining multiple scoring methods.

    Combines:
    1. Learned compatibility (Siamese network)
    2. Color harmony (rule-based)
    3. Style consistency (optional)
    """

    def __init__(
        self,
        compatibility_scorer: CompatibilityScorer,
        color_harmony_analyzer=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ensemble scorer.

        Args:
            compatibility_scorer: Trained CompatibilityScorer
            color_harmony_analyzer: ColorHarmonyAnalyzer instance
            weights: Weights for each scoring component
        """
        self.compatibility_scorer = compatibility_scorer
        self.color_harmony_analyzer = color_harmony_analyzer

        self.weights = weights or {
            "compatibility": 0.7,
            "color_harmony": 0.3
        }

    def score_pair(
        self,
        item1_features: Tuple[np.ndarray, np.ndarray, int],
        item2_features: Tuple[np.ndarray, np.ndarray, int],
        item1_colors=None,
        item2_colors=None
    ) -> CompatibilityScore:
        """
        Score a pair using ensemble of methods.

        Args:
            item1_features, item2_features: Feature tuples
            item1_colors, item2_colors: DominantColor lists for color harmony

        Returns:
            Combined CompatibilityScore
        """
        # Learned compatibility score
        compat_score = self.compatibility_scorer.score_pair(
            item1_features, item2_features
        )

        scores = {"compatibility": compat_score.score}

        # Color harmony score
        if self.color_harmony_analyzer and item1_colors and item2_colors:
            harmony = self.color_harmony_analyzer.analyze_harmony(
                item1_colors, item2_colors
            )
            scores["color_harmony"] = harmony["score"]

        # Weighted average
        total_weight = sum(
            self.weights.get(k, 0) for k in scores.keys()
        )
        weighted_sum = sum(
            scores[k] * self.weights.get(k, 0) for k in scores.keys()
        )
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        return CompatibilityScore(
            score=final_score,
            confidence=compat_score.confidence,
            details=scores
        )
