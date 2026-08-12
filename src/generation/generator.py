"""
Outfit Generation Module

Generate outfit recommendations from a user's wardrobe using
compatibility scoring and ranking algorithms.
"""

import numpy as np
from itertools import product
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import heapq

from ..compatibility.scorer import CompatibilityScorer, OutfitScore


@dataclass
class WardrobeItem:
    """A single item in the user's wardrobe."""
    item_id: str
    category: str
    category_id: int
    image_path: str
    visual_features: np.ndarray
    color_features: np.ndarray
    dominant_colors: List = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_features(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get feature tuple for scoring."""
        return (self.visual_features, self.color_features, self.category_id)


@dataclass
class GeneratedOutfit:
    """A generated outfit recommendation."""
    items: List[WardrobeItem]
    score: float
    pairwise_scores: Dict[Tuple[str, str], float]
    explanation: Optional[str] = None

    def get_item_ids(self) -> List[str]:
        return [item.item_id for item in self.items]

    def get_categories(self) -> List[str]:
        return [item.category for item in self.items]


class OutfitTemplate:
    """
    Defines a valid outfit structure.

    An outfit template specifies which categories must be present
    and optional categories that can be included.
    """

    def __init__(
        self,
        name: str,
        required_categories: List[str],
        optional_categories: Optional[List[str]] = None
    ):
        """
        Args:
            name: Template name (e.g., "casual", "formal")
            required_categories: Categories that must be present
            optional_categories: Categories that can optionally be added
        """
        self.name = name
        self.required_categories = required_categories
        self.optional_categories = optional_categories or []

    def is_valid(self, categories: List[str]) -> bool:
        """Check if a set of categories satisfies this template."""
        return all(cat in categories for cat in self.required_categories)


# Default outfit templates
DEFAULT_TEMPLATES = [
    OutfitTemplate(
        name="casual",
        required_categories=["tops", "bottoms"],
        optional_categories=["shoes", "accessories"]
    ),
    OutfitTemplate(
        name="full_casual",
        required_categories=["tops", "bottoms", "shoes"],
        optional_categories=["accessories"]
    ),
    OutfitTemplate(
        name="dress",
        required_categories=["dresses"],
        optional_categories=["shoes", "accessories"]
    ),
    OutfitTemplate(
        name="complete",
        required_categories=["tops", "bottoms", "shoes", "accessories"],
        optional_categories=[]
    ),
]


class Wardrobe:
    """
    Manages a collection of wardrobe items.

    Provides methods for organizing items by category and
    retrieving items for outfit generation.
    """

    CATEGORY_MAP = {
        "tops": 0,
        "bottoms": 1,
        "dresses": 2,
        "shoes": 3,
        "accessories": 4
    }

    def __init__(self):
        self.items: Dict[str, WardrobeItem] = {}
        self.items_by_category: Dict[str, List[str]] = {
            cat: [] for cat in self.CATEGORY_MAP.keys()
        }

    def add_item(self, item: WardrobeItem):
        """Add an item to the wardrobe."""
        self.items[item.item_id] = item
        if item.item_id not in self.items_by_category[item.category]:
            self.items_by_category[item.category].append(item.item_id)

    def remove_item(self, item_id: str):
        """Remove an item from the wardrobe."""
        if item_id in self.items:
            item = self.items[item_id]
            self.items_by_category[item.category].remove(item_id)
            del self.items[item_id]

    def get_item(self, item_id: str) -> Optional[WardrobeItem]:
        """Get an item by ID."""
        return self.items.get(item_id)

    def get_items_by_category(self, category: str) -> List[WardrobeItem]:
        """Get all items in a category."""
        return [
            self.items[item_id]
            for item_id in self.items_by_category.get(category, [])
        ]

    def get_all_items(self) -> List[WardrobeItem]:
        """Get all items in the wardrobe."""
        return list(self.items.values())

    def get_category_counts(self) -> Dict[str, int]:
        """Get count of items per category."""
        return {
            cat: len(items)
            for cat, items in self.items_by_category.items()
        }

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.items


class OutfitGenerator:
    """
    Generate outfit recommendations from a wardrobe.

    Uses compatibility scoring to find the best combinations
    of items that work well together.
    """

    def __init__(
        self,
        scorer: CompatibilityScorer,
        templates: Optional[List[OutfitTemplate]] = None,
        min_compatibility: float = 0.5,
        diversity_weight: float = 0.2
    ):
        """
        Initialize the outfit generator.

        Args:
            scorer: CompatibilityScorer for evaluating outfits
            templates: Outfit templates defining valid structures
            min_compatibility: Minimum score threshold for recommendations
            diversity_weight: Weight for diversity in ranking
        """
        self.scorer = scorer
        self.templates = templates or DEFAULT_TEMPLATES
        self.min_compatibility = min_compatibility
        self.diversity_weight = diversity_weight

    def _enumerate_combinations(
        self,
        wardrobe: Wardrobe,
        template: OutfitTemplate
    ) -> Iterator[List[WardrobeItem]]:
        """
        Enumerate all valid outfit combinations for a template.

        Args:
            wardrobe: User's wardrobe
            template: Outfit template to use

        Yields:
            Lists of WardrobeItem combinations
        """
        # Get items for each required category
        category_items = []
        for cat in template.required_categories:
            items = wardrobe.get_items_by_category(cat)
            if not items:
                return  # Can't satisfy this template
            category_items.append(items)

        # Generate all combinations
        for combo in product(*category_items):
            yield list(combo)

    def _score_outfit(
        self,
        items: List[WardrobeItem]
    ) -> OutfitScore:
        """Score an outfit combination."""
        features = [item.get_features() for item in items]
        item_ids = [item.item_id for item in items]
        return self.scorer.score_outfit(features, item_ids)

    def generate_outfits(
        self,
        wardrobe: Wardrobe,
        template: Optional[OutfitTemplate] = None,
        anchor_item: Optional[WardrobeItem] = None,
        max_outfits: int = 10
    ) -> List[GeneratedOutfit]:
        """
        Generate outfit recommendations.

        Args:
            wardrobe: User's wardrobe
            template: Specific template to use (default: tries all)
            anchor_item: Item that must be included in outfits
            max_outfits: Maximum number of outfits to return

        Returns:
            List of GeneratedOutfit sorted by score
        """
        all_outfits = []

        # Determine which templates to use
        templates_to_try = [template] if template else self.templates

        for tmpl in templates_to_try:
            # Check if we can satisfy this template
            can_satisfy = all(
                len(wardrobe.get_items_by_category(cat)) > 0
                for cat in tmpl.required_categories
            )
            if not can_satisfy:
                continue

            # Generate combinations
            for items in self._enumerate_combinations(wardrobe, tmpl):
                # If anchor item is specified, check it's included
                if anchor_item:
                    if anchor_item not in items:
                        # Try to include anchor if its category is in template
                        if anchor_item.category in tmpl.required_categories:
                            # Replace the item of same category
                            items = [
                                anchor_item if item.category == anchor_item.category
                                else item
                                for item in items
                            ]
                        else:
                            continue

                # Score the outfit
                score = self._score_outfit(items)

                if score.overall_score >= self.min_compatibility:
                    outfit = GeneratedOutfit(
                        items=items,
                        score=score.overall_score,
                        pairwise_scores=score.pairwise_scores
                    )
                    all_outfits.append(outfit)

        # Sort by score and apply diversity
        all_outfits = self._rank_with_diversity(all_outfits, max_outfits)

        return all_outfits[:max_outfits]

    def _rank_with_diversity(
        self,
        outfits: List[GeneratedOutfit],
        top_k: int
    ) -> List[GeneratedOutfit]:
        """
        Rank outfits considering both score and diversity.

        Uses a greedy algorithm to select diverse outfits.
        """
        if len(outfits) <= top_k:
            return sorted(outfits, key=lambda o: o.score, reverse=True)

        # Sort by score first
        outfits = sorted(outfits, key=lambda o: o.score, reverse=True)

        selected = [outfits[0]]
        remaining = outfits[1:]

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -float('inf')

            for i, outfit in enumerate(remaining):
                # Compute diversity score (how different from selected)
                diversity = self._compute_diversity(outfit, selected)

                # Combined score
                combined = (
                    (1 - self.diversity_weight) * outfit.score +
                    self.diversity_weight * diversity
                )

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _compute_diversity(
        self,
        outfit: GeneratedOutfit,
        selected: List[GeneratedOutfit]
    ) -> float:
        """Compute how different an outfit is from selected ones."""
        if not selected:
            return 1.0

        outfit_items = set(outfit.get_item_ids())
        diversities = []

        for sel_outfit in selected:
            sel_items = set(sel_outfit.get_item_ids())
            # Jaccard distance
            intersection = len(outfit_items & sel_items)
            union = len(outfit_items | sel_items)
            diversity = 1 - (intersection / union) if union > 0 else 1.0
            diversities.append(diversity)

        return min(diversities)  # Minimum diversity from any selected

    def generate_with_beam_search(
        self,
        wardrobe: Wardrobe,
        template: OutfitTemplate,
        beam_width: int = 10,
        max_outfits: int = 10
    ) -> List[GeneratedOutfit]:
        """
        Generate outfits using beam search for large wardrobes.

        More efficient than exhaustive enumeration when wardrobe is large.

        Args:
            wardrobe: User's wardrobe
            template: Outfit template
            beam_width: Number of candidates to keep at each step
            max_outfits: Maximum outfits to return

        Returns:
            List of GeneratedOutfit
        """
        if not template.required_categories:
            return []

        # Start with first category
        first_cat = template.required_categories[0]
        beam = [
            ([item], 1.0)  # (partial_outfit, score)
            for item in wardrobe.get_items_by_category(first_cat)
        ]

        # Extend beam for each subsequent category
        for cat in template.required_categories[1:]:
            candidates = wardrobe.get_items_by_category(cat)
            if not candidates:
                return []

            new_beam = []
            for partial_outfit, current_score in beam:
                for candidate in candidates:
                    # Score candidate with existing items
                    extended = partial_outfit + [candidate]
                    total_score = 0
                    n_pairs = 0

                    for existing in partial_outfit:
                        pair_score = self.scorer.score_pair(
                            existing.get_features(),
                            candidate.get_features()
                        )
                        total_score += pair_score.score
                        n_pairs += 1

                    # Running average score
                    new_score = (
                        current_score * (n_pairs - 1) + (total_score / n_pairs)
                    ) / n_pairs if n_pairs > 0 else current_score

                    new_beam.append((extended, new_score))

            # Keep top beam_width
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

        # Convert to GeneratedOutfit
        outfits = []
        for items, score in beam:
            outfit_score = self._score_outfit(items)
            if outfit_score.overall_score >= self.min_compatibility:
                outfits.append(GeneratedOutfit(
                    items=items,
                    score=outfit_score.overall_score,
                    pairwise_scores=outfit_score.pairwise_scores
                ))

        return sorted(outfits, key=lambda o: o.score, reverse=True)[:max_outfits]

    def suggest_additions(
        self,
        current_items: List[WardrobeItem],
        wardrobe: Wardrobe,
        category: str,
        top_k: int = 5
    ) -> List[Tuple[WardrobeItem, float]]:
        """
        Suggest items to add to a partial outfit.

        Args:
            current_items: Items already selected
            wardrobe: User's wardrobe
            category: Category to suggest from
            top_k: Number of suggestions

        Returns:
            List of (item, score) tuples
        """
        candidates = wardrobe.get_items_by_category(category)
        if not candidates:
            return []

        # Score each candidate against current items
        scored = []
        for candidate in candidates:
            if candidate in current_items:
                continue

            total_score = 0
            for current in current_items:
                pair_score = self.scorer.score_pair(
                    current.get_features(),
                    candidate.get_features()
                )
                total_score += pair_score.score

            avg_score = total_score / len(current_items) if current_items else 0
            scored.append((candidate, avg_score))

        # Sort and return top k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
