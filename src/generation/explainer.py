"""
Outfit Explainer Module

Generate human-readable explanations for outfit recommendations
using color theory, compatibility scores, and style analysis.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .generator import GeneratedOutfit, WardrobeItem
from ..attributes.color_extractor import DominantColor, ColorHarmonyAnalyzer


@dataclass
class OutfitExplanation:
    """Complete explanation for an outfit recommendation."""
    summary: str
    color_analysis: str
    compatibility_analysis: str
    item_contributions: Dict[str, str]
    strengths: List[str]
    suggestions: List[str]
    score_breakdown: Dict[str, float]


class OutfitExplainer:
    """
    Generate natural language explanations for outfit recommendations.

    Combines multiple analysis methods:
    1. Color harmony analysis using color theory
    2. Compatibility score breakdown
    3. Style consistency analysis
    """

    def __init__(
        self,
        color_analyzer: Optional[ColorHarmonyAnalyzer] = None
    ):
        """
        Initialize the explainer.

        Args:
            color_analyzer: ColorHarmonyAnalyzer for color theory analysis
        """
        self.color_analyzer = color_analyzer or ColorHarmonyAnalyzer()

    def explain(self, outfit: GeneratedOutfit) -> OutfitExplanation:
        """
        Generate a complete explanation for an outfit.

        Args:
            outfit: GeneratedOutfit to explain

        Returns:
            OutfitExplanation with detailed analysis
        """
        # Analyze colors
        color_analysis, color_strengths = self._analyze_colors(outfit)

        # Analyze compatibility
        compat_analysis, compat_strengths = self._analyze_compatibility(outfit)

        # Analyze item contributions
        item_contributions = self._analyze_item_contributions(outfit)

        # Generate suggestions
        suggestions = self._generate_suggestions(outfit)

        # Generate summary
        summary = self._generate_summary(
            outfit, color_strengths, compat_strengths
        )

        # Score breakdown
        score_breakdown = {
            "overall": outfit.score,
            "min_pairwise": min(outfit.pairwise_scores.values()) if outfit.pairwise_scores else 0,
            "max_pairwise": max(outfit.pairwise_scores.values()) if outfit.pairwise_scores else 0,
        }

        return OutfitExplanation(
            summary=summary,
            color_analysis=color_analysis,
            compatibility_analysis=compat_analysis,
            item_contributions=item_contributions,
            strengths=color_strengths + compat_strengths,
            suggestions=suggestions,
            score_breakdown=score_breakdown
        )

    def _analyze_colors(
        self,
        outfit: GeneratedOutfit
    ) -> Tuple[str, List[str]]:
        """Analyze color harmony in the outfit."""
        strengths = []

        # Collect all dominant colors
        all_colors: List[DominantColor] = []
        color_names = []

        for item in outfit.items:
            if hasattr(item, 'dominant_colors') and item.dominant_colors:
                all_colors.extend(item.dominant_colors)
                color_names.append(item.dominant_colors[0].name if item.dominant_colors else "unknown")

        if not all_colors:
            return "Color information not available for this outfit.", []

        # Analyze pairwise color harmony
        harmony_types = []
        for i, item1 in enumerate(outfit.items):
            for item2 in outfit.items[i+1:]:
                colors1 = getattr(item1, 'dominant_colors', [])
                colors2 = getattr(item2, 'dominant_colors', [])
                if colors1 and colors2:
                    harmony = self.color_analyzer.analyze_harmony(colors1, colors2)
                    harmony_types.append(harmony["harmony_type"])

        # Determine overall color scheme
        unique_harmony = set(harmony_types)

        if "neutral-pairing" in unique_harmony:
            strengths.append("Neutral colors provide versatile foundation")

        if "complementary" in unique_harmony:
            strengths.append("Complementary colors create visual interest")

        if "analogous" in unique_harmony:
            strengths.append("Analogous colors create a cohesive look")

        # Generate color analysis text
        primary_colors = list(set(color_names))[:3]
        color_list = ", ".join(primary_colors)

        analysis = f"This outfit features {color_list}. "

        if len(unique_harmony) == 1:
            analysis += f"The colors follow a {list(unique_harmony)[0]} harmony."
        elif unique_harmony:
            harmony_list = ", ".join(unique_harmony)
            analysis += f"The color relationships include {harmony_list} harmonies."

        return analysis, strengths

    def _analyze_compatibility(
        self,
        outfit: GeneratedOutfit
    ) -> Tuple[str, List[str]]:
        """Analyze learned compatibility scores."""
        strengths = []
        scores = outfit.pairwise_scores

        if not scores:
            return "Compatibility analysis not available.", []

        avg_score = outfit.score
        min_score = min(scores.values())
        max_score = max(scores.values())

        # Find best and worst pairs
        best_pair = max(scores.items(), key=lambda x: x[1])
        worst_pair = min(scores.items(), key=lambda x: x[1])

        # Generate analysis
        if avg_score >= 0.8:
            strengths.append("Excellent overall compatibility")
            analysis = "This outfit has excellent compatibility. "
        elif avg_score >= 0.6:
            strengths.append("Good overall compatibility")
            analysis = "This outfit has good compatibility. "
        else:
            analysis = "This outfit has moderate compatibility. "

        # Add pair details
        if max_score - min_score > 0.2:
            analysis += f"The strongest pairing is between the items with score {best_pair[1]:.2f}. "
            if min_score < 0.5:
                analysis += f"Consider the pairing with score {worst_pair[1]:.2f}."
        else:
            strengths.append("Consistent compatibility across all items")
            analysis += "All items work well together with consistent scores."

        return analysis, strengths

    def _analyze_item_contributions(
        self,
        outfit: GeneratedOutfit
    ) -> Dict[str, str]:
        """Analyze how each item contributes to the outfit."""
        contributions = {}
        scores = outfit.pairwise_scores

        for item in outfit.items:
            item_id = item.item_id
            category = item.category

            # Find all pairs involving this item
            item_scores = [
                score for (id1, id2), score in scores.items()
                if id1 == item_id or id2 == item_id
            ]

            if item_scores:
                avg_score = sum(item_scores) / len(item_scores)

                if avg_score >= 0.8:
                    contribution = f"This {category} pairs excellently with other items (avg: {avg_score:.2f})"
                elif avg_score >= 0.6:
                    contribution = f"This {category} works well in the outfit (avg: {avg_score:.2f})"
                else:
                    contribution = f"This {category} has moderate compatibility (avg: {avg_score:.2f})"
            else:
                contribution = f"This {category} is part of the outfit"

            contributions[item_id] = contribution

        return contributions

    def _generate_suggestions(self, outfit: GeneratedOutfit) -> List[str]:
        """Generate suggestions for improving the outfit."""
        suggestions = []
        scores = outfit.pairwise_scores

        if not scores:
            return suggestions

        # Find weak pairings
        weak_pairs = [
            (pair, score) for pair, score in scores.items()
            if score < 0.5
        ]

        for (id1, id2), score in weak_pairs:
            suggestions.append(
                f"Consider alternatives for one of the items in the pairing "
                f"with score {score:.2f}"
            )

        # Check category coverage
        categories = [item.category for item in outfit.items]
        if "shoes" not in categories:
            suggestions.append("Consider adding shoes to complete the look")
        if "accessories" not in categories and len(outfit.items) >= 3:
            suggestions.append("Accessories could add a finishing touch")

        return suggestions[:3]  # Limit suggestions

    def _generate_summary(
        self,
        outfit: GeneratedOutfit,
        color_strengths: List[str],
        compat_strengths: List[str]
    ) -> str:
        """Generate a natural language summary."""
        categories = [item.category for item in outfit.items]
        category_list = ", ".join(categories)

        # Start with outfit composition
        summary = f"This {outfit.score:.0%} compatible outfit includes: {category_list}. "

        # Add key strengths
        all_strengths = color_strengths + compat_strengths
        if all_strengths:
            strengths_text = all_strengths[0].lower()
            summary += f"It works because {strengths_text}."

        return summary

    def explain_pair(
        self,
        item1: WardrobeItem,
        item2: WardrobeItem,
        score: float
    ) -> str:
        """
        Explain why two items work (or don't work) together.

        Args:
            item1, item2: The two items
            score: Compatibility score

        Returns:
            Natural language explanation
        """
        colors1 = getattr(item1, 'dominant_colors', [])
        colors2 = getattr(item2, 'dominant_colors', [])

        explanation = f"The {item1.category} and {item2.category} have a "

        if score >= 0.8:
            explanation += "strong compatibility. "
        elif score >= 0.6:
            explanation += "good compatibility. "
        elif score >= 0.4:
            explanation += "moderate compatibility. "
        else:
            explanation += "lower compatibility. "

        # Add color analysis
        if colors1 and colors2:
            harmony = self.color_analyzer.analyze_harmony(colors1, colors2)
            color1_name = colors1[0].name if colors1 else "unknown"
            color2_name = colors2[0].name if colors2 else "unknown"

            explanation += f"The {color1_name} {item1.category} and "
            explanation += f"{color2_name} {item2.category} create a "
            explanation += f"{harmony['harmony_type']} color relationship."

        return explanation

    def format_for_display(
        self,
        explanation: OutfitExplanation,
        verbose: bool = False
    ) -> str:
        """
        Format an explanation for display.

        Args:
            explanation: OutfitExplanation to format
            verbose: Include detailed breakdown

        Returns:
            Formatted string
        """
        lines = [
            "=" * 50,
            "OUTFIT RECOMMENDATION",
            "=" * 50,
            "",
            explanation.summary,
            "",
        ]

        if explanation.strengths:
            lines.append("Strengths:")
            for strength in explanation.strengths:
                lines.append(f"  + {strength}")
            lines.append("")

        if verbose:
            lines.extend([
                "Color Analysis:",
                f"  {explanation.color_analysis}",
                "",
                "Compatibility Analysis:",
                f"  {explanation.compatibility_analysis}",
                "",
            ])

            if explanation.item_contributions:
                lines.append("Item Contributions:")
                for item_id, contribution in explanation.item_contributions.items():
                    lines.append(f"  - {contribution}")
                lines.append("")

        if explanation.suggestions:
            lines.append("Suggestions:")
            for suggestion in explanation.suggestions:
                lines.append(f"  * {suggestion}")
            lines.append("")

        lines.extend([
            f"Overall Score: {explanation.score_breakdown['overall']:.1%}",
            "=" * 50,
        ])

        return "\n".join(lines)


class QuickExplainer:
    """
    Lightweight explainer for quick explanations without full analysis.
    """

    SCORE_DESCRIPTIONS = {
        (0.9, 1.0): "excellent",
        (0.8, 0.9): "great",
        (0.7, 0.8): "good",
        (0.6, 0.7): "decent",
        (0.5, 0.6): "moderate",
        (0.0, 0.5): "low"
    }

    def get_score_description(self, score: float) -> str:
        """Get a description for a compatibility score."""
        for (low, high), desc in self.SCORE_DESCRIPTIONS.items():
            if low <= score < high:
                return desc
        return "excellent" if score >= 1.0 else "low"

    def quick_explain(self, outfit: GeneratedOutfit) -> str:
        """
        Generate a one-line explanation.

        Args:
            outfit: GeneratedOutfit to explain

        Returns:
            Short explanation string
        """
        desc = self.get_score_description(outfit.score)
        categories = " + ".join(item.category for item in outfit.items)
        return f"{categories} - {desc} match ({outfit.score:.0%})"

    def quick_compare(
        self,
        outfit1: GeneratedOutfit,
        outfit2: GeneratedOutfit
    ) -> str:
        """
        Quick comparison of two outfits.

        Returns:
            Comparison string
        """
        diff = outfit1.score - outfit2.score

        if abs(diff) < 0.05:
            return "Both outfits are similarly compatible"
        elif diff > 0:
            return f"First outfit is {abs(diff):.0%} more compatible"
        else:
            return f"Second outfit is {abs(diff):.0%} more compatible"
