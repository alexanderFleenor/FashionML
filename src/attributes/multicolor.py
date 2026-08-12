"""
Multi-Color Classification Module

Classify garments as solid, two-tone, or multi-color based on
the distribution of their dominant colors.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path
from enum import Enum
from PIL import Image
from scipy.spatial import distance

from .color_extractor import ColorExtractor, DominantColor, ColorHarmonyAnalyzer


class ColorPattern(Enum):
    """Classification of garment color patterns."""
    SOLID = "solid"
    TWO_TONE = "two-tone"
    MULTI_COLOR = "multi-color"


@dataclass
class ColorClassification:
    """Result of color pattern classification."""
    pattern: ColorPattern
    primary_color: DominantColor
    secondary_color: Optional[DominantColor]
    accent_colors: List[DominantColor]
    confidence: float

    def __str__(self):
        colors_str = self.primary_color.name
        if self.secondary_color:
            colors_str += f" + {self.secondary_color.name}"
        if self.accent_colors:
            colors_str += f" + {len(self.accent_colors)} accent(s)"
        return f"{self.pattern.value}: {colors_str} ({self.confidence:.0%})"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pattern": self.pattern.value,
            "primary_color": self.primary_color.name,
            "secondary_color": self.secondary_color.name if self.secondary_color else None,
            "accent_colors": [c.name for c in self.accent_colors],
            "confidence": self.confidence
        }


class MultiColorClassifier:
    """
    Classify garments as solid, two-tone, or multi-color.

    Uses the distribution of dominant colors to determine the pattern type:
    - Solid: Primary color >= 85%
    - Two-tone: Primary 50-85% with significant secondary color
    - Multi-color: Primary < 50% or 3+ significant colors
    """

    def __init__(
        self,
        solid_threshold: float = 0.85,
        two_tone_threshold: float = 0.50,
        min_secondary_pct: float = 0.15,
        min_accent_pct: float = 0.05
    ):
        """
        Initialize the classifier.

        Args:
            solid_threshold: Min percentage for primary color to be 'solid'
            two_tone_threshold: Min percentage for primary color in 'two-tone'
            min_secondary_pct: Min percentage for a secondary color
            min_accent_pct: Min percentage for an accent color
        """
        self.solid_threshold = solid_threshold
        self.two_tone_threshold = two_tone_threshold
        self.min_secondary_pct = min_secondary_pct
        self.min_accent_pct = min_accent_pct

    def classify(self, colors: List[DominantColor]) -> ColorClassification:
        """
        Classify the color pattern from extracted dominant colors.

        Args:
            colors: List of DominantColor objects (sorted by percentage, descending)

        Returns:
            ColorClassification with pattern type and color breakdown
        """
        if not colors:
            raise ValueError("No colors provided for classification")

        primary = colors[0]
        secondary = colors[1] if len(colors) > 1 else None

        # Count significant colors
        significant_colors = [c for c in colors if c.percentage >= self.min_accent_pct]

        # Determine pattern type
        if primary.percentage >= self.solid_threshold:
            # Solid: one dominant color
            pattern = ColorPattern.SOLID
            confidence = primary.percentage
            secondary_out = None
            accents = []

        elif primary.percentage >= self.two_tone_threshold:
            # Check if there's a significant secondary color
            if secondary and secondary.percentage >= self.min_secondary_pct:
                pattern = ColorPattern.TWO_TONE
                confidence = primary.percentage + secondary.percentage
                secondary_out = secondary
                accents = [c for c in colors[2:] if c.percentage >= self.min_accent_pct]
            else:
                # Primary is dominant but not quite solid
                pattern = ColorPattern.SOLID
                confidence = primary.percentage
                secondary_out = None
                accents = []

        else:
            # Multi-color: many significant colors or no single dominant
            pattern = ColorPattern.MULTI_COLOR
            confidence = sum(c.percentage for c in significant_colors)
            secondary_out = secondary if secondary and secondary.percentage >= self.min_secondary_pct else None
            accents = [c for c in colors[2:] if c.percentage >= self.min_accent_pct]

        return ColorClassification(
            pattern=pattern,
            primary_color=primary,
            secondary_color=secondary_out,
            accent_colors=accents,
            confidence=confidence
        )


class EnhancedColorExtractor(ColorExtractor):
    """
    Extended ColorExtractor with multi-color classification support.

    Adds the ability to classify garments as solid, two-tone, or multi-color
    based on the distribution of dominant colors.
    """

    def __init__(
        self,
        n_colors: int = 5,
        color_space: str = "LAB",
        min_percentage: float = 0.05,
        remove_background: bool = True,
        filter_skin_tones: bool = True,
        solid_threshold: float = 0.85,
        two_tone_threshold: float = 0.50
    ):
        """
        Initialize the enhanced color extractor.

        Args:
            n_colors: Maximum number of dominant colors to extract
            color_space: Color space for clustering ("LAB" or "HSV")
            min_percentage: Minimum percentage threshold for including a color
            remove_background: Whether to attempt background removal
            filter_skin_tones: Whether to filter out skin tones
            solid_threshold: Min percentage for primary color to be 'solid'
            two_tone_threshold: Min percentage for primary color in 'two-tone'
        """
        super().__init__(
            n_colors=n_colors,
            color_space=color_space,
            min_percentage=min_percentage,
            remove_background=remove_background,
            filter_skin_tones=filter_skin_tones
        )

        self.mc_classifier = MultiColorClassifier(
            solid_threshold=solid_threshold,
            two_tone_threshold=two_tone_threshold,
            min_secondary_pct=min_percentage,
            min_accent_pct=min_percentage
        )

    def extract_with_classification(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Tuple[List[DominantColor], ColorClassification]:
        """
        Extract colors and classify the color pattern.

        Args:
            image: Input image

        Returns:
            Tuple of (dominant_colors, classification)
        """
        colors = self.extract(image)
        classification = self.mc_classifier.classify(colors)
        return colors, classification

    def is_solid(self, image: Union[np.ndarray, Image.Image, str, Path]) -> bool:
        """Quick check if an image is a solid color."""
        _, classification = self.extract_with_classification(image)
        return classification.pattern == ColorPattern.SOLID

    def is_multicolor(self, image: Union[np.ndarray, Image.Image, str, Path]) -> bool:
        """Quick check if an image is multi-colored."""
        _, classification = self.extract_with_classification(image)
        return classification.pattern == ColorPattern.MULTI_COLOR

    def is_two_tone(self, image: Union[np.ndarray, Image.Image, str, Path]) -> bool:
        """Quick check if an image is two-tone."""
        _, classification = self.extract_with_classification(image)
        return classification.pattern == ColorPattern.TWO_TONE


class EnhancedHarmonyAnalyzer(ColorHarmonyAnalyzer):
    """
    Enhanced color harmony analyzer that considers multiple colors.

    Extends the base analyzer to handle:
    - Solid + Solid pairing
    - Solid + Multi-color pairing
    - Multi-color + Multi-color pairing
    """

    def __init__(self, color_match_threshold: float = 30.0):
        """
        Initialize the enhanced harmony analyzer.

        Args:
            color_match_threshold: LAB distance threshold for matching colors
        """
        super().__init__()
        self.color_match_threshold = color_match_threshold

    def analyze_harmony_enhanced(
        self,
        classification1: ColorClassification,
        classification2: ColorClassification
    ) -> dict:
        """
        Analyze harmony considering the full color classification.

        Args:
            classification1: ColorClassification for first item
            classification2: ColorClassification for second item

        Returns:
            Dictionary with harmony analysis
        """
        p1 = classification1.pattern
        p2 = classification2.pattern

        # Get all significant colors for each item
        colors1 = [classification1.primary_color]
        if classification1.secondary_color:
            colors1.append(classification1.secondary_color)
        colors1.extend(classification1.accent_colors)

        colors2 = [classification2.primary_color]
        if classification2.secondary_color:
            colors2.append(classification2.secondary_color)
        colors2.extend(classification2.accent_colors)

        # Base harmony from primary colors
        base_harmony = self.analyze_harmony(
            [classification1.primary_color],
            [classification2.primary_color]
        )

        # Adjust based on pattern combination
        score = base_harmony['score']
        pattern_note = ""

        # Solid + Solid: Use base harmony
        if p1 == ColorPattern.SOLID and p2 == ColorPattern.SOLID:
            pattern_note = "Both solid - classic pairing"

        # Solid + Multi-color: Check if solid color appears in multi-color
        elif p1 == ColorPattern.SOLID and p2 in (ColorPattern.TWO_TONE, ColorPattern.MULTI_COLOR):
            if self._color_matches_any(classification1.primary_color, colors2):
                score += 0.1
                pattern_note = "Solid matches multi-color element - cohesive"
            else:
                pattern_note = "Solid with patterned - check contrast"

        elif p2 == ColorPattern.SOLID and p1 in (ColorPattern.TWO_TONE, ColorPattern.MULTI_COLOR):
            if self._color_matches_any(classification2.primary_color, colors1):
                score += 0.1
                pattern_note = "Solid matches multi-color element - cohesive"
            else:
                pattern_note = "Solid with patterned - check contrast"

        # Multi-color + Multi-color: Generally harder to match
        elif p1 in (ColorPattern.TWO_TONE, ColorPattern.MULTI_COLOR) and \
             p2 in (ColorPattern.TWO_TONE, ColorPattern.MULTI_COLOR):
            shared = self._count_shared_colors(colors1, colors2)
            if shared >= 1:
                score += 0.05 * shared
                pattern_note = f"Patterns share {shared} color(s) - coordinated"
            else:
                score -= 0.1
                pattern_note = "Multiple patterns without shared colors - busy"

        score = min(1.0, max(0.0, score))

        return {
            'harmony_type': base_harmony['harmony_type'],
            'score': score,
            'pattern_combination': f"{p1.value} + {p2.value}",
            'pattern_note': pattern_note,
            'colors_item1': [c.name for c in colors1],
            'colors_item2': [c.name for c in colors2]
        }

    def _color_matches_any(
        self,
        color: DominantColor,
        color_list: List[DominantColor],
        threshold: Optional[float] = None
    ) -> bool:
        """Check if a color closely matches any color in a list (using LAB distance)."""
        threshold = threshold or self.color_match_threshold
        color_lab = np.array(color.lab)
        for c in color_list:
            if distance.euclidean(color_lab, np.array(c.lab)) < threshold:
                return True
        return False

    def _count_shared_colors(
        self,
        colors1: List[DominantColor],
        colors2: List[DominantColor],
        threshold: Optional[float] = None
    ) -> int:
        """Count how many colors are shared between two lists."""
        threshold = threshold or self.color_match_threshold
        shared = 0
        for c1 in colors1:
            if self._color_matches_any(c1, colors2, threshold):
                shared += 1
        return shared
