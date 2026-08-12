"""
Attribute Pipeline Module

Combined pipeline for extracting all attributes from garment images,
including color pattern classification (solid, two-tone, multi-color).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
from pathlib import Path
from PIL import Image

from .color_extractor import ColorExtractor, DominantColor
from .multicolor import (
    ColorPattern,
    ColorClassification,
    MultiColorClassifier,
    EnhancedColorExtractor
)


@dataclass
class GarmentAttributes:
    """Complete attribute representation for a garment."""
    item_id: str
    category: str
    category_id: int
    category_confidence: float
    dominant_colors: List[DominantColor]
    visual_embedding: np.ndarray
    color_vector: np.ndarray
    # Color pattern classification
    color_pattern: Optional[ColorPattern] = None
    color_classification: Optional[ColorClassification] = None
    # Legacy pattern detection (for trained pattern models)
    pattern: Optional[str] = None
    pattern_confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def get_features(self):
        """Get feature tuple for compatibility scoring."""
        return (self.visual_embedding, self.color_vector, self.category_id)

    def is_solid(self) -> bool:
        """Check if this garment is a solid color."""
        return self.color_pattern == ColorPattern.SOLID

    def is_multicolor(self) -> bool:
        """Check if this garment is multi-colored."""
        return self.color_pattern == ColorPattern.MULTI_COLOR

    def is_two_tone(self) -> bool:
        """Check if this garment is two-tone."""
        return self.color_pattern == ColorPattern.TWO_TONE

    def get_color_summary(self) -> str:
        """Get a human-readable color summary."""
        if not self.color_classification:
            if self.dominant_colors:
                return self.dominant_colors[0].name
            return "unknown"

        cc = self.color_classification
        if cc.pattern == ColorPattern.SOLID:
            return f"solid {cc.primary_color.name}"
        elif cc.pattern == ColorPattern.TWO_TONE:
            return f"{cc.primary_color.name} and {cc.secondary_color.name}"
        else:
            colors = [cc.primary_color.name]
            if cc.secondary_color:
                colors.append(cc.secondary_color.name)
            colors.extend([c.name for c in cc.accent_colors[:2]])
            return "multi-color: " + ", ".join(colors)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "category": self.category,
            "category_id": self.category_id,
            "category_confidence": self.category_confidence,
            "dominant_colors": [
                {"name": c.name, "rgb": c.rgb, "percentage": c.percentage}
                for c in self.dominant_colors
            ],
            "color_pattern": self.color_pattern.value if self.color_pattern else None,
            "color_summary": self.get_color_summary(),
            "pattern": self.pattern,
            "metadata": self.metadata
        }


class AttributePipeline:
    """
    Combined pipeline for extracting all attributes from a garment image.

    Orchestrates:
    1. Category classification (GarmentClassifier)
    2. Color extraction (ColorExtractor)
    3. Color pattern classification (solid/two-tone/multi-color)
    4. Visual feature extraction (from classifier backbone)
    """

    CATEGORY_MAP = {
        "tops": 0,
        "bottoms": 1,
        "dresses": 2,
        "shoes": 3,
        "accessories": 4
    }

    def __init__(
        self,
        classifier,
        color_extractor: Optional[ColorExtractor] = None,
        pattern_detector=None,
        enable_multicolor: bool = True,
        solid_threshold: float = 0.85,
        two_tone_threshold: float = 0.50
    ):
        """
        Initialize the pipeline.

        Args:
            classifier: GarmentClassifier instance
            color_extractor: ColorExtractor instance (creates default if None)
            pattern_detector: Optional PatternDetector instance
            enable_multicolor: Whether to enable multi-color classification
            solid_threshold: Min percentage for primary color to be 'solid'
            two_tone_threshold: Min percentage for primary color in 'two-tone'
        """
        self.classifier = classifier
        self.enable_multicolor = enable_multicolor

        # Use EnhancedColorExtractor if multicolor is enabled
        if enable_multicolor:
            if isinstance(color_extractor, EnhancedColorExtractor):
                self.color_extractor = color_extractor
            else:
                # Create enhanced extractor with same settings
                self.color_extractor = EnhancedColorExtractor(
                    n_colors=color_extractor.n_colors if color_extractor else 5,
                    solid_threshold=solid_threshold,
                    two_tone_threshold=two_tone_threshold
                )
        else:
            self.color_extractor = color_extractor or ColorExtractor()

        self.pattern_detector = pattern_detector
        self.mc_classifier = MultiColorClassifier(
            solid_threshold=solid_threshold,
            two_tone_threshold=two_tone_threshold
        ) if enable_multicolor else None

    def process(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        item_id: Optional[str] = None,
        override_category: Optional[str] = None
    ) -> GarmentAttributes:
        """
        Extract all attributes from a single garment image.

        Args:
            image: Input image
            item_id: Optional identifier for the item
            override_category: If provided, use this instead of predicted category

        Returns:
            GarmentAttributes with all extracted information
        """
        # Generate item_id if not provided
        if item_id is None:
            if isinstance(image, (str, Path)):
                item_id = Path(image).stem
            else:
                item_id = f"item_{id(image)}"

        # Classification
        if override_category:
            category = override_category
            category_id = self.CATEGORY_MAP.get(category, 4)
            category_confidence = 1.0
        else:
            result = self.classifier.predict(image)
            category = result.category
            category_id = self.CATEGORY_MAP.get(category, 4)
            category_confidence = result.confidence

        # Visual features
        visual_embedding = self.classifier.get_features(image)

        # Color extraction with classification
        if self.enable_multicolor and isinstance(self.color_extractor, EnhancedColorExtractor):
            dominant_colors, color_classification = self.color_extractor.extract_with_classification(image)
            color_pattern = color_classification.pattern
        else:
            dominant_colors = self.color_extractor.extract(image)
            if self.mc_classifier and dominant_colors:
                color_classification = self.mc_classifier.classify(dominant_colors)
                color_pattern = color_classification.pattern
            else:
                color_classification = None
                color_pattern = None

        color_vector = self.color_extractor.get_color_vector(image)

        # Pattern detection (optional - for trained pattern models)
        pattern = None
        pattern_confidence = None
        if self.pattern_detector:
            pattern, pattern_confidence, _ = self.pattern_detector.detect(image)

        return GarmentAttributes(
            item_id=item_id,
            category=category,
            category_id=category_id,
            category_confidence=category_confidence,
            dominant_colors=dominant_colors,
            visual_embedding=visual_embedding,
            color_vector=color_vector,
            color_pattern=color_pattern,
            color_classification=color_classification,
            pattern=pattern,
            pattern_confidence=pattern_confidence
        )

    def process_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        item_ids: Optional[List[str]] = None
    ) -> List[GarmentAttributes]:
        """
        Process multiple images.

        Args:
            images: List of input images
            item_ids: Optional list of identifiers

        Returns:
            List of GarmentAttributes
        """
        if item_ids is None:
            item_ids = [None] * len(images)

        return [
            self.process(img, item_id)
            for img, item_id in zip(images, item_ids)
        ]
