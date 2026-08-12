"""
Attributes module for garment feature extraction.
"""

from .color_extractor import ColorExtractor, DominantColor, ColorHarmonyAnalyzer
from .pipeline import AttributePipeline, GarmentAttributes
from .multicolor import (
    ColorPattern,
    ColorClassification,
    MultiColorClassifier,
    EnhancedColorExtractor,
    EnhancedHarmonyAnalyzer
)

__all__ = [
    # Color extraction
    "ColorExtractor",
    "DominantColor",
    "ColorHarmonyAnalyzer",
    # Pipeline
    "AttributePipeline",
    "GarmentAttributes",
    # Multi-color classification
    "ColorPattern",
    "ColorClassification",
    "MultiColorClassifier",
    "EnhancedColorExtractor",
    "EnhancedHarmonyAnalyzer",
]
