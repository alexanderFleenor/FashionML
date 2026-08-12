"""Loads the trained ML models once at startup and exposes a thin API
the routes can call. Mirrors the wiring from notebook 05.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

# Imports from the core ML project (mounted as /app/src or installed via
# PYTHONPATH).
from src.recognition.classifier import GarmentClassifier
from src.compatibility.model import SiameseCompatibilityNet
from src.compatibility.scorer import CompatibilityScorer
from src.attributes.color_extractor import ColorHarmonyAnalyzer
from src.attributes.multicolor import EnhancedColorExtractor
from src.attributes.pipeline import AttributePipeline
from src.generation.generator import OutfitGenerator, OutfitTemplate
from src.generation.explainer import OutfitExplainer
from src.data.wardrobe import WardrobeManager

log = logging.getLogger("fashion.ml")


@dataclass
class MLService:
    """Bundle of loaded ML components + the wardrobe manager."""

    classifier: GarmentClassifier
    color_extractor: EnhancedColorExtractor
    compat_model: SiameseCompatibilityNet
    pipeline: AttributePipeline
    scorer: CompatibilityScorer
    generator: OutfitGenerator
    explainer: OutfitExplainer
    manager: WardrobeManager
    # The model objects are shared across requests, so route handlers use this
    # lock around inference.
    lock: threading.Lock

    @classmethod
    def load(cls, models_dir: Path, wardrobe_dir: Path) -> "MLService":
        classifier_path = models_dir / "garment_classifier.pth"
        compat_path = models_dir / "compatibility_model.pth"

        if classifier_path.exists():
            log.info("Loading garment classifier from %s", classifier_path)
            classifier = GarmentClassifier.load(classifier_path)
        else:
            log.warning("No trained classifier at %s; falling back to pretrained backbone", classifier_path)
            classifier = GarmentClassifier(backbone="efficientnet_b0", pretrained=True)

        color_extractor = EnhancedColorExtractor(
            n_colors=5,
            color_space="LAB",
            solid_threshold=0.85,
            two_tone_threshold=0.50,
        )

        if compat_path.exists():
            log.info("Loading compatibility model from %s", compat_path)
            compat_model = SiameseCompatibilityNet.load(str(compat_path))
        else:
            log.warning("No trained compatibility model at %s; using untrained net", compat_path)
            compat_model = SiameseCompatibilityNet()

        pipeline = AttributePipeline(
            classifier=classifier,
            color_extractor=color_extractor,
            enable_multicolor=True,
        )
        scorer = CompatibilityScorer(
            model=compat_model,
            classifier=classifier,
            color_extractor=color_extractor,
        )
        # Keep the cutoff forgiving so a small wardrobe can still return outfits.
        generator = OutfitGenerator(scorer=scorer, min_compatibility=0.3)
        explainer = OutfitExplainer(color_analyzer=ColorHarmonyAnalyzer())

        manager = WardrobeManager(storage_path=wardrobe_dir, pipeline=pipeline)

        return cls(
            classifier=classifier,
            color_extractor=color_extractor,
            compat_model=compat_model,
            pipeline=pipeline,
            scorer=scorer,
            generator=generator,
            explainer=explainer,
            manager=manager,
            lock=threading.Lock(),
        )


class _Holder:
    """Module-level holder so route handlers can import a stable reference
    that gets populated during FastAPI startup."""

    instance: Optional[MLService] = None


ml_service_holder = _Holder()


def get_ml() -> MLService:
    if ml_service_holder.instance is None:
        raise RuntimeError("ML service not initialized; lifespan startup failed")
    return ml_service_holder.instance
