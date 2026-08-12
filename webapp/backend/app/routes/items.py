"""CRUD for wardrobe items, wrapping the existing WardrobeManager."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

# iPhone photos are large and often use an EXIF rotation flag. Normalize them
# before inference so classification and color extraction stay fast and stable.
# The long edge is downscaled to this many pixels before any ML runs.
MAX_DIMENSION = 1024

from src.attributes.multicolor import MultiColorClassifier
from src.generation.generator import WardrobeItem

from ..ml_service import get_ml
from .auth import require_auth

# Reused across all items. The classifier only stores thresholds.
_pattern_classifier = MultiColorClassifier()

log = logging.getLogger("fashion.items")

router = APIRouter(dependencies=[Depends(require_auth)])

VALID_CATEGORIES = {"tops", "bottoms", "dresses", "shoes", "accessories"}


def _prepare_image(raw: bytes) -> Image.Image:
    """Decode upload, apply EXIF rotation, and downscale.
    Without exif_transpose, iPhone portrait photos arrive sideways (the model
    then sees the wrong orientation and often predicts the wrong category).
    Without downscale, K-means on a 12MP image takes 10+ seconds.
    """
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
    return img


def _clean_color_summary(s: str) -> str:
    """Remove repeated color names produced by the extractor.
    The K-means extractor often returns several near-identical shades of the
    same color (e.g. four "navy" buckets), which surfaces as
    "multi-color: navy, navy, navy, navy". Collapse those and downgrade the
    label if the list shrinks to one or two distinct names.
    """
    if not s:
        return s
    prefix, sep, rest = s.partition(":")
    if not sep:
        # "X and Y" form: collapse if both sides are the same name.
        if " and " in s:
            a, b = [p.strip() for p in s.split(" and ", 1)]
            if a == b:
                return f"solid {a}"
        return s
    colors: list[str] = []
    for c in rest.split(","):
        name = c.strip()
        if name and name not in colors:
            colors.append(name)
    if len(colors) == 1:
        return f"solid {colors[0]}"
    if len(colors) == 2:
        return f"{colors[0]} and {colors[1]}"
    return f"{prefix.strip()}: " + ", ".join(colors)


class DominantColorOut(BaseModel):
    name: str
    hex: str
    percentage: float


class ItemOut(BaseModel):
    item_id: str
    category: str  # current category (user-confirmed if overridden)
    predicted_category: str  # what the model originally said
    predicted_confidence: float
    color_summary: str
    color_pattern: Optional[str]  # "solid" | "two-tone" | "multi-color"
    dominant_colors: list[DominantColorOut]
    image_url: str

    @classmethod
    def from_wardrobe_item(cls, item) -> "ItemOut":
        meta = item.metadata or {}
        # Build color swatches from the cached dominant_colors. Each entry is a
        # DominantColor dataclass (has .name, .rgb tuple, .percentage).
        swatches = []
        for c in (item.dominant_colors or [])[:5]:
            try:
                rgb = tuple(int(x) for x in c.rgb)
                hexstr = "#{:02x}{:02x}{:02x}".format(*rgb)
                swatches.append(
                    DominantColorOut(name=c.name, hex=hexstr, percentage=float(c.percentage))
                )
            except (AttributeError, TypeError, ValueError):
                continue
        # Derive pattern from cached colors if not already in metadata (old items).
        pattern = meta.get("color_pattern")
        if pattern is None and item.dominant_colors:
            try:
                classification = _pattern_classifier.classify(item.dominant_colors)
                pattern = classification.pattern.value
            except Exception:
                pattern = None
        return cls(
            item_id=item.item_id,
            category=item.category,
            predicted_category=meta.get("predicted_category", item.category),
            predicted_confidence=float(meta.get("predicted_confidence", 1.0)),
            color_summary=_clean_color_summary(meta.get("color_summary", "")),
            color_pattern=pattern,
            dominant_colors=swatches,
            image_url=f"/api/items/{item.item_id}/image",
        )


@router.get("")
def list_items() -> dict:
    ml = get_ml()
    items = ml.manager.wardrobe.get_all_items()
    return {
        "items": [ItemOut.from_wardrobe_item(i).model_dump() for i in items],
        "summary": ml.manager.get_summary(),
    }


@router.post("", status_code=201)
async def add_item(
    image: UploadFile = File(...),
    category_override: Optional[str] = Form(None),
) -> dict:
    """Upload an image, classify it, and return the saved item.
    The user can then PATCH the category if the prediction was wrong."""
    ml = get_ml()
    raw = await image.read()
    try:
        pil_image = _prepare_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode image: {e}")

    if category_override and category_override not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {category_override}")

    # Single ML pass. This mirrors WardrobeManager.add_item persistence without
    # running the pipeline a second time.
    with ml.lock:
        attrs = ml.pipeline.process(pil_image)
        predicted_category = attrs.category
        predicted_confidence = float(attrs.category_confidence)

        final_category = category_override or predicted_category
        final_category_id = ml.manager.wardrobe.CATEGORY_MAP.get(final_category, 4)

        # Pick the next unused item_id (mirrors WardrobeManager's logic).
        existing_ids = set(ml.manager.wardrobe.items.keys())
        counter = len(existing_ids)
        while f"item_{counter}" in existing_ids:
            counter += 1
        item_id = f"item_{counter}"

        image_path = ml.manager.images_path / f"{item_id}.jpg"
        pil_image.save(image_path, "JPEG", quality=88)

        item = WardrobeItem(
            item_id=item_id,
            category=final_category,
            category_id=final_category_id,
            image_path=str(image_path),
            visual_features=attrs.visual_embedding,
            color_features=attrs.color_vector,
            dominant_colors=attrs.dominant_colors,
            metadata={
                "predicted_category": predicted_category,
                "predicted_confidence": predicted_confidence,
                "color_summary": _clean_color_summary(attrs.get_color_summary()),
                "color_pattern": attrs.color_pattern.value if attrs.color_pattern else None,
            },
        )
        ml.manager.wardrobe.add_item(item)
        ml.manager._save_cache(item)  # type: ignore[attr-defined]
        ml.manager._save_metadata()  # type: ignore[attr-defined]

    return ItemOut.from_wardrobe_item(item).model_dump()


class PatchBody(BaseModel):
    category: str


@router.patch("/{item_id}")
def update_item(item_id: str, body: PatchBody) -> dict:
    ml = get_ml()
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {body.category}")
    item = ml.manager.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")

    # The Wardrobe index is keyed by category, so update both the item and the
    # category lookup.
    old_category = item.category
    if old_category != body.category:
        with ml.lock:
            ml.manager.wardrobe.remove_item(item_id)
            item.category = body.category
            item.category_id = ml.manager.wardrobe.CATEGORY_MAP.get(body.category, 4)
            ml.manager.wardrobe.add_item(item)
            ml.manager._save_metadata()  # type: ignore[attr-defined]
    return ItemOut.from_wardrobe_item(item).model_dump()


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: str):
    ml = get_ml()
    with ml.lock:
        removed = ml.manager.remove_item(item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="item not found")
    return None


@router.get("/{item_id}/image")
def get_image(item_id: str):
    ml = get_ml()
    item = ml.manager.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    path = Path(item.image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path)
