"""Outfit generation + wear logging."""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.attributes.color_extractor import ColorHarmonyAnalyzer
from src.generation.generator import Wardrobe as WardrobeView

_harmony_analyzer = ColorHarmonyAnalyzer()

# Max items per category to feed the exhaustive enumerator. With 4 required
# categories filled, 12^4 = 20736 combos, which scores in ~1s with cached
# embeddings. Bumping this past ~15 makes the "complete" template slow.
MAX_ITEMS_PER_CATEGORY = 12


def _subsample_wardrobe(full, anchor_item=None, per_cat=MAX_ITEMS_PER_CATEGORY):
    """Return a fresh Wardrobe with at most `per_cat` items per category.
    The anchor (if provided) is always included. Sampling is random per call
    so consecutive requests naturally see different candidates."""
    sampled = WardrobeView()
    for cat in WardrobeView.CATEGORY_MAP.keys():
        items = list(full.get_items_by_category(cat))
        if anchor_item is not None and anchor_item.category == cat:
            items = [i for i in items if i.item_id != anchor_item.item_id]
            if len(items) > per_cat - 1:
                items = random.sample(items, per_cat - 1)
            items.append(anchor_item)
        else:
            if len(items) > per_cat:
                items = random.sample(items, per_cat)
        for it in items:
            sampled.add_item(it)
    return sampled

# Number of high-scoring candidates to keep before sampling for variety.
CANDIDATE_POOL_SIZE = 40
# Outfits within this many points of the top score are eligible for random
# selection.
SCORE_WINDOW = 0.15

from ..config import settings
from ..ml_service import get_ml
from ..wear_log import WearLog
from .auth import require_auth
from .items import ItemOut

log = logging.getLogger("fashion.outfits")

router = APIRouter(dependencies=[Depends(require_auth)])

_wear_log = WearLog(settings.WEAR_LOG_PATH)


class GenerateBody(BaseModel):
    anchor_item_id: Optional[str] = None
    template: Optional[str] = None  # "casual" | "full_casual" | "dress" | "complete"
    max_outfits: int = 3


class OutfitOut(BaseModel):
    items: List[ItemOut]
    score: float
    explanation: str
    summary: str
    harmony_type: Optional[str] = None  # analogous, complementary, neutral, etc.
    palette: List[str] = []  # hex codes from items, ordered by prominence


def _summarize_palette(outfit_items) -> tuple[Optional[str], List[str]]:
    """Aggregate item-level dominant colors into an outfit-level palette,
    and run ColorHarmonyAnalyzer on the primary colors of the two highest-
    weighted pieces (typically tops + bottoms)."""
    # Build a counter weighted by each color's percentage on each item.
    weighted = Counter()
    rgb_by_name = {}
    for item in outfit_items:
        for c in (item.dominant_colors or []):
            weighted[c.name] += c.percentage
            rgb_by_name.setdefault(c.name, c.rgb)
    palette = [
        "#{:02x}{:02x}{:02x}".format(*rgb_by_name[name])
        for name, _ in weighted.most_common(5)
    ]

    harmony = None
    item_color_lists = [getattr(i, "dominant_colors", None) for i in outfit_items]
    item_color_lists = [c for c in item_color_lists if c]
    if len(item_color_lists) >= 2:
        try:
            result = _harmony_analyzer.analyze_harmony(item_color_lists[0], item_color_lists[1])
            harmony = result.get("harmony_type")
        except Exception:
            harmony = None
    return harmony, palette


def _recency_penalty(item_ids: List[str]) -> float:
    """Small penalty for items that were worn recently.
    This changes ranking enough to keep the same pieces from showing up
    every time.
    """
    recent = _wear_log.recent(days=7)
    if not recent:
        return 0.0
    worn_recently = set()
    for entry in recent:
        worn_recently.update(entry.get("item_ids", []))
    overlap = sum(1 for i in item_ids if i in worn_recently)
    return 0.05 * overlap  # at most 0.05 per repeated item


@router.post("/today")
def todays_outfits(body: GenerateBody):
    ml = get_ml()
    wardrobe = ml.manager.wardrobe

    if len(wardrobe) == 0:
        raise HTTPException(status_code=400, detail="wardrobe is empty; add some items first")

    anchor = None
    if body.anchor_item_id:
        anchor = wardrobe.get_item(body.anchor_item_id)
        if anchor is None:
            raise HTTPException(status_code=404, detail="anchor item not found")

    template = None
    if body.template:
        for t in ml.generator.templates:
            if t.name == body.template:
                template = t
                break
        if template is None:
            raise HTTPException(status_code=400, detail=f"unknown template: {body.template}")
    else:
        # No explicit template: use the most complete outfit template the
        # current wardrobe can satisfy.
        priority = ["complete", "full_casual", "dress", "casual"]
        templates_by_name = {t.name: t for t in ml.generator.templates}
        for name in priority:
            t = templates_by_name.get(name)
            if t and all(
                len(wardrobe.get_items_by_category(cat)) > 0
                for cat in t.required_categories
            ):
                if anchor is None or anchor.category in t.required_categories:
                    template = t
                    break

    # Subsample to keep combination scoring fast when the closet has many items.
    sampled = _subsample_wardrobe(wardrobe, anchor_item=anchor)

    with ml.lock:
        candidates = ml.generator.generate_outfits(
            wardrobe=sampled,
            template=template,
            anchor_item=anchor,
            max_outfits=CANDIDATE_POOL_SIZE,
        )

    if not candidates:
        return {"outfits": []}

    # Dedupe combinations. Anchored outfits can repeat after the anchor replaces
    # another item from the same category.
    seen: set = set()
    deduped = []
    for o in candidates:
        key = tuple(sorted(o.get_item_ids()))
        if key not in seen:
            seen.add(key)
            deduped.append(o)

    # Apply recency penalty and sort.
    scored = sorted(
        deduped,
        key=lambda o: o.score - _recency_penalty(o.get_item_ids()),
        reverse=True,
    )

    # Pick a random subset from the top score band so Shuffle changes the list
    # without dropping too far in quality.
    best_score = scored[0].score - _recency_penalty(scored[0].get_item_ids())
    threshold = best_score - SCORE_WINDOW
    eligible = [o for o in scored if (o.score - _recency_penalty(o.get_item_ids())) >= threshold]

    if len(eligible) > body.max_outfits:
        ranked = random.sample(eligible, body.max_outfits)
    else:
        ranked = eligible[: body.max_outfits]

    # Show highest-scoring of the chosen first.
    ranked.sort(key=lambda o: -o.score)

    out = []
    for outfit in ranked:
        with ml.lock:
            explanation = ml.explainer.explain(outfit)
        item_outs = [ItemOut.from_wardrobe_item(i) for i in outfit.items]
        # The harmony analyzer needs the source wardrobe objects, not the API
        # response shape.
        harmony, palette = _summarize_palette(outfit.items)
        out.append(
            OutfitOut(
                items=item_outs,
                score=outfit.score,
                summary=explanation.summary,
                explanation=explanation.color_analysis,
                harmony_type=harmony,
                palette=palette,
            ).model_dump()
        )
    return {"outfits": out}


class LogBody(BaseModel):
    item_ids: List[str]


@router.post("/log")
def log_wear(body: LogBody):
    ml = get_ml()
    # Validate item IDs before writing to the log.
    for item_id in body.item_ids:
        if ml.manager.get_item(item_id) is None:
            raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    entry = _wear_log.append(body.item_ids)
    return entry


@router.get("/history")
def history():
    return {"entries": _wear_log.all()}
