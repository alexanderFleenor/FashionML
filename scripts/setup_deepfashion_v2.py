#!/usr/bin/env python3
"""
Setup DeepFashion2 dataset for the outfit recommendation project.
VERSION 2: Crops individual items using bounding boxes.

This script:
1. Reads DeepFashion2 annotations with bounding boxes
2. Crops each item from the full image
3. Saves cropped images organized by category
4. Creates outfit pairs from images with multiple items
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
DEEPFASHION_DIR = Path("/Users/richardgardner/Library/CloudStorage/GoogleDrive-922766@lcps.org/My Drive/fashion2/train")
OUTPUT_DIR = Path("/Users/richardgardner/Developer/Fashion/data/deepfashion2")

# Map DeepFashion2 categories to our 5 categories
CATEGORY_MAP = {
    "short sleeve top": "tops",
    "long sleeve top": "tops",
    "vest": "tops",
    "sling": "tops",
    "short sleeve outwear": "tops",
    "long sleeve outwear": "tops",

    "trousers": "bottoms",
    "shorts": "bottoms",
    "skirt": "bottoms",

    "short sleeve dress": "dresses",
    "long sleeve dress": "dresses",
    "vest dress": "dresses",
    "sling dress": "dresses",
}

CATEGORY_IDS = {"tops": 0, "bottoms": 1, "dresses": 2, "shoes": 3, "accessories": 4}


def crop_item(image_path: Path, bbox: list, padding: int = 10) -> Image.Image:
    """
    Crop an item from an image using its bounding box.

    Args:
        image_path: Path to the full image
        bbox: [x1, y1, x2, y2] bounding box coordinates
        padding: Extra pixels around the crop

    Returns:
        Cropped PIL Image
    """
    img = Image.open(image_path).convert('RGB')
    x1, y1, x2, y2 = bbox

    # Add padding
    w, h = img.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return img.crop((x1, y1, x2, y2))


def main():
    print("Setting up DeepFashion2 dataset (with cropped items)...")
    print("This will take a few minutes...\n")

    images_dir = DEEPFASHION_DIR / "image"
    annos_dir = DEEPFASHION_DIR / "annos"

    if not images_dir.exists():
        print(f"ERROR: Images not found at {images_dir}")
        return

    # Clean and create output directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cat in ["tops", "bottoms", "dresses", "shoes", "accessories"]:
        (OUTPUT_DIR / "images" / cat).mkdir(parents=True, exist_ok=True)

    # Track items and outfits
    items = {}  # item_id -> {category, image_path, ...}
    items_by_category = defaultdict(list)
    outfit_pairs = []  # Images with 2+ items of DIFFERENT categories

    # Process annotations
    anno_files = sorted(annos_dir.glob("*.json"))
    print(f"Found {len(anno_files)} annotation files")

    # Limit for faster processing
    MAX_FILES = 10000
    anno_files = anno_files[:MAX_FILES]

    for anno_path in tqdm(anno_files, desc="Processing images"):
        try:
            with open(anno_path) as f:
                anno = json.load(f)
        except Exception:
            continue

        image_id = anno_path.stem
        image_path = images_dir / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        # Collect items from this image
        image_items = []

        for key in ["item1", "item2", "item3"]:
            if key not in anno:
                continue

            item_data = anno[key]
            df_category = item_data.get("category_name", "").lower()
            our_category = CATEGORY_MAP.get(df_category)
            bbox = item_data.get("bounding_box")

            if our_category is None or bbox is None:
                continue

            item_id = f"{image_id}_{key}"

            # Crop and save the item
            dest_path = OUTPUT_DIR / "images" / our_category / f"{item_id}.jpg"

            try:
                cropped = crop_item(image_path, bbox)
                # Resize to reasonable size
                cropped = cropped.resize((224, 224), Image.LANCZOS)
                cropped.save(dest_path, quality=90)
            except Exception as e:
                continue

            item_info = {
                "item_id": item_id,
                "category": our_category,
                "category_id": CATEGORY_IDS[our_category],
                "image_path": str(dest_path),
                "bbox": bbox,
            }

            items[item_id] = item_info
            items_by_category[our_category].append(item_id)
            image_items.append(item_info)

        # If image has 2+ items of DIFFERENT categories, they form an outfit pair
        if len(image_items) >= 2:
            categories = set(item["category"] for item in image_items)
            if len(categories) >= 2:  # Must have different categories
                outfit_pairs.append({
                    "outfit_id": image_id,
                    "items": [item["item_id"] for item in image_items]
                })

    print(f"\nProcessed items by category:")
    for cat, item_list in items_by_category.items():
        print(f"  {cat}: {len(item_list)}")

    print(f"\nReal outfit pairs (different categories): {len(outfit_pairs)}")

    # Only use real outfit pairs (no synthetic ones - they're meaningless)
    all_outfits = outfit_pairs
    random.seed(42)
    random.shuffle(all_outfits)

    # Split into train/val/test
    n_total = len(all_outfits)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_outfits = all_outfits[:n_train]
    val_outfits = all_outfits[n_train:n_train + n_val]
    test_outfits = all_outfits[n_train + n_val:]

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_outfits)}")
    print(f"  Val: {len(val_outfits)}")
    print(f"  Test: {len(test_outfits)}")

    # Save in Polyvore-compatible format
    def save_split(outfits, filename):
        data = []
        for outfit in outfits:
            outfit_items = []
            for item_id in outfit["items"]:
                if item_id in items:
                    outfit_items.append({
                        "item_id": item_id,
                        "category": items[item_id]["category"]
                    })
            if len(outfit_items) >= 2:
                data.append({
                    "set_id": outfit["outfit_id"],
                    "items": outfit_items
                })

        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)

        return len(data)

    n_train = save_split(train_outfits, "train.json")
    n_val = save_split(val_outfits, "valid.json")
    n_test = save_split(test_outfits, "test.json")

    print(f"\nFinal dataset:")
    print(f"  Train outfits: {n_train}")
    print(f"  Val outfits: {n_val}")
    print(f"  Test outfits: {n_test}")

    # Save item metadata
    with open(OUTPUT_DIR / "item_metadata.json", "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nDataset saved to {OUTPUT_DIR}")
    print("\nIMPORTANT: Update notebook 3 to use:")
    print(f"  DATA_DIR = Path('../data/deepfashion2')")
    print("\nAnd delete the old cache:")
    print("  rm -rf data/processed/*.pkl")


if __name__ == "__main__":
    main()
