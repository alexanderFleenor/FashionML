#!/usr/bin/env python3
"""
Setup DeepFashion2 dataset for the outfit recommendation project.

This script:
1. Reads DeepFashion2 annotations
2. Organizes images by category
3. Creates outfit pairs from images with multiple items
4. Generates compatible format for the compatibility model
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

# === CONFIGURATION ===
DEEPFASHION_DIR = Path("/Users/richardgardner/Library/CloudStorage/GoogleDrive-922766@lcps.org/My Drive/fashion2/train")
OUTPUT_DIR = Path("/Users/richardgardner/Developer/Fashion/data/deepfashion")

# Map DeepFashion2 categories to our 5 categories
CATEGORY_MAP = {
    "short sleeve top": "tops",
    "long sleeve top": "tops",
    "vest": "tops",
    "sling": "tops",

    "trousers": "bottoms",
    "shorts": "bottoms",
    "skirt": "bottoms",

    "short sleeve dress": "dresses",
    "long sleeve dress": "dresses",
    "vest dress": "dresses",
    "sling dress": "dresses",

    "short sleeve outwear": "tops",
    "long sleeve outwear": "tops",
}

CATEGORY_IDS = {"tops": 0, "bottoms": 1, "dresses": 2, "shoes": 3, "accessories": 4}


def main():
    print("Setting up DeepFashion2 dataset...")

    images_dir = DEEPFASHION_DIR / "image"
    annos_dir = DEEPFASHION_DIR / "annos"

    if not images_dir.exists():
        print(f"ERROR: Images not found at {images_dir}")
        return

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    for cat in ["tops", "bottoms", "dresses", "shoes", "accessories"]:
        (OUTPUT_DIR / "images" / cat).mkdir(exist_ok=True)

    # Track items and outfits
    items = {}  # item_id -> {category, image_path, ...}
    outfits = []  # List of outfit dicts with item_ids

    # Process annotations
    anno_files = sorted(annos_dir.glob("*.json"))
    print(f"Found {len(anno_files)} annotation files")

    # Limit for faster processing (remove for full dataset)
    MAX_FILES = 10000
    anno_files = anno_files[:MAX_FILES]

    items_by_category = defaultdict(list)
    outfit_pairs = []  # Images with 2+ items

    for i, anno_path in enumerate(anno_files):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(anno_files)}...")

        try:
            with open(anno_path) as f:
                anno = json.load(f)
        except Exception as e:
            continue

        image_id = anno_path.stem  # e.g., "000001"
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

            if our_category is None:
                continue

            item_id = f"{image_id}_{key}"

            # Copy image to category folder (symlink to save space)
            dest_path = OUTPUT_DIR / "images" / our_category / f"{item_id}.jpg"
            if not dest_path.exists():
                try:
                    # Use symlink to save disk space
                    dest_path.symlink_to(image_path)
                except:
                    # Fall back to copy if symlink fails
                    shutil.copy2(image_path, dest_path)

            item_info = {
                "item_id": item_id,
                "category": our_category,
                "category_id": CATEGORY_IDS[our_category],
                "image_path": str(dest_path),
                "source_image": str(image_path),
                "bbox": item_data.get("bounding_box"),
            }

            items[item_id] = item_info
            items_by_category[our_category].append(item_id)
            image_items.append(item_info)

        # If image has 2+ items, they form an outfit pair
        if len(image_items) >= 2:
            outfit_pairs.append({
                "outfit_id": image_id,
                "items": [item["item_id"] for item in image_items]
            })

    print(f"\nProcessed items by category:")
    for cat, item_list in items_by_category.items():
        print(f"  {cat}: {len(item_list)}")

    print(f"\nOutfit pairs (images with 2+ items): {len(outfit_pairs)}")

    # Create synthetic outfits by pairing tops with bottoms
    print("\nCreating synthetic outfit combinations...")
    synthetic_outfits = []

    tops = items_by_category["tops"]
    bottoms = items_by_category["bottoms"]
    dresses = items_by_category["dresses"]

    # Random top + bottom combinations
    random.seed(42)
    n_synthetic = min(5000, len(tops) * len(bottoms) // 10)

    for i in range(n_synthetic):
        top_id = random.choice(tops)
        bottom_id = random.choice(bottoms)
        synthetic_outfits.append({
            "outfit_id": f"synthetic_{i}",
            "items": [top_id, bottom_id]
        })

    # Combine real and synthetic outfits
    all_outfits = outfit_pairs + synthetic_outfits
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
            data.append({
                "set_id": outfit["outfit_id"],
                "items": [
                    {"item_id": item_id, "category": items[item_id]["category"]}
                    for item_id in outfit["items"]
                    if item_id in items
                ]
            })

        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)

    save_split(train_outfits, "train.json")
    save_split(val_outfits, "valid.json")
    save_split(test_outfits, "test.json")

    # Save item metadata
    with open(OUTPUT_DIR / "item_metadata.json", "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nDataset saved to {OUTPUT_DIR}")
    print("\nTo use this dataset, update notebook 3 to use:")
    print(f'  DATA_DIR = Path("{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
