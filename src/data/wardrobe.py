"""
Wardrobe Management Module

Manage user's personal wardrobe with persistence and organization.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import shutil

from ..generation.generator import WardrobeItem, Wardrobe
from ..attributes.pipeline import AttributePipeline, GarmentAttributes


class WardrobeManager:
    """
    Manage a user's wardrobe with persistence.

    Handles:
    - Adding items from images
    - Organizing by category
    - Saving/loading wardrobe state
    - Exporting embeddings for fast loading
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        pipeline: Optional[AttributePipeline] = None
    ):
        """
        Initialize the wardrobe manager.

        Args:
            storage_path: Directory for storing wardrobe data
            pipeline: AttributePipeline for processing new items
        """
        self.storage_path = Path(storage_path)
        self.pipeline = pipeline
        self.wardrobe = Wardrobe()

        # Create storage directories
        self.images_path = self.storage_path / "images"
        self.cache_path = self.storage_path / "cache"
        self.metadata_path = self.storage_path / "metadata.json"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)

        # Load existing wardrobe if present
        if self.metadata_path.exists():
            self._load_metadata()

    def _load_metadata(self):
        """Load wardrobe metadata from disk."""
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        for item_data in metadata.get("items", []):
            item_id = item_data["item_id"]

            # Load cached features
            cache_file = self.cache_path / f"{item_id}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    visual_features = cached["visual_features"]
                    color_features = cached["color_features"]
                    dominant_colors = cached.get("dominant_colors", [])
            else:
                visual_features = np.zeros(1280)
                color_features = np.zeros(15)
                dominant_colors = []

            item = WardrobeItem(
                item_id=item_id,
                category=item_data["category"],
                category_id=item_data["category_id"],
                image_path=str(self.images_path / item_data["image_filename"]),
                visual_features=visual_features,
                color_features=color_features,
                dominant_colors=dominant_colors,
                metadata=item_data.get("metadata", {})
            )

            self.wardrobe.add_item(item)

    def _save_metadata(self):
        """Save wardrobe metadata to disk."""
        items_data = []

        for item in self.wardrobe.get_all_items():
            image_filename = Path(item.image_path).name
            items_data.append({
                "item_id": item.item_id,
                "category": item.category,
                "category_id": item.category_id,
                "image_filename": image_filename,
                "metadata": item.metadata
            })

        metadata = {"items": items_data}

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_cache(self, item: WardrobeItem):
        """Save item features to cache."""
        cache_file = self.cache_path / f"{item.item_id}.pkl"

        cache_data = {
            "visual_features": item.visual_features,
            "color_features": item.color_features,
            "dominant_colors": item.dominant_colors
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def add_item(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        item_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> WardrobeItem:
        """
        Add a new item to the wardrobe.

        Args:
            image: Image data or path
            item_id: Optional ID (auto-generated if not provided)
            category: Optional category (auto-detected if not provided)
            metadata: Optional additional metadata

        Returns:
            Created WardrobeItem
        """
        if self.pipeline is None:
            raise ValueError("Pipeline required to add items")

        # Generate item_id
        if item_id is None:
            existing_ids = set(self.wardrobe.items.keys())
            counter = len(existing_ids)
            while f"item_{counter}" in existing_ids:
                counter += 1
            item_id = f"item_{counter}"

        # Process image
        attrs = self.pipeline.process(
            image,
            item_id=item_id,
            override_category=category
        )

        # Save image
        if isinstance(image, (str, Path)):
            image_path = self.images_path / f"{item_id}{Path(image).suffix}"
            shutil.copy(image, image_path)
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_path = self.images_path / f"{item_id}.jpg"
            image.save(image_path)

        # Create WardrobeItem
        item = WardrobeItem(
            item_id=item_id,
            category=attrs.category,
            category_id=attrs.category_id,
            image_path=str(image_path),
            visual_features=attrs.visual_embedding,
            color_features=attrs.color_vector,
            dominant_colors=attrs.dominant_colors,
            metadata=metadata or {}
        )

        # Add to wardrobe
        self.wardrobe.add_item(item)

        # Save to disk
        self._save_cache(item)
        self._save_metadata()

        return item

    def add_from_directory(
        self,
        directory: Union[str, Path],
        category: Optional[str] = None
    ) -> List[WardrobeItem]:
        """
        Add all images from a directory.

        Args:
            directory: Directory containing images
            category: Optional category for all items

        Returns:
            List of created WardrobeItems
        """
        directory = Path(directory)
        items = []

        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        for image_path in directory.iterdir():
            if image_path.suffix.lower() in extensions:
                try:
                    item = self.add_item(
                        image=str(image_path),
                        category=category
                    )
                    items.append(item)
                    print(f"Added: {image_path.name} -> {item.category}")
                except Exception as e:
                    print(f"Failed to add {image_path.name}: {e}")

        return items

    def remove_item(self, item_id: str) -> bool:
        """
        Remove an item from the wardrobe.

        Args:
            item_id: ID of item to remove

        Returns:
            True if removed, False if not found
        """
        item = self.wardrobe.get_item(item_id)
        if item is None:
            return False

        # Remove from wardrobe
        self.wardrobe.remove_item(item_id)

        # Remove files
        try:
            Path(item.image_path).unlink(missing_ok=True)
            (self.cache_path / f"{item_id}.pkl").unlink(missing_ok=True)
        except Exception:
            pass

        # Update metadata
        self._save_metadata()

        return True

    def get_wardrobe(self) -> Wardrobe:
        """Get the underlying Wardrobe instance."""
        return self.wardrobe

    def get_item(self, item_id: str) -> Optional[WardrobeItem]:
        """Get an item by ID."""
        return self.wardrobe.get_item(item_id)

    def get_items_by_category(self, category: str) -> List[WardrobeItem]:
        """Get all items in a category."""
        return self.wardrobe.get_items_by_category(category)

    def get_summary(self) -> Dict:
        """Get a summary of the wardrobe."""
        return {
            "total_items": len(self.wardrobe),
            "categories": self.wardrobe.get_category_counts()
        }

    def visualize_item(self, item_id: str) -> Optional[Image.Image]:
        """
        Load and return an item's image.

        Args:
            item_id: ID of item

        Returns:
            PIL Image or None if not found
        """
        item = self.wardrobe.get_item(item_id)
        if item is None:
            return None

        try:
            return Image.open(item.image_path)
        except Exception:
            return None


def create_sample_wardrobe(
    wardrobe_manager: WardrobeManager,
    sample_images_dir: Union[str, Path]
) -> WardrobeManager:
    """
    Create a sample wardrobe from a directory of images.

    Expects subdirectories for each category or flat structure.

    Args:
        wardrobe_manager: WardrobeManager instance
        sample_images_dir: Directory with sample images

    Returns:
        Updated WardrobeManager
    """
    sample_dir = Path(sample_images_dir)

    # Check for category subdirectories
    subdirs = [d for d in sample_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Organized by category
        for category_dir in subdirs:
            category = category_dir.name.lower()
            if category in Wardrobe.CATEGORY_MAP:
                wardrobe_manager.add_from_directory(
                    category_dir,
                    category=category
                )
    else:
        # Flat directory - auto-categorize
        wardrobe_manager.add_from_directory(sample_dir)

    return wardrobe_manager
