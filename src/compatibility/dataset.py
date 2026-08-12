"""
Polyvore Dataset Module

Load and preprocess the Polyvore Outfits dataset for training
outfit compatibility models.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


@dataclass
class OutfitItem:
    """Represents a single item in an outfit."""
    item_id: str
    category: str
    category_id: int
    image_path: Optional[str] = None


@dataclass
class Outfit:
    """Represents a complete outfit."""
    outfit_id: str
    items: List[OutfitItem]


# Category mapping for Polyvore dataset
POLYVORE_CATEGORY_MAP = {
    # Tops
    "tops": 0,
    "blouse": 0,
    "shirt": 0,
    "sweater": 0,
    "cardigan": 0,
    "tee": 0,
    "tank": 0,
    "top": 0,

    # Bottoms
    "bottoms": 1,
    "pants": 1,
    "jeans": 1,
    "shorts": 1,
    "skirt": 1,
    "trousers": 1,

    # Dresses
    "dresses": 2,
    "dress": 2,
    "jumpsuit": 2,
    "romper": 2,

    # Shoes
    "shoes": 3,
    "shoe": 3,
    "boots": 3,
    "sneakers": 3,
    "heels": 3,
    "sandals": 3,
    "flats": 3,

    # Accessories
    "accessories": 4,
    "bag": 4,
    "bags": 4,
    "jewelry": 4,
    "hat": 4,
    "scarf": 4,
    "belt": 4,
    "sunglasses": 4,
    "watch": 4,
}

CATEGORY_NAMES = ["tops", "bottoms", "dresses", "shoes", "accessories"]


class PolyvoreDataset:
    """
    Load and manage the Polyvore Outfits dataset.

    The dataset contains outfit compositions with item metadata,
    which can be used to generate compatible/incompatible pairs
    for training compatibility models.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[T.Compose] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to Polyvore dataset directory
            split: Dataset split ("train", "valid", or "test")
            transform: Image transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._default_transform()

        self.outfits: List[Outfit] = []
        self.items: Dict[str, OutfitItem] = {}
        self.items_by_category: Dict[int, List[str]] = {i: [] for i in range(5)}

        self._load_data()

    def _default_transform(self) -> T.Compose:
        """Default image transform."""
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_data(self):
        """Load outfit compositions from JSON file."""
        json_path = self.data_dir / f"{self.split}_no_dup.json"

        if not json_path.exists():
            # Try alternative naming
            json_path = self.data_dir / f"{self.split}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"Could not find dataset file at {json_path}. "
                f"Please download the Polyvore dataset."
            )

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Load item metadata if available
        metadata_path = self.data_dir / "item_metadata.json"
        item_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                item_metadata = json.load(f)

        # Parse outfits
        for outfit_data in data:
            outfit_id = outfit_data.get("set_id", outfit_data.get("outfit_id", str(len(self.outfits))))
            items_data = outfit_data.get("items", outfit_data.get("item_ids", []))

            outfit_items = []
            for item_data in items_data:
                if isinstance(item_data, dict):
                    item_id = str(item_data.get("item_id", item_data.get("id", "")))
                    category = item_data.get("category", item_data.get("categoryid", ""))
                else:
                    item_id = str(item_data)
                    category = item_metadata.get(item_id, {}).get("category", "accessories")

                # Map category to ID
                category_lower = str(category).lower()
                category_id = POLYVORE_CATEGORY_MAP.get(category_lower, 4)  # Default to accessories

                # Determine image path - first check metadata, then search
                image_path = item_metadata.get(item_id, {}).get("image_path")
                if image_path is None or not Path(image_path).exists():
                    image_path = self._find_image_path(item_id, category_lower)

                item = OutfitItem(
                    item_id=item_id,
                    category=CATEGORY_NAMES[category_id],
                    category_id=category_id,
                    image_path=image_path
                )

                outfit_items.append(item)
                self.items[item_id] = item
                if item_id not in self.items_by_category[category_id]:
                    self.items_by_category[category_id].append(item_id)

            if len(outfit_items) >= 2:  # Only keep outfits with at least 2 items
                self.outfits.append(Outfit(
                    outfit_id=outfit_id,
                    items=outfit_items
                ))

        print(f"Loaded {len(self.outfits)} outfits with {len(self.items)} unique items")

    def _find_image_path(self, item_id: str, category: str = None) -> Optional[str]:
        """Find the image path for an item."""
        # Try different possible image locations
        possible_paths = [
            self.data_dir / "images" / f"{item_id}.jpg",
            self.data_dir / "images" / item_id / "image.jpg",
            self.data_dir / "images" / f"{item_id}.png",
        ]

        # Also try category subdirectories
        if category:
            possible_paths.insert(0, self.data_dir / "images" / category / f"{item_id}.jpg")
        for cat in CATEGORY_NAMES:
            possible_paths.append(self.data_dir / "images" / cat / f"{item_id}.jpg")

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    def load_image(self, item_id: str) -> Optional[torch.Tensor]:
        """Load and transform an item's image."""
        item = self.items.get(item_id)
        if item is None or item.image_path is None:
            return None

        try:
            image = Image.open(item.image_path).convert("RGB")
            return self.transform(image)
        except Exception:
            return None

    def get_compatible_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all compatible item pairs (items from the same outfit).

        Returns:
            List of (item_id1, item_id2) tuples
        """
        pairs = []
        for outfit in self.outfits:
            items = outfit.items
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pairs.append((items[i].item_id, items[j].item_id))
        return pairs

    def get_incompatible_pairs(self, n_pairs: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Generate incompatible item pairs (items from different outfits).

        Args:
            n_pairs: Number of pairs to generate (default: same as compatible pairs)

        Returns:
            List of (item_id1, item_id2) tuples
        """
        compatible_pairs = set(self.get_compatible_pairs())
        if n_pairs is None:
            n_pairs = len(compatible_pairs)

        all_item_ids = list(self.items.keys())
        pairs = []
        attempts = 0
        max_attempts = n_pairs * 10

        while len(pairs) < n_pairs and attempts < max_attempts:
            id1, id2 = random.sample(all_item_ids, 2)

            # Check that this isn't a compatible pair
            if (id1, id2) not in compatible_pairs and (id2, id1) not in compatible_pairs:
                pairs.append((id1, id2))

            attempts += 1

        return pairs

    def generate_triplets(
        self,
        n_triplets: Optional[int] = None,
        hard_negative: bool = True
    ) -> Iterator[Tuple[str, str, str]]:
        """
        Generate training triplets (anchor, positive, negative).

        Args:
            n_triplets: Number of triplets to generate
            hard_negative: If True, prefer negatives of the same category

        Yields:
            Tuples of (anchor_id, positive_id, negative_id)
        """
        compatible_pairs = self.get_compatible_pairs()
        if n_triplets is None:
            n_triplets = len(compatible_pairs)

        all_item_ids = list(self.items.keys())
        generated = 0

        while generated < n_triplets:
            # Sample a compatible pair
            anchor_id, positive_id = random.choice(compatible_pairs)
            anchor_item = self.items[anchor_id]

            # Sample a negative
            if hard_negative and random.random() < 0.5:
                # Hard negative: same category, different outfit
                candidates = [
                    id for id in self.items_by_category[anchor_item.category_id]
                    if id != anchor_id and id != positive_id
                ]
                if candidates:
                    negative_id = random.choice(candidates)
                else:
                    negative_id = random.choice(all_item_ids)
            else:
                # Random negative
                negative_id = random.choice(all_item_ids)
                while negative_id == anchor_id or negative_id == positive_id:
                    negative_id = random.choice(all_item_ids)

            yield (anchor_id, positive_id, negative_id)
            generated += 1


class TripletDataset(Dataset):
    """
    PyTorch Dataset for triplet training.

    Combines the Polyvore dataset with feature extractors to provide
    ready-to-use triplets for training the compatibility model.
    """

    def __init__(
        self,
        polyvore: PolyvoreDataset,
        visual_features: Dict[str, np.ndarray],
        color_features: Dict[str, np.ndarray],
        n_triplets: int = 100000,
        hard_negative: bool = True
    ):
        """
        Initialize the triplet dataset.

        Args:
            polyvore: PolyvoreDataset instance
            visual_features: Pre-extracted visual features {item_id: features}
            color_features: Pre-extracted color features {item_id: features}
            n_triplets: Number of triplets per epoch
            hard_negative: Whether to use hard negative mining
        """
        self.polyvore = polyvore
        self.visual_features = visual_features
        self.color_features = color_features
        self.n_triplets = n_triplets
        self.hard_negative = hard_negative

        # Pre-generate triplets for deterministic epoch
        self._regenerate_triplets()

    def _regenerate_triplets(self):
        """Generate new triplets for the epoch."""
        self.triplets = list(self.polyvore.generate_triplets(
            n_triplets=self.n_triplets,
            hard_negative=self.hard_negative
        ))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        anchor_id, positive_id, negative_id = self.triplets[idx]

        # Get features
        def get_features(item_id):
            item = self.polyvore.items[item_id]
            vis = self.visual_features.get(item_id, np.zeros(1280))
            col = self.color_features.get(item_id, np.zeros(15))
            type_idx = item.category_id
            return vis, col, type_idx

        anchor_vis, anchor_col, anchor_type = get_features(anchor_id)
        pos_vis, pos_col, pos_type = get_features(positive_id)
        neg_vis, neg_col, neg_type = get_features(negative_id)

        return (
            torch.tensor(anchor_vis, dtype=torch.float32),
            torch.tensor(anchor_col, dtype=torch.float32),
            torch.tensor(anchor_type, dtype=torch.long),
            torch.tensor(pos_vis, dtype=torch.float32),
            torch.tensor(pos_col, dtype=torch.float32),
            torch.tensor(pos_type, dtype=torch.long),
            torch.tensor(neg_vis, dtype=torch.float32),
            torch.tensor(neg_col, dtype=torch.float32),
            torch.tensor(neg_type, dtype=torch.long),
        )


class PairDataset(Dataset):
    """
    Dataset for evaluating compatibility on item pairs.

    Used for computing AUC-ROC metrics.
    """

    def __init__(
        self,
        polyvore: PolyvoreDataset,
        visual_features: Dict[str, np.ndarray],
        color_features: Dict[str, np.ndarray],
        n_positive: int = 5000,
        n_negative: int = 5000
    ):
        """
        Initialize the pair dataset.

        Args:
            polyvore: PolyvoreDataset instance
            visual_features: Pre-extracted visual features
            color_features: Pre-extracted color features
            n_positive: Number of positive (compatible) pairs
            n_negative: Number of negative (incompatible) pairs
        """
        self.polyvore = polyvore
        self.visual_features = visual_features
        self.color_features = color_features

        # Get compatible pairs
        compatible = polyvore.get_compatible_pairs()
        random.shuffle(compatible)
        self.positive_pairs = compatible[:n_positive]

        # Get incompatible pairs
        self.negative_pairs = polyvore.get_incompatible_pairs(n_negative)

        # Combine with labels
        self.pairs = []
        for id1, id2 in self.positive_pairs:
            self.pairs.append((id1, id2, 1))
        for id1, id2 in self.negative_pairs:
            self.pairs.append((id1, id2, 0))

        random.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        id1, id2, label = self.pairs[idx]

        def get_features(item_id):
            item = self.polyvore.items[item_id]
            vis = self.visual_features.get(item_id, np.zeros(1280))
            col = self.color_features.get(item_id, np.zeros(15))
            type_idx = item.category_id
            return vis, col, type_idx

        vis1, col1, type1 = get_features(id1)
        vis2, col2, type2 = get_features(id2)

        return (
            torch.tensor(vis1, dtype=torch.float32),
            torch.tensor(col1, dtype=torch.float32),
            torch.tensor(type1, dtype=torch.long),
            torch.tensor(vis2, dtype=torch.float32),
            torch.tensor(col2, dtype=torch.float32),
            torch.tensor(type2, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


def extract_all_features(
    polyvore: PolyvoreDataset,
    classifier,
    color_extractor,
    batch_size: int = 32
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract visual and color features for all items in the dataset.
    Memory-efficient version that processes in batches and clears memory.

    Args:
        polyvore: PolyvoreDataset instance
        classifier: GarmentClassifier for visual features
        color_extractor: ColorExtractor for color features
        batch_size: Batch size for feature extraction

    Returns:
        Tuple of (visual_features, color_features) dictionaries
    """
    import gc
    from tqdm import tqdm

    visual_features = {}
    color_features = {}

    item_ids = list(polyvore.items.keys())
    total_items = len(item_ids)

    # Process in batches to manage memory
    BATCH_SIZE = 100  # Process 100 items, then clear memory

    for batch_start in tqdm(range(0, total_items, BATCH_SIZE), desc="Extracting features"):
        batch_end = min(batch_start + BATCH_SIZE, total_items)
        batch_ids = item_ids[batch_start:batch_end]

        for item_id in batch_ids:
            item = polyvore.items[item_id]

            if item.image_path is None:
                visual_features[item_id] = np.zeros(1280)
                color_features[item_id] = np.zeros(15)
                continue

            try:
                # Visual features (with no_grad to save memory)
                with torch.no_grad():
                    vis_feat = classifier.get_features(item.image_path)
                    visual_features[item_id] = vis_feat

                # Color features
                col_feat = color_extractor.get_color_vector(item.image_path)
                color_features[item_id] = col_feat

            except Exception as e:
                visual_features[item_id] = np.zeros(1280)
                color_features[item_id] = np.zeros(15)

        # Clear memory after each batch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return visual_features, color_features
