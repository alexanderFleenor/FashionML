"""
Outfit Compatibility Modeling

Siamese network for learning outfit compatibility using triplet loss.
The network learns to embed compatible items close together in a
shared embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TypeAwareProjection(nn.Module):
    """
    Type-aware projection layer that applies different transformations
    based on garment type.

    Different clothing categories have different compatibility rules,
    so we learn separate projections for each type.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_types: int = 5
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
            num_types: Number of garment types
        """
        super().__init__()
        self.num_types = num_types

        # Separate projection for each type
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_types)
        ])

    def forward(self, x: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        """
        Apply type-specific projection.

        Args:
            x: Input features of shape (batch_size, input_dim)
            type_idx: Garment type indices of shape (batch_size,)

        Returns:
            Projected features of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        output_dim = self.projections[0].out_features

        # Initialize output
        output = torch.zeros(batch_size, output_dim, device=x.device)

        # Apply type-specific projections
        for t in range(self.num_types):
            mask = type_idx == t
            if mask.any():
                output[mask] = self.projections[t](x[mask])

        return output


class SiameseCompatibilityNet(nn.Module):
    """
    Siamese network for outfit compatibility prediction.

    Architecture:
    1. Feature fusion: Concatenate visual (CNN) and color features
    2. Shared encoder: MLP to encode fused features
    3. Type-aware projection: Different projection per garment type
    4. L2 normalization: Project onto unit sphere

    Training uses triplet margin loss to push compatible items
    together and incompatible items apart.
    """

    def __init__(
        self,
        visual_feature_dim: int = 1280,
        color_feature_dim: int = 15,
        hidden_dim: int = 512,
        embedding_dim: int = 64,
        num_types: int = 5,
        use_type_aware: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize the compatibility network.

        Args:
            visual_feature_dim: Dimension of visual features from CNN
            color_feature_dim: Dimension of color features (n_colors * 3)
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_types: Number of garment types
            use_type_aware: Whether to use type-aware projections
            dropout: Dropout probability
        """
        super().__init__()

        self.visual_feature_dim = visual_feature_dim
        self.color_feature_dim = color_feature_dim
        self.embedding_dim = embedding_dim
        self.use_type_aware = use_type_aware

        input_dim = visual_feature_dim + color_feature_dim

        # Feature fusion and encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Projection layer
        if use_type_aware:
            self.projection = TypeAwareProjection(
                input_dim=hidden_dim // 2,
                output_dim=embedding_dim,
                num_types=num_types
            )
        else:
            self.projection = nn.Linear(hidden_dim // 2, embedding_dim)

    def encode(
        self,
        visual_features: torch.Tensor,
        color_features: torch.Tensor,
        type_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a single item into the compatibility embedding space.

        Args:
            visual_features: Visual features of shape (batch_size, visual_feature_dim)
            color_features: Color features of shape (batch_size, color_feature_dim)
            type_idx: Garment type indices of shape (batch_size,)

        Returns:
            L2-normalized embeddings of shape (batch_size, embedding_dim)
        """
        # Concatenate features
        x = torch.cat([visual_features, color_features], dim=1)

        # Encode
        hidden = self.encoder(x)

        # Project
        if self.use_type_aware:
            if type_idx is None:
                raise ValueError("type_idx required when use_type_aware=True")
            embedding = self.projection(hidden, type_idx)
        else:
            embedding = self.projection(hidden)

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(
        self,
        anchor_visual: torch.Tensor,
        anchor_color: torch.Tensor,
        anchor_type: torch.Tensor,
        positive_visual: torch.Tensor,
        positive_color: torch.Tensor,
        positive_type: torch.Tensor,
        negative_visual: torch.Tensor,
        negative_color: torch.Tensor,
        negative_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triplet training.

        Args:
            anchor_*: Features for anchor items
            positive_*: Features for positive (compatible) items
            negative_*: Features for negative (incompatible) items

        Returns:
            Tuple of (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encode(anchor_visual, anchor_color, anchor_type)
        positive_emb = self.encode(positive_visual, positive_color, positive_type)
        negative_emb = self.encode(negative_visual, negative_color, negative_type)

        return anchor_emb, positive_emb, negative_emb

    def compute_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.

        Args:
            emb1, emb2: L2-normalized embeddings

        Returns:
            Similarity scores in [-1, 1], higher means more compatible
        """
        return (emb1 * emb2).sum(dim=1)

    def save(self, path: str):
        """Save model weights and config."""
        state = {
            "model_state_dict": self.state_dict(),
            "visual_feature_dim": self.visual_feature_dim,
            "color_feature_dim": self.color_feature_dim,
            "embedding_dim": self.embedding_dim,
            "use_type_aware": self.use_type_aware
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "SiameseCompatibilityNet":
        """Load model from saved weights."""
        state = torch.load(path, map_location="cpu")

        model = cls(
            visual_feature_dim=state["visual_feature_dim"],
            color_feature_dim=state["color_feature_dim"],
            embedding_dim=state["embedding_dim"],
            use_type_aware=state["use_type_aware"]
        )
        model.load_state_dict(state["model_state_dict"])

        if device is None:
            device = get_device()
        model.to(device)

        return model


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss for learning embeddings.

    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))

    where d is Euclidean distance (since embeddings are L2-normalized,
    this is equivalent to 2 - 2*cosine_similarity).
    """

    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Minimum distance between positive and negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        # Euclidean distances (embeddings are L2-normalized)
        pos_dist = (anchor - positive).pow(2).sum(dim=1).sqrt()
        neg_dist = (anchor - negative).pow(2).sum(dim=1).sqrt()

        # Triplet loss
        loss = F.relu(self.margin + pos_dist - neg_dist)

        return loss.mean()


class CompatibilityTrainer:
    """
    Trainer for the SiameseCompatibilityNet.
    """

    def __init__(
        self,
        model: SiameseCompatibilityNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 5e-5,
        epochs: int = 50,
        margin: float = 0.2,
        early_stopping_patience: int = 10
    ):
        """
        Initialize the trainer.

        Args:
            model: SiameseCompatibilityNet to train
            train_loader: DataLoader yielding triplets
            val_loader: Optional validation DataLoader
            lr: Learning rate
            epochs: Maximum training epochs
            margin: Triplet loss margin
            early_stopping_patience: Epochs before early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = early_stopping_patience

        self.device = get_device()
        self.model.to(self.device)

        self.criterion = TripletMarginLoss(margin=margin)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "triplet_accuracy": []
        }

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack triplet batch
            (anchor_vis, anchor_col, anchor_type,
             pos_vis, pos_col, pos_type,
             neg_vis, neg_col, neg_type) = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            # Forward pass
            anchor_emb, pos_emb, neg_emb = self.model(
                anchor_vis, anchor_col, anchor_type,
                pos_vis, pos_col, pos_type,
                neg_vis, neg_col, neg_type
            )

            # Compute loss
            loss = self.criterion(anchor_emb, pos_emb, neg_emb)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (loss, triplet_accuracy)
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.val_loader:
            (anchor_vis, anchor_col, anchor_type,
             pos_vis, pos_col, pos_type,
             neg_vis, neg_col, neg_type) = [x.to(self.device) for x in batch]

            # Forward pass
            anchor_emb, pos_emb, neg_emb = self.model(
                anchor_vis, anchor_col, anchor_type,
                pos_vis, pos_col, pos_type,
                neg_vis, neg_col, neg_type
            )

            # Compute loss
            loss = self.criterion(anchor_emb, pos_emb, neg_emb)
            total_loss += loss.item()

            # Compute triplet accuracy
            # (positive should be closer than negative)
            pos_dist = (anchor_emb - pos_emb).pow(2).sum(dim=1)
            neg_dist = (anchor_emb - neg_emb).pow(2).sum(dim=1)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor_emb.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, triplet_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["triplet_accuracy"].append(triplet_acc)

            # Learning rate scheduling
            if self.val_loader is not None:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Triplet Acc: {triplet_acc:.4f}")

            # Early stopping
            if self.val_loader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return self.history

    @torch.no_grad()
    def compute_auc(self, test_loader: DataLoader) -> float:
        """
        Compute AUC-ROC for compatibility prediction.

        The loader should yield (item1_features, item2_features, label)
        where label is 1 for compatible pairs and 0 for incompatible.

        Returns:
            AUC-ROC score
        """
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        all_scores = []
        all_labels = []

        for batch in test_loader:
            (vis1, col1, type1, vis2, col2, type2, labels) = [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch
            ]

            # Get embeddings
            emb1 = self.model.encode(vis1, col1, type1)
            emb2 = self.model.encode(vis2, col2, type2)

            # Compute similarity scores
            scores = self.model.compute_similarity(emb1, emb2)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return roc_auc_score(all_labels, all_scores)
