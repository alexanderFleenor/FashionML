"""
Garment Recognition Module

CNN-based garment classification using pretrained EfficientNet/ResNet.
Fine-tuned on fashion categories: tops, bottoms, dresses, shoes, accessories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
import torchvision.transforms as T


@dataclass
class ClassificationResult:
    """Result of garment classification."""
    category: str
    confidence: float
    all_probabilities: Dict[str, float]


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GarmentClassifier(nn.Module):
    """
    CNN-based garment classification using pretrained EfficientNet-B0.

    Classifies clothing items into categories:
    - tops (t-shirts, shirts, blouses, sweaters)
    - bottoms (pants, jeans, shorts, skirts)
    - dresses (dresses, jumpsuits)
    - shoes (sneakers, boots, heels, sandals)
    - accessories (bags, hats, jewelry, belts)

    Also extracts visual features for downstream compatibility modeling.
    """

    CATEGORIES = ["tops", "bottoms", "dresses", "shoes", "accessories"]

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        feature_extract_only: bool = False
    ):
        """
        Initialize the garment classifier.

        Args:
            backbone: Model architecture ("efficientnet_b0" or "resnet18")
            num_classes: Number of garment categories
            pretrained: Whether to load pretrained ImageNet weights
            feature_extract_only: If True, freeze backbone for feature extraction
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.feature_extract_only = feature_extract_only

        # Load pretrained model from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        if "efficientnet" in backbone:
            self.feature_dim = 1280
        elif "resnet18" in backbone:
            self.feature_dim = 512
        else:
            # Try to infer from model
            self.feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Freeze backbone if only extracting features
        if feature_extract_only:
            self._freeze_backbone()

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.device = get_device()

    def _freeze_backbone(self):
        """Freeze backbone parameters for feature extraction."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features without classification.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Features of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> torch.Tensor:
        """
        Preprocess an image for model input.

        Args:
            image: Input image as numpy array, PIL Image, or path

        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> ClassificationResult:
        """
        Predict the category of a single garment image.

        Args:
            image: Input image

        Returns:
            ClassificationResult with category, confidence, and all probabilities
        """
        self.eval()

        # Preprocess
        tensor = self.preprocess_image(image).to(self.device)

        # Forward pass
        logits = self.forward(tensor)
        probs = F.softmax(logits, dim=1)[0]

        # Get prediction
        confidence, pred_idx = probs.max(dim=0)
        category = self.CATEGORIES[pred_idx.item()]

        # All probabilities
        all_probs = {
            cat: probs[i].item()
            for i, cat in enumerate(self.CATEGORIES)
        }

        return ClassificationResult(
            category=category,
            confidence=confidence.item(),
            all_probabilities=all_probs
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]]
    ) -> List[ClassificationResult]:
        """
        Predict categories for a batch of images.

        Args:
            images: List of input images

        Returns:
            List of ClassificationResult
        """
        self.eval()

        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)

        # Forward pass
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=1)

        # Get predictions
        results = []
        for i in range(len(images)):
            confidence, pred_idx = probs[i].max(dim=0)
            category = self.CATEGORIES[pred_idx.item()]
            all_probs = {
                cat: probs[i, j].item()
                for j, cat in enumerate(self.CATEGORIES)
            }
            results.append(ClassificationResult(
                category=category,
                confidence=confidence.item(),
                all_probabilities=all_probs
            ))

        return results

    @torch.no_grad()
    def get_features(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> np.ndarray:
        """
        Extract visual features from a single image.

        Args:
            image: Input image

        Returns:
            Feature vector as numpy array of shape (feature_dim,)
        """
        self.eval()

        tensor = self.preprocess_image(image).to(self.device)
        features = self.extract_features(tensor)

        return features.cpu().numpy()[0]

    def save(self, path: Union[str, Path]):
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "categories": self.CATEGORIES
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "GarmentClassifier":
        """Load model from saved weights."""
        state = torch.load(path, map_location="cpu")

        model = cls(
            backbone=state["backbone"],
            num_classes=state["num_classes"],
            pretrained=False
        )
        model.load_state_dict(state["model_state_dict"])

        if device is None:
            device = get_device()
        model.to(device)
        model.device = device

        return model


class GarmentClassifierTrainer:
    """
    Trainer for fine-tuning the GarmentClassifier.
    """

    def __init__(
        self,
        model: GarmentClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        epochs: int = 20,
        early_stopping_patience: int = 5
    ):
        """
        Initialize the trainer.

        Args:
            model: GarmentClassifier to train
            train_loader: Training data loader
            val_loader: Validation data loader
            lr: Learning rate
            epochs: Number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.patience = early_stopping_patience

        self.device = model.device
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model. Returns (loss, accuracy)."""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Learning rate scheduling
            if self.val_loader is not None:
                self.scheduler.step(val_acc)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if self.val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if self.val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return self.history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Returns:
            Dictionary with accuracy and per-class metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in test_loader:
            images, labels = batch
            images = images.to(self.device)

            logits = self.model(images)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Overall accuracy
        accuracy = (all_preds == all_labels).mean()

        # Per-class accuracy
        per_class_acc = {}
        for i, cat in enumerate(self.model.CATEGORIES):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[cat] = (all_preds[mask] == all_labels[mask]).mean()
            else:
                per_class_acc[cat] = 0.0

        return {
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc
        }
