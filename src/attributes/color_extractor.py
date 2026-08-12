"""
Color Extraction Module

Extract dominant colors from garment images using k-means clustering
in LAB color space for perceptually accurate color representation.
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import distance
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path


@dataclass
class DominantColor:
    """Represents a dominant color extracted from an image."""
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]
    percentage: float
    name: str

    def to_hex(self) -> str:
        """Convert to hex color code."""
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)


# Basic color definitions in LAB space for color naming
# These are approximate LAB values for common color names
COLOR_DEFINITIONS = {
    "red": (53.23, 80.11, 67.22),
    "orange": (74.94, 23.93, 78.95),
    "yellow": (97.14, -21.56, 94.48),
    "green": (87.74, -86.18, 83.18),
    "blue": (32.30, 79.20, -107.86),
    "purple": (29.78, 58.94, -36.50),
    "pink": (83.59, 24.14, 3.33),
    "brown": (37.99, 13.56, 23.91),
    "black": (0.0, 0.0, 0.0),
    "white": (100.0, 0.0, 0.0),
    "gray": (53.59, 0.0, 0.0),
    "beige": (91.12, 2.17, 14.30),
    "navy": (20.78, 14.88, -46.27),
    "burgundy": (27.08, 39.40, 17.48),
    "olive": (51.87, -12.93, 56.68),
    "teal": (48.25, -28.85, -8.48),
    "coral": (65.57, 45.35, 35.66),
    "cream": (95.76, -0.93, 10.40),
}


class ColorExtractor:
    """
    Extract dominant colors from garment images using k-means clustering.

    Uses LAB color space for perceptually uniform color representation,
    which means Euclidean distances in LAB correspond to perceived
    color differences.
    """

    def __init__(
        self,
        n_colors: int = 5,
        color_space: str = "LAB",
        min_percentage: float = 0.05,
        remove_background: bool = True,
        filter_skin_tones: bool = True
    ):
        """
        Initialize the color extractor.

        Args:
            n_colors: Maximum number of dominant colors to extract
            color_space: Color space for clustering ("LAB" or "HSV")
            min_percentage: Minimum percentage threshold for including a color
            remove_background: Whether to attempt background removal
            filter_skin_tones: Whether to filter out skin tones
        """
        self.n_colors = n_colors
        self.color_space = color_space.upper()
        self.min_percentage = min_percentage
        self.remove_background = remove_background
        self.filter_skin_tones = filter_skin_tones

        # Precompute color name LAB values
        self._color_lab_values = np.array(list(COLOR_DEFINITIONS.values()))
        self._color_names = list(COLOR_DEFINITIONS.keys())

    def _load_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Load and convert image to RGB numpy array."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            return np.array(image)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                return image
        raise ValueError(f"Unsupported image type: {type(image)}")

    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask to exclude background pixels.

        Uses edge detection to find the garment region, avoiding
        exclusion of white garments.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Start with all pixels included
        mask = np.ones_like(gray, dtype=np.uint8) * 255

        # Use edge detection to find the garment region
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=3)

        # Fill from edges to find connected regions
        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Find the largest contour (likely the garment)
            largest_contour = max(contours, key=cv2.contourArea)
            # Only use contour mask if it covers a reasonable area
            contour_area = cv2.contourArea(largest_contour)
            image_area = gray.shape[0] * gray.shape[1]

            if contour_area > image_area * 0.1:  # At least 10% of image
                contour_mask = np.zeros_like(gray)
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                mask = contour_mask

        return mask

    def _is_skin_tone(self, lab: Tuple[float, float, float]) -> bool:
        """Check if a LAB color is likely a skin tone."""
        l, a, b = lab

        # Skin tones typically have:
        # - L between 40-80 (not too bright, not too dark)
        # - a between 5-25 (slightly reddish)
        # - b between 10-40 (yellowish)
        return (40 < l < 80) and (5 < a < 25) and (10 < b < 40)

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB array to LAB color space (OpenCV format: L 0-255, a/b 0-255)."""
        # OpenCV uses BGR, so convert RGB to BGR first
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, 1, 3)
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).astype(np.float32)

    def _opencv_lab_to_standard_lab(self, opencv_lab: np.ndarray) -> np.ndarray:
        """Convert OpenCV LAB (L: 0-255, a/b: 0-255) to standard LAB (L: 0-100, a/b: -128 to 127)."""
        standard_lab = opencv_lab.copy().astype(np.float32)
        standard_lab[..., 0] = opencv_lab[..., 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
        standard_lab[..., 1] = opencv_lab[..., 1] - 128.0  # a: 0-255 -> -128 to 127
        standard_lab[..., 2] = opencv_lab[..., 2] - 128.0  # b: 0-255 -> -128 to 127
        return standard_lab

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB array to RGB color space."""
        if lab.ndim == 1:
            lab = lab.reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.reshape(-1, 3)

    def _get_color_name(self, lab: Tuple[float, float, float]) -> str:
        """Map LAB color to nearest named color."""
        lab_array = np.array(lab).reshape(1, -1)
        distances = distance.cdist(lab_array, self._color_lab_values, metric='euclidean')
        nearest_idx = distances.argmin()
        return self._color_names[nearest_idx]

    def extract(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> List[DominantColor]:
        """
        Extract dominant colors from an image.

        Args:
            image: Input image (numpy array, PIL Image, or path)

        Returns:
            List of DominantColor objects, sorted by percentage (descending)
        """
        # Load image
        img_rgb = self._load_image(image)

        # Create mask if background removal is enabled
        if self.remove_background:
            mask = self._create_mask(img_rgb)
            # Get only masked pixels
            pixels = img_rgb[mask > 0].reshape(-1, 3)
        else:
            pixels = img_rgb.reshape(-1, 3)

        if len(pixels) == 0:
            # Fallback if mask removed everything
            pixels = img_rgb.reshape(-1, 3)

        # Convert to LAB for clustering
        pixels_lab = self._rgb_to_lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)

        # Count unique colors
        unique_colors = np.unique(pixels_lab, axis=0)
        n_unique = len(unique_colors)

        # Handle single-color case without k-means (avoids convergence warning)
        if n_unique == 1:
            centers_lab = unique_colors
            percentages = np.array([1.0])
        else:
            # K-means clustering
            n_clusters = min(self.n_colors, n_unique)

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(pixels_lab)
            centers_lab = kmeans.cluster_centers_

            # Calculate percentages
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels)

        # Create DominantColor objects
        colors = []
        for i, (center_lab_opencv, pct) in enumerate(zip(centers_lab, percentages)):
            if pct < self.min_percentage:
                continue

            # Convert OpenCV LAB center back to RGB
            center_lab_uint8 = center_lab_opencv.astype(np.uint8).reshape(1, 1, 3)
            center_rgb = self._lab_to_rgb(center_lab_uint8)[0]

            # Convert OpenCV LAB to standard LAB for color naming
            standard_lab = self._opencv_lab_to_standard_lab(center_lab_opencv.reshape(1, 3))[0]
            lab_tuple = (float(standard_lab[0]), float(standard_lab[1]), float(standard_lab[2]))
            rgb_tuple = (int(center_rgb[0]), int(center_rgb[1]), int(center_rgb[2]))

            # Filter skin tones if enabled (using standard LAB values)
            if self.filter_skin_tones and self._is_skin_tone(lab_tuple):
                continue

            color_name = self._get_color_name(lab_tuple)

            colors.append(DominantColor(
                rgb=rgb_tuple,
                lab=lab_tuple,
                percentage=float(pct),
                name=color_name
            ))

        # Sort by percentage (descending)
        colors.sort(key=lambda c: c.percentage, reverse=True)

        return colors[:self.n_colors]

    def get_color_vector(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> np.ndarray:
        """
        Get a fixed-size color feature vector for an image.

        Returns a vector of shape (n_colors * 3,) containing LAB values
        of the top dominant colors. Pads with zeros if fewer colors found.

        Args:
            image: Input image

        Returns:
            numpy array of shape (n_colors * 3,)
        """
        colors = self.extract(image)

        # Initialize with zeros
        vector = np.zeros(self.n_colors * 3)

        # Fill with LAB values
        for i, color in enumerate(colors[:self.n_colors]):
            vector[i*3:(i+1)*3] = color.lab

        return vector

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        colors: Optional[List[DominantColor]] = None
    ) -> np.ndarray:
        """
        Create a visualization of the image with its dominant colors.

        Args:
            image: Input image
            colors: Pre-extracted colors, or None to extract them

        Returns:
            Visualization as numpy array (RGB)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        img_rgb = self._load_image(image)

        if colors is None:
            colors = self.extract(image)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Show original image
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Show color palette
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, len(colors))

        for i, color in enumerate(colors):
            rect = Rectangle(
                (0, len(colors) - i - 1),
                color.percentage,
                0.8,
                color=np.array(color.rgb) / 255
            )
            ax2.add_patch(rect)
            ax2.text(
                color.percentage + 0.02,
                len(colors) - i - 0.5,
                f"{color.name} ({color.percentage:.1%})",
                va="center"
            )

        ax2.set_title("Dominant Colors")
        ax2.set_xlabel("Percentage")
        ax2.set_yticks([])
        ax2.set_xlim(0, 1.5)

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        return data[:, :, :3]  # Return RGB only


class ColorHarmonyAnalyzer:
    """
    Analyze color harmony between garment colors.

    Implements basic color theory rules for determining
    if colors work well together.
    """

    @staticmethod
    def rgb_to_hsl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSL."""
        r, g, b = [x / 255.0 for x in rgb]
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        l = (max_c + min_c) / 2

        if max_c == min_c:
            h = s = 0.0
        else:
            d = max_c - min_c
            s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
            if max_c == r:
                h = (g - b) / d + (6.0 if g < b else 0.0)
            elif max_c == g:
                h = (b - r) / d + 2.0
            else:
                h = (r - g) / d + 4.0
            h /= 6.0

        return (h * 360, s * 100, l * 100)

    @staticmethod
    def hue_difference(h1: float, h2: float) -> float:
        """Calculate the smallest angle between two hues."""
        diff = abs(h1 - h2)
        return min(diff, 360 - diff)

    def analyze_harmony(
        self,
        colors1: List[DominantColor],
        colors2: List[DominantColor]
    ) -> dict:
        """
        Analyze color harmony between two sets of colors.

        Args:
            colors1: Dominant colors from first garment
            colors2: Dominant colors from second garment

        Returns:
            Dictionary with harmony analysis
        """
        if not colors1 or not colors2:
            return {"harmony_type": "unknown", "score": 0.5}

        # Get primary colors
        c1 = colors1[0].rgb
        c2 = colors2[0].rgb

        hsl1 = self.rgb_to_hsl(c1)
        hsl2 = self.rgb_to_hsl(c2)

        h1, s1, l1 = hsl1
        h2, s2, l2 = hsl2

        hue_diff = self.hue_difference(h1, h2)

        # Determine harmony type
        if hue_diff < 30:
            harmony_type = "analogous"
            base_score = 0.8
        elif 150 < hue_diff < 210:
            harmony_type = "complementary"
            base_score = 0.85
        elif 90 < hue_diff < 150 or 210 < hue_diff < 270:
            harmony_type = "triadic"
            base_score = 0.75
        else:
            harmony_type = "split-complementary"
            base_score = 0.7

        # Adjust for neutrals (low saturation colors go with everything)
        if s1 < 15 or s2 < 15:
            harmony_type = "neutral-pairing"
            base_score = 0.9

        # Adjust for similar lightness
        lightness_diff = abs(l1 - l2)
        if lightness_diff < 20:
            base_score += 0.05
        elif lightness_diff > 50:
            base_score -= 0.1

        return {
            "harmony_type": harmony_type,
            "score": min(1.0, max(0.0, base_score)),
            "hue_difference": hue_diff,
            "colors": {
                "item1": {"name": colors1[0].name, "rgb": c1},
                "item2": {"name": colors2[0].name, "rgb": c2}
            }
        }
