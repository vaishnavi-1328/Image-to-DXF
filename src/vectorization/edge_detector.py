"""
Edge detection using Canny algorithm.
"""
import numpy as np
import cv2
from typing import Tuple


class EdgeDetector:
    """
    Edge detector using Canny algorithm with configurable parameters.
    """

    def __init__(self, config: dict):
        """
        Initialize edge detector with configuration.

        Args:
            config: Configuration dictionary with edge_detection parameters
        """
        self.config = config
        self.low_threshold = config.get('canny_low_threshold', 50)
        self.high_threshold = config.get('canny_high_threshold', 150)
        self.gaussian_sigma = config.get('gaussian_blur_sigma', 1.0)
        self.morphological_closing = config.get('morphological_closing', True)

    def detect_edges(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Detect edges in binary image using Canny algorithm.

        Args:
            binary_image: Binary preprocessed image (0 or 255)

        Returns:
            Edge map (0 or 255)
        """
        # Step 1: Optional Gaussian blur to reduce noise
        if self.gaussian_sigma > 0:
            blurred = cv2.GaussianBlur(
                binary_image,
                (0, 0),  # Auto-calculate kernel size
                self.gaussian_sigma
            )
        else:
            blurred = binary_image

        # Step 2: Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.low_threshold,
            self.high_threshold
        )

        # Step 3: Optional morphological closing to connect nearby edges
        if self.morphological_closing:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def detect_edges_auto_threshold(
        self,
        binary_image: np.ndarray,
        sigma: float = 0.33
    ) -> np.ndarray:
        """
        Detect edges with automatic threshold calculation.

        Uses median-based automatic threshold (Canny's recommendation).

        Args:
            binary_image: Binary preprocessed image
            sigma: Spread around median (default: 0.33)

        Returns:
            Edge map
        """
        # Calculate median
        v = np.median(binary_image)

        # Calculate thresholds
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        # Apply Gaussian blur
        if self.gaussian_sigma > 0:
            blurred = cv2.GaussianBlur(binary_image, (0, 0), self.gaussian_sigma)
        else:
            blurred = binary_image

        # Canny edge detection
        edges = cv2.Canny(blurred, lower, upper)

        if self.morphological_closing:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def detect_edges_with_params(
        self,
        binary_image: np.ndarray,
        low_threshold: int,
        high_threshold: int,
        blur_sigma: float = 0
    ) -> np.ndarray:
        """
        Detect edges with custom parameters.

        Args:
            binary_image: Binary preprocessed image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
            blur_sigma: Gaussian blur sigma (0 = no blur)

        Returns:
            Edge map
        """
        if blur_sigma > 0:
            blurred = cv2.GaussianBlur(binary_image, (0, 0), blur_sigma)
        else:
            blurred = binary_image

        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        if self.morphological_closing:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def get_suggested_thresholds(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Get suggested Canny thresholds based on image statistics.

        Uses Otsu's method to estimate optimal threshold, then
        derives Canny thresholds from it.

        Args:
            image: Input image (grayscale or binary)

        Returns:
            Tuple of (low_threshold, high_threshold)
        """
        # Calculate Otsu threshold
        otsu_val, _ = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Use Otsu as reference
        # Typical ratio: low = 0.5 * otsu, high = 1.5 * otsu
        low = int(otsu_val * 0.5)
        high = int(otsu_val * 1.5)

        # Clamp to valid range
        low = max(0, min(255, low))
        high = max(0, min(255, high))

        return low, high

    def visualize_edges_on_original(
        self,
        original: np.ndarray,
        edges: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Overlay edges on original image for visualization.

        Args:
            original: Original image (BGR)
            edges: Edge map
            color: Color for edges (BGR), default green

        Returns:
            Visualization image
        """
        # Convert original to BGR if needed
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()

        # Resize if needed
        if vis.shape[:2] != edges.shape[:2]:
            vis = cv2.resize(vis, (edges.shape[1], edges.shape[0]))

        # Overlay edges
        vis[edges > 0] = color

        return vis

    def __repr__(self) -> str:
        return (
            f"EdgeDetector(low={self.low_threshold}, "
            f"high={self.high_threshold}, "
            f"sigma={self.gaussian_sigma})"
        )
