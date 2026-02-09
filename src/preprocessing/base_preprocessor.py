"""
Abstract base class for image preprocessing.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class BasePreprocessor(ABC):
    """Abstract base class for image preprocessing."""

    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary for this preprocessor
        """
        self.config = config

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Binary image (0 or 255)
        """
        pass

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: Input image (can be BGR or already grayscale)

        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def resize(
        self,
        image: np.ndarray,
        max_dimension: Optional[int] = None,
        scale: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: Input image
            max_dimension: Maximum dimension (width or height)
            scale: Scale factor (alternative to max_dimension)

        Returns:
            Tuple of (resized image, actual scale factor applied)
        """
        if scale is not None:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            return resized, scale

        if max_dimension is not None:
            h, w = image.shape[:2]
            if max(h, w) > max_dimension:
                if h > w:
                    new_h = max_dimension
                    new_w = int(w * (max_dimension / h))
                    scale_factor = max_dimension / h
                else:
                    new_w = max_dimension
                    new_h = int(h * (max_dimension / w))
                    scale_factor = max_dimension / w
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return resized, scale_factor

        return image, 1.0

    def validate_image(self, image: np.ndarray) -> None:
        """
        Validate image dimensions and format.

        Args:
            image: Input image

        Raises:
            ValueError: If image is invalid
        """
        if image is None:
            raise ValueError("Image is None")

        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}")

        h, w = image.shape[:2]
        if h < 10 or w < 10:
            raise ValueError(f"Image too small: {w}x{h}")

        if h > 10000 or w > 10000:
            raise ValueError(f"Image too large: {w}x{h} (max 10000x10000)")

    def get_preview(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """
        Create a side-by-side preview of original and processed image.

        Args:
            image: Original image
            binary: Binary processed image

        Returns:
            Side-by-side comparison image
        """
        # Convert binary to BGR for display
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Resize original to match binary if needed
        if image.shape[:2] != binary.shape[:2]:
            image = cv2.resize(image, (binary.shape[1], binary.shape[0]))

        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Concatenate horizontally
        preview = np.hstack([image, binary_bgr])
        return preview

    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filter for noise removal.

        Args:
            image: Input image
            kernel_size: Kernel size (must be odd)

        Returns:
            Filtered image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
        return cv2.medianBlur(image, kernel_size)

    def apply_gaussian_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 0
    ) -> np.ndarray:
        """
        Apply Gaussian blur.

        Args:
            image: Input image
            kernel_size: Kernel size (must be odd)
            sigma: Standard deviation (0 = auto-calculate)

        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def apply_morphology(
        self,
        image: np.ndarray,
        operation: str,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations.

        Args:
            image: Binary input image
            operation: 'erode', 'dilate', 'open', 'close'
            kernel_size: Kernel size
            iterations: Number of iterations

        Returns:
            Processed image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown morphology operation: {operation}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
