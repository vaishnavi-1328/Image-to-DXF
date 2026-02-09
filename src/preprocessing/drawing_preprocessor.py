"""
Preprocessor for line drawings and clean images.
"""
import numpy as np
import cv2
from .base_preprocessor import BasePreprocessor


class DrawingPreprocessor(BasePreprocessor):
    """
    Preprocessor optimized for line drawings and clean images.

    Uses simple thresholding and minimal noise removal since
    line drawings typically have clean edges and high contrast.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a line drawing.

        Pipeline:
        1. Convert to grayscale
        2. Apply light median filtering for noise
        3. Otsu's thresholding for binarization
        4. Optional morphological opening to remove small noise

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Binary image (0 or 255)
        """
        self.validate_image(image)

        # Step 1: Convert to grayscale
        gray = self.to_grayscale(image)

        # Step 2: Light median filtering to remove noise
        median_size = self.config.get('median_filter_size', 3)
        if median_size > 0:
            gray = self.apply_median_filter(gray, median_size)

        # Step 3: Otsu's thresholding for automatic binarization
        _, binary = cv2.threshold(
            gray,
            0,  # Threshold value (ignored with Otsu)
            255,  # Maximum value
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 4: Optional morphological opening to remove small dots
        morphology_config = self.config.get('morphology', {})
        kernel_size = morphology_config.get('kernel_size', 2)
        iterations = morphology_config.get('opening_iterations', 1)

        if iterations > 0:
            binary = self.apply_morphology(
                binary,
                'open',
                kernel_size=kernel_size,
                iterations=iterations
            )

        return binary

    def preprocess_with_manual_threshold(
        self,
        image: np.ndarray,
        threshold_value: int = 127
    ) -> np.ndarray:
        """
        Preprocess with manual threshold instead of Otsu.

        Useful when Otsu doesn't work well for a particular image.

        Args:
            image: Input image
            threshold_value: Manual threshold value (0-255)

        Returns:
            Binary image
        """
        self.validate_image(image)

        gray = self.to_grayscale(image)

        median_size = self.config.get('median_filter_size', 3)
        if median_size > 0:
            gray = self.apply_median_filter(gray, median_size)

        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        morphology_config = self.config.get('morphology', {})
        kernel_size = morphology_config.get('kernel_size', 2)
        iterations = morphology_config.get('opening_iterations', 1)

        if iterations > 0:
            binary = self.apply_morphology(
                binary,
                'open',
                kernel_size=kernel_size,
                iterations=iterations
            )

        return binary

    def preprocess_inverted(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess and invert (for white-on-black drawings).

        Args:
            image: Input image

        Returns:
            Binary image with inverted colors
        """
        binary = self.preprocess(image)
        return cv2.bitwise_not(binary)
