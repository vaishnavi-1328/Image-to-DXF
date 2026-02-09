"""
Preprocessor for photos and complex images.
"""
import numpy as np
import cv2
from .base_preprocessor import BasePreprocessor


class PhotoPreprocessor(BasePreprocessor):
    """
    Preprocessor optimized for photos and complex images.

    Uses aggressive noise removal, contrast enhancement, and
    adaptive thresholding to handle varying lighting and noise.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a photo.

        Pipeline:
        1. Convert to grayscale
        2. Apply bilateral filter for noise reduction (preserves edges)
        3. Apply CLAHE for contrast enhancement
        4. Adaptive Gaussian thresholding
        5. Morphological closing to connect broken lines

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Binary image (0 or 255)
        """
        self.validate_image(image)

        # Step 1: Convert to grayscale
        gray = self.to_grayscale(image)

        # Step 2: Bilateral filter - removes noise while preserving edges
        bilateral_config = self.config.get('bilateral_filter', {})
        d = bilateral_config.get('d', 9)
        sigma_color = bilateral_config.get('sigma_color', 75)
        sigma_space = bilateral_config.get('sigma_space', 75)

        gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

        # Step 3: CLAHE - Contrast Limited Adaptive Histogram Equalization
        clahe_config = self.config.get('clahe', {})
        clip_limit = clahe_config.get('clip_limit', 2.0)
        tile_grid_size = tuple(clahe_config.get('tile_grid_size', [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe.apply(gray)

        # Step 4: Adaptive Gaussian thresholding
        adaptive_config = self.config.get('adaptive_threshold', {})
        block_size = adaptive_config.get('block_size', 11)
        C = adaptive_config.get('C', 2)

        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )

        # Step 5: Morphological closing to connect broken lines
        morphology_config = self.config.get('morphology', {})
        kernel_size = morphology_config.get('kernel_size', 3)
        closing_iterations = morphology_config.get('closing_iterations', 2)

        if closing_iterations > 0:
            binary = self.apply_morphology(
                binary,
                'close',
                kernel_size=kernel_size,
                iterations=closing_iterations
            )

        return binary

    def preprocess_simple_photo(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified preprocessing for less noisy photos.

        Uses Gaussian blur instead of bilateral filter and
        reduces morphological processing.

        Args:
            image: Input image

        Returns:
            Binary image
        """
        self.validate_image(image)

        gray = self.to_grayscale(image)

        # Gaussian blur instead of bilateral
        gray = self.apply_gaussian_blur(gray, kernel_size=5, sigma=1.0)

        # CLAHE for contrast
        clahe_config = self.config.get('clahe', {})
        clip_limit = clahe_config.get('clip_limit', 2.0)
        tile_grid_size = tuple(clahe_config.get('tile_grid_size', [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe.apply(gray)

        # Adaptive thresholding
        adaptive_config = self.config.get('adaptive_threshold', {})
        block_size = adaptive_config.get('block_size', 11)
        C = adaptive_config.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )

        return binary

    def preprocess_with_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing with additional edge enhancement.

        Good for images with very faint lines.

        Args:
            image: Input image

        Returns:
            Binary image
        """
        self.validate_image(image)

        gray = self.to_grayscale(image)

        # Edge enhancement using unsharp mask
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

        # Bilateral filter
        bilateral_config = self.config.get('bilateral_filter', {})
        d = bilateral_config.get('d', 9)
        sigma_color = bilateral_config.get('sigma_color', 75)
        sigma_space = bilateral_config.get('sigma_space', 75)

        unsharp = cv2.bilateralFilter(unsharp, d, sigma_color, sigma_space)

        # CLAHE
        clahe_config = self.config.get('clahe', {})
        clip_limit = clahe_config.get('clip_limit', 2.0)
        tile_grid_size = tuple(clahe_config.get('tile_grid_size', [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(unsharp)

        # Adaptive thresholding
        adaptive_config = self.config.get('adaptive_threshold', {})
        block_size = adaptive_config.get('block_size', 11)
        C = adaptive_config.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )

        # Morphological closing
        morphology_config = self.config.get('morphology', {})
        kernel_size = morphology_config.get('kernel_size', 3)
        closing_iterations = morphology_config.get('closing_iterations', 2)

        if closing_iterations > 0:
            binary = self.apply_morphology(
                binary,
                'close',
                kernel_size=kernel_size,
                iterations=closing_iterations
            )

        return binary
