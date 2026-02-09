"""
Perspective Correction Module

This module provides functionality to correct perspective distortion in images
taken at an angle. It uses homography transformation to warp the image from a
tilted perspective to an orthogonal (top-down) view.

Typical use case:
- User photographs a physical 2D object (metal sheet, template) at an angle
- User selects 4 corners of the object in the image
- System applies perspective transformation to create undistorted view
- Corrected image proceeds through edge detection and DXF generation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """
    Corrects perspective distortion in images using homography transformation.

    Attributes:
        image: The input image (numpy array in BGR format)
        source_points: User-selected corners [(x,y), ...] in pixel coordinates
        real_width_mm: Real-world width of the object in millimeters
        real_height_mm: Real-world height of the object in millimeters
        config: Configuration dictionary with validation and interpolation settings
    """

    def __init__(self, image: np.ndarray, config: Optional[Dict] = None):
        """
        Initialize the perspective corrector with an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            config: Optional configuration dictionary. If None, uses defaults.
        """
        self.image = image
        self.source_points = None
        self.real_width_mm = None
        self.real_height_mm = None
        self.transformation_matrix = None
        self.corrected_image = None

        # Default configuration
        self.config = {
            'max_output_megapixels': 2.0,
            'interpolation_downsize': 'CUBIC',
            'interpolation_upsize': 'LINEAR',
            'validate_quadrilateral': True,
            'min_angle_degrees': 15,
            'aspect_ratio_tolerance': 0.3,
            'min_transformation_determinant': 0.01,
            'max_transformation_determinant': 100.0,
        }

        if config:
            self.config.update(config)

    def set_source_points(self, points: List[Tuple[float, float]]) -> None:
        """
        Set the 4 corner points selected by the user.

        Args:
            points: List of 4 (x, y) tuples representing corners in order:
                   [top_left, top_right, bottom_right, bottom_left]
                   Coordinates are in pixel space (x=column, y=row)

        Raises:
            ValueError: If points don't form a valid quadrilateral
        """
        if len(points) != 4:
            raise ValueError(f"Expected 4 points, got {len(points)}")

        self.source_points = np.array(points, dtype=np.float32)

        if self.config['validate_quadrilateral']:
            self._validate_quadrilateral()

    def set_output_dimensions(self, width_mm: float, height_mm: float) -> None:
        """
        Set the real-world dimensions of the object.

        Args:
            width_mm: Real width of the object in millimeters
            height_mm: Real height of the object in millimeters

        Raises:
            ValueError: If dimensions are invalid
        """
        if width_mm <= 0 or height_mm <= 0:
            raise ValueError(f"Dimensions must be positive: {width_mm}mm x {height_mm}mm")

        self.real_width_mm = width_mm
        self.real_height_mm = height_mm

    def compute_transform(self) -> np.ndarray:
        """
        Compute the perspective transformation matrix.

        Returns:
            3x3 homography matrix

        Raises:
            ValueError: If source points or dimensions not set, or transformation is invalid
        """
        if self.source_points is None:
            raise ValueError("Source points not set. Call set_source_points() first.")

        if self.real_width_mm is None or self.real_height_mm is None:
            raise ValueError("Output dimensions not set. Call set_output_dimensions() first.")

        # Calculate adaptive output resolution
        output_width_px, output_height_px = self._calculate_output_size()

        # Destination points: perfect rectangle
        dst_points = np.array([
            [0, 0],                                    # top-left
            [output_width_px, 0],                      # top-right
            [output_width_px, output_height_px],       # bottom-right
            [0, output_height_px]                      # bottom-left
        ], dtype=np.float32)

        # Compute homography matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.source_points,
            dst_points
        )

        # Validate transformation matrix
        if self.config['validate_quadrilateral']:
            self._validate_transformation_matrix()

        logger.info(f"Computed transformation matrix for {output_width_px}x{output_height_px}px output")

        return self.transformation_matrix

    def apply_correction(self) -> np.ndarray:
        """
        Apply the perspective correction to the image.

        Returns:
            Corrected image as numpy array (BGR format)

        Raises:
            ValueError: If transformation matrix not computed
        """
        if self.transformation_matrix is None:
            self.compute_transform()

        # Calculate output size
        output_width_px, output_height_px = self._calculate_output_size()

        # Select interpolation method based on sizing
        input_area = self.image.shape[0] * self.image.shape[1]
        output_area = output_width_px * output_height_px
        is_downsizing = output_area < input_area

        interp_method = self.config['interpolation_downsize'] if is_downsizing else self.config['interpolation_upsize']
        flags = getattr(cv2, f"INTER_{interp_method}")

        # Apply transformation
        self.corrected_image = cv2.warpPerspective(
            self.image,
            self.transformation_matrix,
            (output_width_px, output_height_px),
            flags=flags
        )

        logger.info(f"Applied perspective correction: {self.image.shape[:2]} -> {self.corrected_image.shape[:2]}")
        logger.info(f"Interpolation: {interp_method} ({'downsizing' if is_downsizing else 'upsizing'})")

        return self.corrected_image

    def get_calculated_dpi(self) -> float:
        """
        Calculate the DPI of the corrected image based on real-world dimensions.

        This provides accurate pixel-to-mm conversion for downstream processing.

        Returns:
            DPI (dots per inch) of the corrected image

        Raises:
            ValueError: If correction not yet applied
        """
        if self.corrected_image is None:
            raise ValueError("Correction not applied yet. Call apply_correction() first.")

        # DPI = pixels_per_inch = (pixels / mm) * 25.4
        output_width_px = self.corrected_image.shape[1]
        dpi = (output_width_px / self.real_width_mm) * 25.4

        return dpi

    def _calculate_output_size(self) -> Tuple[int, int]:
        """
        Calculate adaptive output resolution based on input size and config.

        Returns:
            (width_px, height_px) tuple
        """
        aspect_ratio = self.real_width_mm / self.real_height_mm
        input_area = self.image.shape[0] * self.image.shape[1]
        max_megapixels = self.config['max_output_megapixels']

        # Target megapixels: use input size if smaller than max, otherwise cap at max
        target_megapixels = min(max_megapixels, input_area / 1_000_000)

        # Calculate dimensions maintaining aspect ratio
        output_height_px = int(np.sqrt(target_megapixels * 1_000_000 / aspect_ratio))
        output_width_px = int(output_height_px * aspect_ratio)

        # Ensure minimum size
        output_width_px = max(100, output_width_px)
        output_height_px = max(100, output_height_px)

        return output_width_px, output_height_px

    def _validate_quadrilateral(self) -> None:
        """
        Validate that the 4 points form a valid convex quadrilateral.

        Raises:
            ValueError: If quadrilateral is invalid
        """
        points = self.source_points

        # Check all points are within image boundaries
        h, w = self.image.shape[:2]
        for i, (x, y) in enumerate(points):
            if x < 0 or x >= w or y < 0 or y >= h:
                raise ValueError(f"Point {i+1} ({x:.1f}, {y:.1f}) is outside image bounds ({w}x{h})")

        # Check for convexity using cross products
        # All cross products should have the same sign for a convex quadrilateral
        def cross_product_z(p1, p2, p3):
            """Calculate z-component of cross product of vectors (p1->p2) and (p2->p3)"""
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        cross_products = [
            cross_product_z(points[0], points[1], points[2]),
            cross_product_z(points[1], points[2], points[3]),
            cross_product_z(points[2], points[3], points[0]),
            cross_product_z(points[3], points[0], points[1])
        ]

        signs = [np.sign(cp) for cp in cross_products if abs(cp) > 1e-6]  # Ignore near-zero

        if len(set(signs)) > 1:
            # Try reversing the order (user clicked clockwise instead of counter-clockwise)
            self.source_points = self.source_points[::-1]
            logger.info("Detected reversed winding order, corrected automatically")
            # Re-validate
            self._validate_quadrilateral()
            return

        if len(signs) < 3:
            raise ValueError("Points are nearly collinear - cannot form a valid quadrilateral")

        # Check minimum area
        area = 0.5 * abs(sum(cross_products))
        if area < 100:  # Minimum 100 square pixels
            raise ValueError(f"Selected region too small ({area:.0f} sq px). Select a larger area.")

        # Check for acute angles
        min_angle = self.config['min_angle_degrees']
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            if angle_deg < min_angle:
                raise ValueError(f"Corner angle too acute ({angle_deg:.1f}°). Minimum: {min_angle}°")

        logger.debug("Quadrilateral validation passed")

    def _validate_transformation_matrix(self) -> None:
        """
        Validate the computed transformation matrix for extreme distortions.

        Raises:
            ValueError: If transformation appears invalid
        """
        det = np.linalg.det(self.transformation_matrix[:2, :2])

        min_det = self.config['min_transformation_determinant']
        max_det = self.config['max_transformation_determinant']

        if det < min_det or det > max_det:
            logger.warning(
                f"Transformation matrix determinant {det:.3f} outside normal range "
                f"[{min_det}, {max_det}]. This may indicate extreme perspective distortion."
            )

    def check_aspect_ratio_mismatch(self) -> Optional[float]:
        """
        Check if the aspect ratio of the selected region differs from provided dimensions.

        Returns:
            Mismatch ratio (0.0-1.0), or None if within tolerance.
            E.g., 0.15 means 15% mismatch.
        """
        if self.source_points is None or self.real_width_mm is None:
            return None

        # Calculate bounding box of selected points
        xs = self.source_points[:, 0]
        ys = self.source_points[:, 1]
        clicked_width = np.max(xs) - np.min(xs)
        clicked_height = np.max(ys) - np.min(ys)
        clicked_aspect = clicked_width / clicked_height

        # Compare to provided dimensions
        provided_aspect = self.real_width_mm / self.real_height_mm

        mismatch = abs(clicked_aspect - provided_aspect) / provided_aspect

        tolerance = self.config['aspect_ratio_tolerance']
        if mismatch > tolerance:
            return mismatch

        return None


def correct_perspective(
    image: np.ndarray,
    corners: List[Tuple[float, float]],
    width_mm: float,
    height_mm: float,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to perform perspective correction in one call.

    Args:
        image: Input image (BGR format)
        corners: List of 4 corner points [(x, y), ...]
        width_mm: Real width in millimeters
        height_mm: Real height in millimeters
        config: Optional configuration dictionary

    Returns:
        Tuple of (corrected_image, calculated_dpi)
    """
    corrector = PerspectiveCorrector(image, config)
    corrector.set_source_points(corners)
    corrector.set_output_dimensions(width_mm, height_mm)
    corrected_image = corrector.apply_correction()
    dpi = corrector.get_calculated_dpi()

    return corrected_image, dpi
